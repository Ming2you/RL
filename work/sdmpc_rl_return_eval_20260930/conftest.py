"""Synthetic ledger fixtures; no canonical profile ever enters a physical env."""
import ast
import copy
import uuid
from dataclasses import dataclass, field, asdict
from types import SimpleNamespace as NS
import pytest
import support as s
import checks
import worker
import readout
import preflight
from policy import authenticate

ACTOR, AUTH = authenticate()
SCHEMA = AUTH["contract"]["observation_schema"]
HELPERS = checks.helpers()

# Reuse just the proven synthetic ledger definitions, never its old fixtures or
# module-level authentication. This namespace is private to the new test module.
path = s.WAVE_SOURCE / "conftest.py"
s.check_hash(path, "bb93f70e1843438526fcb5a48c6b2bcefd908b41e01bd30dce672681a0507c47")
tree = ast.parse(path.read_text(encoding="utf-8-sig"))
assignments = [n for n in tree.body if isinstance(n, ast.Assign) and
               isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("CONFIG", "OPTIONS")]
exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), globals())
w = s.old
s.reuse("conftest.py", ("State", "FakeEnv"), globals())
WaveFakeEnv = FakeEnv


class FakeEnv(WaveFakeEnv):
    def __init__(self, rt, scenario, training_seed, guard_mode):
        assert training_seed is None and guard_mode == "physical"
        self.training_seed = training_seed
        self.profile_hash = AUTH["profiles"][scenario]
        self.protocol = dict(scenario=scenario)

    def contract(self):
        return copy.deepcopy(self.contract_value)


@pytest.fixture(autouse=True)
def forbid_real(monkeypatch):
    import budget_runtime
    import budget_env
    import run_budget

    def forbidden(*args, **kwargs):
        pytest.fail("Actual simulation or optimizer construction is forbidden")

    for mod, name in ((s, "boot"), (s.old, "boot"), (budget_runtime, "boot"), (run_budget, "boot")):
        monkeypatch.setattr(mod, name, forbidden)
    for name in ("__init__", "reset", "step", "restore"):
        monkeypatch.setattr(budget_env.BudgetEnv, name, forbidden)
    monkeypatch.setattr(s.torch.optim.Adam, "__init__", forbidden)
    monkeypatch.setattr(s.torch.optim.Adam, "step", forbidden)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    auth = copy.deepcopy(AUTH)
    auth["contract"]["environment_contract"]["config_sha256"] = s.digest(CONFIG)
    auth["contract"]["environment_contract"]["options_sha256"] = s.digest(OPTIONS)
    root = s.HERE / "_t" / uuid.uuid4().hex[:8]
    root.mkdir(parents=True)
    monkeypatch.setattr(s, "REPO", root)
    monkeypatch.setattr(s, "GOAL", root / "goal")
    monkeypatch.setattr(s, "ROOT", root / "goal/canonical")
    source = {"synthetic": "unchanged"}
    monkeypatch.setattr(s, "sources", lambda: source.copy())
    monkeypatch.setattr(s, "import_boundary", lambda **kwargs: None)
    monkeypatch.setattr(s, "process_identity", lambda: dict(pid=123, created=456, parent_pid=12,
        command=["synthetic"], executable="synthetic"))
    for mod in (worker, readout):
        monkeypatch.setattr(mod, "authenticate", lambda: (ACTOR, copy.deepcopy(auth)))
    cfg = NS(network=NS(total_ramp_capacity=6000.))
    monkeypatch.setattr(s, "boot", lambda *args: dict(cfg=cfg, options=OPTIONS,
        rc=NS(to_plain_dict=lambda value: copy.deepcopy(CONFIG if value is cfg else OPTIONS))))
    monkeypatch.setattr(s, "observation_schema", lambda env: copy.deepcopy(SCHEMA))
    monkeypatch.setattr(worker, "BudgetEnv", FakeEnv)
    FakeEnv.contract_value = dict(auth["contract"]["environment_contract"]["coordinator"],
        cfg=CONFIG, options=OPTIONS, observation=SCHEMA["normalization"], source_snapshot=auth["physical_snapshot"])
    FakeEnv.resets = FakeEnv.steps = FakeEnv.restores = 0
    baseline = [dict(scenario=scenario, ttt=s.BASE_TTT[scenario], warmup_ttt=5.,
        interval_ttt=[(s.BASE_TTT[scenario]-5.)/75]*75, profile_sha256=auth["profiles"][scenario])
        for scenario in s.SCENARIOS]
    admitted = dict(format=s.FORMAT+"-preflight", status="authenticated", spec=s.SPEC, spec_sha256=s.SPEC_SHA,
        sources=source, sources_sha256=s.digest(source), authentication=auth,
        centers=dict(rows=baseline, files={}), input_sha256={}, parity=dict(observations=375, bitexact=True))
    def live(value):
        if value != admitted or s.sources() != source:
            raise ValueError("Synthetic preflight changed")
    monkeypatch.setattr(preflight, "live", live)
    preflight_path, receipt_path = root / "preflight.json", root / "synthetic-review.json"
    s.save(preflight_path, admitted)
    s.save(receipt_path, dict(format=s.FORMAT+"-parent-review", decision="approved_for_physical_evaluation",
        source_sha256=s.digest(source), spec_sha256=s.SPEC_SHA, preflight_sha256=s.file_hash(preflight_path),
        model_sha256=s.MODEL_SHA, physical_contract_sha256=s.CONTRACT_SHA,
        reviewer="synthetic test only", parent_admission="synthetic test only"))
    def request(scenario=None, resume=False, limit=None):
        return NS(scenario=scenario or s.SCENARIOS[0], resume=resume, max_new_steps=limit,
                  preflight=preflight_path, review_receipt=receipt_path)
    return NS(auth=auth, actor=ACTOR, source=source, patch=monkeypatch, request=request,
              preflight=preflight_path, receipt=receipt_path, admitted=admitted, env=FakeEnv)

