"""Authenticate complete P15 branches before recovering an interrupted wave."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/sdmpc_rl_p15_20261002"
OLD = ROOT / "wave1"
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p15_20261002"))
from strength_policy import StrengthPolicy
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p10_20261001"))
from p10_analysis import validate_experience


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "recovery_audit_20261003.json"
    if target.exists() or (OLD / "completion.json").exists():
        raise FileExistsError("Preserve prior audit/completion")
    plan = read(OLD / "plan.json")
    base, cache = Path(plan["source_wave"]), Path(plan["cache"])
    roots = [REPO, ROOT, OLD, base, base.parent,
             REPO / "results/sdmpc_rl_balanced_goal_20260930", REPO / "results/sdmpc_rl_machine_b_20260930"]
    roots += [wave / f"{s}_s{n}" for wave in (OLD, base) for s,n in plan["jobs"]]
    if any((p / "STOP").exists() for p in roots):
        raise RuntimeError("STOP before recovery")
    for rel,digest in plan["sources"].items():
        assert sha(REPO / rel) == digest, rel
    assert sha(base / "analysis.json") == plan["source_analysis_sha256"]
    assert sha(ROOT / "preflight.json") == plan["preflight_sha256"]
    assert sha(ROOT / "protocol.md") == plan["protocol_sha256"]
    for rel,digest in plan["cache_hashes"].items():
        assert sha(cache / rel) == digest
    for rel,digest in plan["imported_controls"].items():
        assert sha(OLD / rel) == sha(base / rel) == digest
    files, complete, missing = {}, [], []
    for scenario,seed in plan["jobs"]:
        slot = OLD / f"{scenario}_s{seed}"
        carry = read(slot / "carry.json")
        assert {k:v for k,v in carry.items() if k != "reused_from"} == read(cache / slot.name / "carry.json")
        assert not carry["truncated"] and [r["control_step"] for r in carry["rows"]] == list(range(75))
        files[str((slot / "carry.json").relative_to(OLD))] = sha(slot / "carry.json")
        expected = {"k16_carry":{"name":"carry"}, **{"k16_p15_actor_"+s["tag"]:s for s in plan["options"]}}
        assert {p.stem for p in (slot / "branches").glob("*.json")} <= set(expected)
        xp_files = set()
        for label,option in expected.items():
            branch = slot / "branches" / (label+".json")
            tag = option.get("tag", "carry")
            xp = slot / "experience" / (tag+".npz")
            meta_path = xp.with_suffix(".json")
            if not branch.exists():
                assert tag != "carry" and not xp.exists() and not meta_path.exists(), "Orphan experience requires separate audit"
                missing.append(dict(scenario=scenario,seed=seed,policy=tag))
                continue
            b = read(branch); rows = b["rows"]
            assert b["option"] == option and b["label"] == label
            assert [r["control_step"] for r in rows] == list(range(15,75))
            assert b["ttt"] == rows[-1]["total_ttt"] and b["carry_ttt"] == carry["ttt"]
            assert b["prefix_ttt"] == carry["rows"][14]["total_ttt"]
            assert abs(b["prefix_ttt"] + sum(r["interval_ttt"] for r in rows) - b["ttt"]) < 1e-7
            if tag == "carry":
                assert b["ttt"] == carry["ttt"]
            meta = read(meta_path)
            assert sha(xp) == meta["file_sha256"]
            with np.load(xp, allow_pickle=False) as arrays:
                validate_experience(arrays, meta, b, carry)
                if tag != "carry":
                    actor = StrengthPolicy.load(REPO / option["spec_path"], meta["observation_names"])
                    for obs,action,mem,next_mem in zip(arrays["obs"],arrays["action"],arrays["memory"],arrays["next_memory"]):
                        np.testing.assert_array_equal(actor.memory(),mem)
                        np.testing.assert_array_equal(actor.act(obs),action)
                        np.testing.assert_array_equal(actor.memory(),next_mem)
            for p in (branch,xp,meta_path):
                files[str(p.relative_to(OLD))] = sha(p)
            xp_files.update((xp.name,meta_path.name))
            complete.append(dict(scenario=scenario,seed=seed,policy=tag))
        assert {p.name for p in (slot / "experience").iterdir()} == xp_files, "Unexpected partial file"
    assert len(complete) == 16 and len(missing) == 4
    assert {(r["scenario"],r["policy"]) for r in missing} == {
        ("sweet_190_w","restore_nuf"), *( ("sweet_155_w",p["tag"]) for p in plan["options"] )}
    report = dict(status="authenticated_interrupted_p15", checked=time.time(), old_wave=str(OLD),
                  old_plan_sha256=sha(OLD / "plan.json"), old_events_sha256=sha(OLD / "events.jsonl"),
                  complete=complete, missing=missing, preserved_files=files,
                  complete_candidates=11, reused_controls=5, validated_transitions=960,
                  source_count=len(plan["sources"]), canonical_eligible=False,
                  interruption_cause="unknown; processes absent, no completion/STOP/traceback; OS boot predates launch")
    with target.open("x",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ("complete","preserved_files")},indent=2))


if __name__ == "__main__":
    main()
