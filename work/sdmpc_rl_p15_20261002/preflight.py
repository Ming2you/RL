"""Verify paired P15 timing changes and one unchanged physical decision."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_p15_20261002"
BASE = REPO / "results/sdmpc_rl_p13_20261002/pilot1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (ROOT / "preflight.json").exists():
        raise FileExistsError("Preserve preflight")
    if any((p / "STOP").exists() for p in (REPO, ROOT, BASE, BASE.parent,
            REPO / "results/sdmpc_rl_balanced_goal_20260930", REPO / "results/sdmpc_rl_machine_b_20260930")):
        raise RuntimeError("STOP")
    report = read(BASE / "analysis.json")
    assert report["status"] == "authenticated_neural_training_pilot"
    assert read(BASE / "completion.json")["status"] == "completed"
    for rel, digest in read(BASE / "plan.json")["sources"].items():
        assert sha(REPO / rel) == digest, rel
    sys.path.insert(0, str(REPO / "work/sdmpc_rl_multi_20260929"))
    from budget_runtime import boot, DEFAULT_SNAPSHOT
    rt = boot(DEFAULT_SNAPSHOT, 1)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    from strength_policy import StrengthPolicy, MODES
    from bounded_policy import BoundedPolicy
    torch.set_num_threads(1)
    specs = [HERE / "specs" / (mode + ".json") for mode in MODES]
    original_spec = REPO / "results/sdmpc_rl_p13_20261002/fit_v1/policy.json"
    unchanged, modified = 0, 0
    for row in report["rows"]:
        if row["policy"] != "rwbc_v2":
            continue
        slot = BASE / f"{row['scenario']}_s{row['seed']}"
        xp = slot / "experience/rwbc_v2.npz"
        assert sha(xp) == row["experience_sha256"]
        names = read(xp.with_suffix(".json"))["observation_names"]
        actors = [StrengthPolicy.load(p, names) for p in specs]
        with np.load(xp, allow_pickle=False) as arrays:
            for i, (obs, original) in enumerate(zip(arrays["obs"], arrays["action"]), 16):
                for mode, actor in zip(MODES, actors):
                    np.testing.assert_array_equal(actor.memory(), arrays["memory"][i-16])
                    action = actor.act(obs)
                    np.testing.assert_array_equal(actor.memory(), arrays["next_memory"][i-16])
                    assert action.dtype == np.float32 and np.isfinite(action).all() and abs(action).max() <= 1.
                    if not 16 <= i <= 30:
                        np.testing.assert_array_equal(action, original)
                        unchanged += 1
                    else:
                        p = actor.base.base
                        expected = original.copy()
                        if mode in ("half_np", "half_np_restore_nuf"):
                            expected[0] = original[0] * (.5 if original[0] < 0 else 1.)
                        if mode in ("restore_nuf", "half_np_restore_nuf"):
                            expected[1] = min(1., max(0., (float(p.anchor[1])-float(obs[p.anchors[1]]))*10.))
                        np.testing.assert_array_equal(action, expected)
                        modified += 1
    assert unchanged == 675 and modified == 225
    from types import SimpleNamespace
    for mode in MODES:
        for proposed in ([.8,.3],[-.8,.3]):
            for budget, nuf_expected in ((.4,1.),(.7,0.)):
                o=np.zeros(2367,np.float32);o[1]=budget
                holder=SimpleNamespace(previous_step=27, anchor=np.array([0.,.6],np.float32), anchors=[0,1])
                base=SimpleNamespace(base=holder, act=lambda obs:np.array(proposed,np.float32))
                actual=StrengthPolicy(base,mode).act(o)
                expected=np.array(proposed,np.float32)
                if "half_np" in mode and proposed[0]<0:expected[0]/=2.
                if "restore_nuf" in mode:expected[1]=nuf_expected
                np.testing.assert_array_equal(actual,expected)
    # Exercise the exact UTF-8 cached metadata path that failed in P12.
    import p15_worker as worker
    sample = REPO / "results/sdmpc_rl_p12_20261002/wave1/sweet_170_w_s9102/carry.json"
    original_run, original_out = worker.collector.run_to_end, getattr(worker.collector, "OUT", None)
    try:
        worker.collector.OUT = sample.parent
        token = object()
        def capture(*args):
            assert args == (None, None, None, [], 123., 75)
            assert worker.collector.EXPECTED_TTT == read(sample)["ttt"]
            return token
        worker.collector.run_to_end = capture
        assert worker.run_to_end(None, None, None, [], 123.) is token
    finally:
        worker.collector.run_to_end, worker.collector.OUT = original_run, original_out
    env = BudgetEnv(rt, scenario="sweet_170_w", training_seed=9202, guard_mode="physical")
    try:
        checkpoint = BASE / "cache/sweet_170_w_s9202/k16.pt"
        obs = env.restore(torch.load(checkpoint, map_location="cpu", weights_only=False))
        names = env.observer.names
        original = BoundedPolicy.load(original_spec, names)
        ia = original.base.anchors
        ridx = original.base.remaining
        for path in specs:
            a, b = StrengthPolicy.load(path, names), StrengthPolicy.load(path, names)
            for step in range(1, 76):
                o = obs.copy(); o[ridx] = (76-step)/75.
                if step > 16:
                    o[ia] -= np.array([.1, .1], np.float32)
                x, y = a.act(o), b.act(o)
                np.testing.assert_array_equal(x, y)
                if step < 16:
                    np.testing.assert_array_equal(x, np.zeros(2, np.float32))
            for bad in (obs.astype(np.float64), np.full_like(obs, np.nan)):
                try:
                    StrengthPolicy.load(path, names).act(bad)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Bad observation accepted")
            fresh = StrengthPolicy.load(path, names)
            action = fresh.act(obs)
            expected = BoundedPolicy.load(original_spec, names).act(obs)
            if fresh.mode in ("half_np", "half_np_restore_nuf") and expected[0] < 0:
                expected[0] /= 2.
            if fresh.mode in ("restore_nuf", "half_np_restore_nuf"):
                expected[1] = 0.  # The first observation is exactly the latched anchor.
            np.testing.assert_array_equal(action, expected)
            try:
                fresh.act(obs)
            except ValueError:
                pass
            else:
                raise AssertionError("Repeated decision accepted")
        action = StrengthPolicy.load(specs[0], names).act(obs)
        next_obs, reward, terminal, row = env.step(action, "rl", 0., actor_cpu_seconds=0.)
        assert row["control_step"] == 15 and not terminal and next_obs.shape == (2367,)
        compact = worker.collector.probe.compact(row)
        np.testing.assert_array_equal(compact["action"], action)
        prefix = read(BASE / "sweet_170_w_s9202/carry.json")["rows"][14]["total_ttt"]
        assert abs(compact["total_ttt"] - prefix - compact["interval_ttt"]) < 1e-8
        assert reward == -row["interval_ttt"]/100.
        rt["verify"]()
        result = dict(status="passed", scope="strength/recovery contract and one physical interval",
            unchanged_decisions_checked=unchanged, override_formula_checks=modified,
            specs={str(p.relative_to(REPO)): sha(p) for p in specs},
            sources={str(p.relative_to(REPO)): sha(p) for p in HERE.glob("*.py")},
            source_analysis_sha256=sha(BASE / "analysis.json"), checkpoint_sha256=sha(checkpoint),
            utf8_worker_regression=True, torch_threads=torch.get_num_threads(), snapshot=rt["snapshot_identity"])
        with (ROOT / "preflight.json").open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
