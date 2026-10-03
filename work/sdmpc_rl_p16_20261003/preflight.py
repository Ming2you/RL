"""Check P16 action contracts and actual unchanged physical execution."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_p16_20261003"
BASE = REPO / "results/sdmpc_rl_p13_20261002/pilot1"
P15 = REPO / "results/sdmpc_rl_p15_20261002/wave2_recovery"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (ROOT / "preflight.json").exists():
        raise FileExistsError("Preserve preflight")
    jobs = read(BASE / "plan.json")["jobs"]
    stop_roots = [REPO, ROOT, BASE, BASE.parent, P15, P15.parent,
                 REPO / "results/sdmpc_rl_balanced_goal_20260930",
                 REPO / "results/sdmpc_rl_machine_b_20260930"]
    stop_roots += [wave / f"{s}_s{n}" for wave in (BASE, P15) for s, n in jobs]
    if any((p / "STOP").exists() for p in stop_roots):
        raise RuntimeError("STOP")
    report = read(BASE / "analysis.json")
    assert report["status"] == "authenticated_neural_training_pilot"
    assert read(BASE / "completion.json")["status"] == "completed"
    assert read(P15 / "completion.json")["status"] == "completed"
    assert read(P15 / "analysis.json")["status"] == "authenticated_paired_strength_recovery_diagnostic"
    for wave in (BASE, P15):
        for rel, digest in read(wave / "plan.json")["sources"].items():
            assert sha(REPO / rel) == digest, rel
    sys.path.insert(0, str(REPO / "work/sdmpc_rl_multi_20260929"))
    from budget_runtime import boot, DEFAULT_SNAPSHOT
    rt = boot(DEFAULT_SNAPSHOT, 1)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    from direction_policy import DirectionPolicy, MODES
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
        actors = [DirectionPolicy.load(p, names) for p in specs]
        with np.load(xp, allow_pickle=False) as arrays:
            for step, (obs, original) in enumerate(zip(arrays["obs"], arrays["action"]), 16):
                for mode, actor in zip(MODES, actors):
                    np.testing.assert_array_equal(actor.memory(), arrays["memory"][step-16])
                    action = actor.act(obs)
                    np.testing.assert_array_equal(actor.memory(), arrays["next_memory"][step-16])
                    assert action.dtype == np.float32 and np.isfinite(action).all() and abs(action).max() <= 1.
                    assert action[1] == original[1]
                    if not 16 <= step <= 30:
                        np.testing.assert_array_equal(action, original)
                        unchanged += 1
                    else:
                        p = actor.base.base
                        expected = original.copy()
                        expected[0] = 0. if mode == "np_neutral" else min(abs(float(original[0])),
                            max(0., min(1., (float(p.anchor[0])+.1-float(obs[p.anchors[0]]))*20.)))
                        np.testing.assert_array_equal(action, expected)
                        assert action[0] >= 0.
                        if mode == "np_relax_100" and action[0] > 0.:
                            assert float(obs[p.anchors[0]]) + float(action[0])/20. <= float(p.anchor[0])+.1+1e-7
                        modified += 1
    assert unchanged == 450 and modified == 150
    # Check signs, available headroom, existing overshoot and unchanged NUF.
    from types import SimpleNamespace
    for mode in MODES:
        for proposed in ([.8,.3],[-.8,.3]):
            for budget, expected_relax in ((.9,.8),(1.,.8),(1.075,.5),(1.1,0.),(1.2,0.)):
                o=np.zeros(2367,np.float32);o[0]=budget
                holder=SimpleNamespace(previous_step=20, anchor=np.array([1.,.6],np.float32), anchors=[0,1])
                base=SimpleNamespace(base=holder, act=lambda obs:np.array(proposed,np.float32))
                actual=DirectionPolicy(base,mode).act(o)
                expected=np.array([0. if mode=="np_neutral" else expected_relax,.3],np.float32)
                np.testing.assert_allclose(actual,expected,rtol=0.,atol=1e-6)
    import p16_worker as worker
    # Use a cached file containing a non-ASCII reused_from path.
    sample=REPO / "results/sdmpc_rl_p12_20261002/wave1/sweet_170_w_s9102/carry.json"
    original_run, original_out=worker.collector.run_to_end,getattr(worker.collector,"OUT",None)
    try:
        worker.collector.OUT=sample.parent
        token=object()
        def capture(*args):
            assert args==(None,None,None,[],123.,75)
            assert worker.collector.EXPECTED_TTT==read(sample)["ttt"]
            return token
        worker.collector.run_to_end=capture
        assert worker.run_to_end(None,None,None,[],123.) is token
    finally:
        worker.collector.run_to_end,worker.collector.OUT=original_run,original_out
    env=BudgetEnv(rt,scenario="sweet_170_w",training_seed=9202,guard_mode="physical")
    try:
        checkpoint=BASE / "cache/sweet_170_w_s9202/k16.pt"
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        obs=env.restore(saved)
        names=env.observer.names
        original=BoundedPolicy.load(original_spec,names)
        ridx=original.base.remaining;indices=original.base.anchors
        for path in specs:
            a,b=DirectionPolicy.load(path,names),DirectionPolicy.load(path,names)
            for step in range(1,76):
                o=obs.copy();o[ridx]=(76-step)/75.
                if step>16:o[indices]+=np.array([.11,-.1],np.float32)
                x,y=a.act(o),b.act(o)
                np.testing.assert_array_equal(x,y)
                if step<16:np.testing.assert_array_equal(x,np.zeros(2,np.float32))
                if step==31:assert x[0]<0. and x[1]>0.  # Both-budget return still handles NP overshoot.
            for bad in (obs.astype(np.float64),np.full_like(obs,np.nan)):
                try:DirectionPolicy.load(path,names).act(bad)
                except ValueError:pass
                else:raise AssertionError("Bad observation accepted")
            fresh=DirectionPolicy.load(path,names);fresh.act(obs)
            try:fresh.act(obs)
            except ValueError:pass
            else:raise AssertionError("Repeated decision accepted")
        actual_intervals=[]
        for path in specs:
            obs=env.restore(saved)
            action=DirectionPolicy.load(path,names).act(obs)
            next_obs,reward,terminal,row=env.step(action,"rl",0.,actor_cpu_seconds=0.)
            compact=worker.collector.probe.compact(row)
            assert row["control_step"]==15 and not terminal and next_obs.shape==(2367,)
            np.testing.assert_array_equal(compact["action"],action)
            carry=read(BASE / "sweet_170_w_s9202/carry.json")
            assert abs(compact["total_ttt"]-carry["rows"][14]["total_ttt"]-compact["interval_ttt"])<1e-8
            assert reward==-row["interval_ttt"]/100.
            if path.stem=="np_neutral":
                np.testing.assert_array_equal(action,np.zeros(2,np.float32))
                for key,value in carry["rows"][15].items():
                    if key != "decision_wall_seconds":
                        assert compact[key]==value, key
            else:
                assert action[0]>0. and action[1]==0.
            actual_intervals.append(dict(policy=path.stem,action=action.tolist(),row=compact))
        rt["verify"]()
        result=dict(status="passed",scope="NP direction contract, exact carry interval and positive-NP physical interval",
            unchanged_decisions_checked=unchanged,override_formula_checks=modified,
            specs={str(p.relative_to(REPO)):sha(p) for p in specs},
            sources={str(p.relative_to(REPO)):sha(p) for p in HERE.glob("*.py")},
            source_analysis_sha256=sha(BASE / "analysis.json"),checkpoint_sha256=sha(checkpoint),
            decision_evidence={str((P15 / "analysis.json").relative_to(REPO)):sha(P15 / "analysis.json")},
            utf8_worker_regression=True,torch_threads=torch.get_num_threads(),snapshot=rt["snapshot_identity"],
            physical_intervals=actual_intervals)
        with (ROOT / "preflight.json").open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
        print(json.dumps(dict(status=result["status"],unchanged=unchanged,modified=modified,
                             physical_intervals=len(actual_intervals),source_files=len(result["sources"])),indent=2))
    finally:
        env.close()


if __name__=="__main__":main()
