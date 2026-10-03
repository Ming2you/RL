"""P12 bound invariants, fitting support and one paired physical interval."""
import hashlib
import json
import os
from pathlib import Path
import sys

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ROOT=REPO / "results/sdmpc_rl_p12_20261002"
BASE=REPO / "results/sdmpc_rl_p11_20261002/pilot1"


def main():
    if (ROOT / "preflight.json").exists(): raise FileExistsError("Preserve P12 preflight")
    if any((r / "STOP").exists() for r in (REPO,ROOT,BASE,BASE.parent)):
        raise RuntimeError("STOP")
    # Boot before loading Torch/NumPy policy modules into the physical process.
    sys.path.insert(0,str(REPO / "work/sdmpc_rl_multi_20260929"))
    from budget_runtime import boot,DEFAULT_SNAPSHOT
    rt=boot(DEFAULT_SNAPSHOT,1)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    from bounded_policy import BoundedPolicy,project,MODES
    from neural_policy import NeuralPolicy
    torch.set_num_threads(1)
    manifest=json.loads((REPO / "work/sdmpc_rl_p11_20261002/fit_manifest.json").read_text())
    tested=0
    for row in manifest["rows"]:
        slot=REPO / "results/sdmpc_rl_p10_20261001/wave2_recovery" / f"{row['scenario']}_s{row['seed']}"
        path=slot / "experience" / (row["policy"]+".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row["experience_sha256"]:
            raise ValueError("Fitting evidence changed")
        meta=json.loads(path.with_suffix(".json").read_text());names=meta["observation_names"]
        indices=[names.index("memory/action_anchor/0"),names.index("memory/action_anchor/1")]
        with np.load(path,allow_pickle=False) as arrays:
            anchor=arrays["obs"][0,indices]
            for i in range(15):
                for mode in MODES:
                    np.testing.assert_allclose(project(arrays["action"][i],arrays["obs"][i],anchor,indices,16+i,mode),
                        arrays["action"][i],rtol=0,atol=2e-6)
                tested+=1
    assert tested==600
    spec_paths=[HERE / "specs" / (mode+".json") for mode in MODES]
    env=BudgetEnv(rt,scenario="sweet_155_w",training_seed=9101,guard_mode="physical")
    try:
        cache=BASE / "cache/sweet_155_w_s9101/k16.pt"
        obs=env.restore(torch.load(cache,map_location="cpu",weights_only=False));names=env.observer.names
        raw_spec=REPO / "results/sdmpc_rl_p11_20261002/fit_v1/policy.json"
        raw=NeuralPolicy.load(raw_spec,names);actors=[BoundedPolicy.load(p,names) for p in spec_paths]
        ridx=list(names).index("memory/remaining/0");ii=raw.anchors
        for step in range(1,76):
            o=obs.copy();o[ridx]=(76-step)/75.
            if step>16:o[ii]-=np.array([.1,.1],np.float32)
            original=raw.act(o)
            for mode,actor in zip(MODES,actors):
                action=actor.act(o)
                assert action.dtype==np.float32 and np.isfinite(action).all() and np.abs(action).max()<=1.
                if step<16 or step>30:np.testing.assert_array_equal(action,original)
                else:
                    assert action[1]>=0
                    if mode=="nuf_nonnegative":assert action[0]==original[0]
                    else:
                        gap=(actor.base.anchor.astype(float)-o[ii].astype(float))*[20.,10.]
                        assert action[1]<=max(0.,gap[1])+1e-6
                        if step<=25:assert action[0]<=0
                        elif action[0]>0:assert action[0]<=max(0.,gap[0])+1e-6
        # Explicit regression: unsupported signs/overshoot at the latched anchor.
        label=np.array([.9,-.2],np.float32)
        np.testing.assert_array_equal(project(label,obs,obs[ii],ii,20,"nuf_nonnegative"),[np.float32(.9),0.])
        np.testing.assert_array_equal(project(label,obs,obs[ii],ii,20,"fitting_bounds"),[0.,0.])
        np.testing.assert_array_equal(project(np.ones(2,np.float32),obs,obs[ii],ii,28,"fitting_bounds"),[0.,0.])
        original=NeuralPolicy.load(raw_spec,names).act(obs)
        action=BoundedPolicy.load(spec_paths[0],names).act(obs)
        np.testing.assert_array_equal(action,original)
        previous=json.loads((BASE / "sweet_155_w_s9101/branches/k16_p11_actor_rwbc_v1.json").read_text())["rows"][0]
        next_obs,reward,terminal,row=env.step(action,"rl",0.,actor_cpu_seconds=0.)
        assert row["control_step"]==15 and not terminal and row["total_ttt"]==previous["total_ttt"]
        assert reward==-row["interval_ttt"]/100.
        rt["verify"]()
        report=dict(status="passed",scope="action_bound_contract_and_one_interval_only",fitting_labels_unchanged=tested,
            tests=["two bounds preserve fitting labels","unsupported signs removed","anchor overshoot capped",
                "unchanged outside window","bounded float32 actions","identical first physical interval"],
            specs={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in spec_paths},
            torch_threads=torch.get_num_threads(),snapshot=rt["snapshot_identity"])
        ROOT.mkdir(exist_ok=True)
        with (ROOT / "preflight.json").open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2))
    finally:env.close()


if __name__=="__main__":main()
