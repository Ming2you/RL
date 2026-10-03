"""P17 learned-return contracts, zero carry identity and old-profile physics."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO / "results/sdmpc_rl_p17_20261003"
BASE=REPO / "results/sdmpc_rl_p13_20261002/pilot1"


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (ROOT / "preflight.json").exists():raise FileExistsError("Preserve preflight")
    if any((p / "STOP").exists() for p in (REPO,ROOT,ROOT / "fit_v1",BASE,BASE.parent,
        REPO / "results/sdmpc_rl_balanced_goal_20260930",REPO / "results/sdmpc_rl_machine_b_20260930")):
        raise RuntimeError("STOP")
    fit=read(ROOT / "fit_v1/completion.json")
    assert fit["status"]=="fit_passed" and fit["optimizer_updates"]==2000
    assert fit["labels"]==1200 and fit["full_training_trajectories"]==20
    for rel,h in read(ROOT / "fit_v1/plan.json")["source_pins"].items():assert sha(REPO / rel)==h
    sys.path.insert(0,str(REPO / "work/sdmpc_rl_multi_20260929"))
    from budget_runtime import boot,DEFAULT_SNAPSHOT
    rt=boot(DEFAULT_SNAPSHOT,1)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    from neural_policy import NeuralPolicy
    from bounded_policy import BoundedPolicy,project
    torch.set_num_threads(1)
    # Explicit examples catch incorrect directions, caps and forced recovery.
    anchor=np.array([1.,.6],np.float32);indices=[0,1]
    examples=[(20,[1.,.5],[.8,-.8],[.8,0.]),
              (20,[1.075,.5],[.8,.8],[.5,.8]),
              (20,[1.2,.5],[-.8,.8],[-.8,.8]),
              (20,[1.2,.7],[.8,-.8],[0.,-.8]),
              (31,[.99,.59],[-.8,-.8],[0.,0.]),
              (31,[.99,.59],[.8,.8],[.2,.1]),
              (75,[1.02,.65],[-.8,-.8],[-.4,-.5]),
              (31,[.5,.1],[0.,0.],[0.,0.])]
    for step,budgets,action,expected in examples:
        o=np.zeros(2367,np.float32);o[indices]=budgets
        np.testing.assert_allclose(project(np.array(action,np.float32),o,anchor,indices,step),expected,rtol=0,atol=2e-6)
    manifest=read(HERE / "fit_manifest.json")
    labels=0;projected_differences=0
    for row in manifest["rows"]:
        path=REPO / row["experience_path"]
        assert sha(path)==row["experience_sha256"]
        names=read(path.with_suffix(".json"))["observation_names"]
        ia=[names.index("memory/action_anchor/0"),names.index("memory/action_anchor/1")]
        with np.load(path,allow_pickle=False) as a:
            assert a["action"].shape==(60,2) and np.isfinite(a["action"]).all() and abs(a["action"]).max()<=1.
            for step,(o,act) in enumerate(zip(a["obs"],a["action"]),16):
                p=project(act,o,a["obs"][0,ia],ia,step)
                projected_differences+=int(np.max(np.abs(p-act))>1e-5)
                labels+=1
    assert labels==1200
    # The model must not require incompatible demonstrations to fit a return phase.
    assert projected_differences==0, "Selected demonstration actions exceed the deployed support"
    import p17_worker as worker
    sample=REPO / "results/sdmpc_rl_p12_20261002/wave1/sweet_170_w_s9102/carry.json"
    old_run,old_out=worker.collector.run_to_end,getattr(worker.collector,"OUT",None)
    try:
        worker.collector.OUT=sample.parent;token=object()
        def capture(*args):
            assert worker.collector.EXPECTED_TTT==read(sample)["ttt"]
            return token
        worker.collector.run_to_end=capture
        assert worker.run_to_end(None,None,None,[],123.) is token
    finally:worker.collector.run_to_end,worker.collector.OUT=old_run,old_out
    env=BudgetEnv(rt,scenario="sweet_170_w",training_seed=9202,guard_mode="physical")
    try:
        cache=BASE / "cache/sweet_170_w_s9202/k16.pt"
        saved=torch.load(cache,map_location="cpu",weights_only=False)
        obs=env.restore(saved);names=env.observer.names
        spec=ROOT / "fit_v1/policy.json"
        a,b=BoundedPolicy.load(spec,names),BoundedPolicy.load(spec,names)
        ridx=a.base.remaining;ia=a.base.anchors
        payload=torch.load(ROOT / "fit_v1/model.pt",map_location="cpu",weights_only=True)
        for key in ("4.weight","4.bias"):payload["state_dict"][key].zero_()
        zero=BoundedPolicy(NeuralPolicy(payload,names))
        for step in range(1,76):
            o=obs.copy();o[ridx]=(76-step)/75.
            if step>16:o[ia]+=np.array([.11 if step%2 else -.1,-.1 if step%2 else .1],np.float32)
            x,y=a.act(o),b.act(o)
            np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(zero.act(o),np.zeros(2,np.float32))
            if step<16:np.testing.assert_array_equal(x,np.zeros(2,np.float32))
            if step>=16:
                gap=(a.base.anchor.astype(float)-o[ia].astype(float))*[20.,10.]
                for i in (range(2) if step>30 else (1,)):
                    assert max(-1.,min(0.,gap[i]))-1e-6<=x[i]<=min(1.,max(0.,gap[i]))+1e-6
                if step<=30 and x[0]>0.:
                    assert o[ia[0]]+float(x[0])/20.<=float(a.base.anchor[0])+.1+1e-7
            assert x.dtype==np.float32 and np.isfinite(x).all() and abs(x).max()<=1.
        for bad in (obs.astype(np.float64),np.full_like(obs,np.nan)):
            try:BoundedPolicy.load(spec,names).act(bad)
            except ValueError:pass
            else:raise AssertionError("Bad observation accepted")
        fresh=BoundedPolicy.load(spec,names);fresh.act(obs)
        try:fresh.act(obs)
        except ValueError:pass
        else:raise AssertionError("Repeated decision accepted")
        intervals=[]
        for mode in ("zero_network","fitted_network"):
            o=env.restore(saved)
            policy=BoundedPolicy(NeuralPolicy(payload,names)) if mode=="zero_network" else BoundedPolicy.load(spec,names)
            action=policy.act(o)
            next_obs,reward,terminal,row=env.step(action,"rl",0.,actor_cpu_seconds=0.)
            compact=worker.original_compact(row)
            assert row["control_step"]==15 and not terminal and next_obs.shape==(2367,)
            assert reward==-row["interval_ttt"]/100.
            if mode=="zero_network":
                carry=read(BASE / "sweet_170_w_s9202/carry.json")
                for key,value in carry["rows"][15].items():
                    if key!="decision_wall_seconds":assert compact[key]==value,key
            intervals.append(dict(mode=mode,action=action.tolist(),row=compact))
        rt["verify"]()
        report=dict(status="passed",spec_sha256=sha(spec),model_sha256=fit["model_sha256"],
            fitting_labels_checked=labels,demonstration_actions_outside_support=projected_differences,
            zero_carry_decisions=75,utf8_worker_regression=True,physical_intervals=intervals,
            torch_threads=torch.get_num_threads(),snapshot=rt["snapshot_identity"],
            source_pins={str(p.relative_to(REPO)):sha(p) for p in HERE.glob("*.py")})
        with (ROOT / "preflight.json").open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
        print(json.dumps(dict(status=report["status"],labels=labels,zero_carry_decisions=75,physical_intervals=2),indent=2))
    finally:env.close()


if __name__=="__main__":main()
