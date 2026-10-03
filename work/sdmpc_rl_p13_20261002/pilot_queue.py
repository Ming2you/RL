"""Five prospective training profiles, one frozen shared neural policy."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_p13_20261002"
OUT = ROOT / "pilot1"
JOBS = [["sweet_170_w",9202],["sweet_170_incident_w",9203],["sweet_190_w",9205],
        ["sweet_170_skew15_w",9204],["sweet_155_w",9201]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if OUT.exists():
        raise FileExistsError("Existing P13 wave must be inspected, never overwritten")
    stop_roots = [REPO,ROOT,ROOT / "fit_v1",OUT,
        REPO / "results/sdmpc_rl_balanced_goal_20260930",REPO / "results/sdmpc_rl_machine_b_20260930"]
    stop_roots += [OUT / f"{s}_s{n}" for s,n in JOBS]
    if any((p / "STOP").exists() for p in stop_roots):
        raise RuntimeError("STOP before pilot")
    fit = read(ROOT / "fit_v1/completion.json")
    preflight = read(ROOT / "preflight.json")
    policy = ROOT / "fit_v1/policy.json"
    if fit["status"] != "fit_passed" or preflight["status"] != "passed" or preflight["spec_sha256"] != sha(policy):
        raise ValueError("Neural fitting/preflight gates required")
    if fit["model_sha256"] != sha(ROOT / "fit_v1/model.pt"):
        raise ValueError("Neural weights changed")
    manifest=read(HERE / "fit_manifest.json")
    pins=dict(manifest["source_pins"])
    trained=read(ROOT / "fit_v1/plan.json")
    for rel,digest in trained["source_pins"].items():
        if sha(REPO / rel)!=digest:raise ValueError("Source changed after fitting: "+rel)
    for path in list(HERE.glob("*.py"))+[HERE / "fit_manifest.json",policy,ROOT / "fit_v1/model.pt",ROOT / "fit_v1/neural_policy.json"]:
        pins[str(path.relative_to(REPO))] = sha(path)
    def verify():
        for rel,digest in pins.items():
            if sha(REPO / rel)!=digest:
                raise RuntimeError("Pinned source/model changed: "+rel)
    verify()
    OUT.mkdir()
    lock = (OUT / "queue.lock").open("a+b"); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    options=[dict(name="p13_actor",tag="rwbc_v2",at=16,
        spec_path=str(policy.relative_to(REPO)),spec_sha256=sha(policy))]
    numeric={k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
        "NUMEXPR_MAX_THREADS","BLIS_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS")}
    numeric.update(OMP_DYNAMIC="FALSE",MKL_DYNAMIC="FALSE")
    plan=dict(pid=os.getpid(),started=time.time(),jobs=JOBS,sources=pins,options=options,
        model_sha256=fit["model_sha256"],numerical_environment=numeric,cpus=[0,1,2,3],
        fit_completion_sha256=sha(ROOT / "fit_v1/completion.json"),preflight_sha256=sha(ROOT / "preflight.json"),
        source_kind="training_only",mode="prospective_neural_pilot",canonical_eligible=False,
        reason="One fresh seed per scenario; complete carry/control/policy, then independent seed expansion only if promising")
    (OUT / "plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    (OUT / "options.json").write_text(json.dumps(options,indent=2),encoding="utf-8")
    (OUT / "logs").mkdir()
    jobs=list(JOBS); free=[0,1,2,3]; running={}; finished=[]; stopped=failed=False
    def event(value):
        value["time"]=time.time()
        with (OUT / "events.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    while jobs or running:
        if any((p / "STOP").exists() for p in stop_roots):
            stopped=True; jobs.clear()
            if not (OUT / "STOP").exists(): (OUT / "STOP").write_text("Propagated STOP\n")
        while jobs and free and not failed:
            verify(); scenario,seed=jobs.pop(0); cpu=free.pop(0); slot=f"{scenario}_s{seed}"
            cmd=[sys.executable,"-B","-u",str(HERE / "p13_worker.py"),"--scenario",scenario,"--seed",str(seed),
                "--cpu-mask",str(1<<cpu),"--branch-steps","16","--options",str(OUT / "options.json"),
                "--output",str(OUT / slot),"--checkpoint-dir",str(OUT / "cache" / slot)]
            with (OUT / "logs" / f"{slot}.stdout.log").open("x") as stdout, (OUT / "logs" / f"{slot}.stderr.log").open("x") as stderr:
                p=subprocess.Popen(cmd,cwd=REPO,stdout=stdout,stderr=stderr,env=dict(os.environ,**numeric),creationflags=subprocess.CREATE_NO_WINDOW)
            running[p]=(slot,cpu); event(dict(event="start",pid=p.pid,slot=slot,cpu=cpu,command=cmd))
        time.sleep(5)
        for p in list(running):
            if p.poll() is None: continue
            slot,cpu=running.pop(p); free.append(cpu); item=dict(slot=slot,exit_code=p.returncode)
            finished.append(item); event(dict(event="exit",pid=p.pid,**item))
            if p.returncode: failed=True; jobs.clear()
    verify()
    (OUT / "completion.json").write_text(json.dumps(dict(status="stopped" if stopped else "failed" if failed else "completed",
        finished=finished,ended=time.time()),indent=2),encoding="utf-8")
    lock.close()


if __name__ == "__main__":
    main()
