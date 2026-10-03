"""Preserve 11 authenticated P15 candidates and run only four missing branches."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_p15_20261002"
OLD, OUT = ROOT / "wave1", ROOT / "wave2_recovery"
SOURCE = REPO / "work/sdmpc_rl_p15_20261002"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("Existing recovery wave must not be overwritten")
    audit_path = ROOT / "recovery_audit_20261003.json"
    audit = read(audit_path)
    original = read(OLD / "plan.json")
    preflight = read(ROOT / "recovery_preflight_20261003.json")
    if audit["status"] != "authenticated_interrupted_p15" or preflight["status"] != "passed":
        raise ValueError("Recovery audit and physical preflight required")
    if sha(OLD / "plan.json") != audit["old_plan_sha256"] or sha(OLD / "events.jsonl") != audit["old_events_sha256"]:
        raise ValueError("Original wave changed after audit")
    if (OLD / "completion.json").exists():
        raise RuntimeError("Original wave completed; do not dispatch")
    base, cache = Path(original["source_wave"]), Path(original["cache"])
    all_jobs = original["jobs"]
    roots = [REPO,ROOT,OLD,OUT,base,base.parent,REPO / "results/sdmpc_rl_balanced_goal_20260930",
             REPO / "results/sdmpc_rl_machine_b_20260930"]
    roots += [wave / f"{s}_s{n}" for wave in (OLD,OUT,base) for s,n in all_jobs]
    if any((p / "STOP").exists() for p in roots):
        raise RuntimeError("STOP before recovery")
    pins = dict(original["sources"])
    pins.update(preflight["sources"])
    def verify():
        for rel,digest in pins.items():
            if sha(REPO / rel) != digest:
                raise ValueError("Pinned source/model changed: "+rel)
    verify()
    for rel,digest in original["cache_hashes"].items():
        if sha(cache / rel) != digest:
            raise ValueError("Carry/checkpoint changed")
    locks = []
    for wave in (OLD,base):
        lock = (wave / "queue.lock").open("a+b"); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1); locks.append(lock)
    OUT.mkdir()
    lock = (OUT / "queue.lock").open("a+b"); lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1); locks.append(lock)
    for rel,digest in audit["preserved_files"].items():
        source,target = OLD / rel, OUT / rel
        if sha(source) != digest:
            raise ValueError("Completed artifact changed")
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if sha(target) != digest:
            raise ValueError("Copy verification failed")
    pending = {(r["scenario"],r["seed"]) for r in audit["missing"]}
    jobs = [j for j in all_jobs if tuple(j) in pending]
    finished = [dict(slot=f"{s}_s{n}",exit_code=0,reused_from_original=True) for s,n in all_jobs if (s,n) not in pending]
    plan = dict(original,pid=os.getpid(),started=time.time(),sources=pins,
                recovery_from=str(OLD),recovered_files=audit["preserved_files"],
                recovery_audit_sha256=sha(audit_path),recovery_preflight_sha256=sha(ROOT / "recovery_preflight_20261003.json"),
                missing_branches=audit["missing"],dispatch_jobs=jobs,reused_candidate_branches=11,new_branches=4)
    (OUT / "plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    shutil.copyfile(OLD / "options.json",OUT / "options.json")
    if read(OUT / "options.json") != original["options"]:
        raise ValueError("Original policy options differ")
    (OUT / "logs").mkdir()
    def event(value):
        value["time"] = time.time()
        with (OUT / "events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    event(dict(event="recovery",complete_candidates=11,remaining_candidates=4,dispatch_jobs=jobs))
    running,free,stopped,failed = {},[0,1,2,3],False,False
    while jobs or running:
        if any((p / "STOP").exists() for p in roots):
            stopped=True; jobs.clear()
            if not (OUT / "STOP").exists():
                (OUT / "STOP").write_text("Propagated STOP\n",encoding="utf-8")
        while jobs and free and not failed:
            verify(); scenario,seed=jobs.pop(0); cpu=free.pop(0); slot=f"{scenario}_s{seed}"
            cmd=[sys.executable,"-B","-u",str(SOURCE / "p15_worker.py"),"--scenario",scenario,"--seed",str(seed),
                 "--cpu-mask",str(1<<cpu),"--branch-steps","16","--options",str(OUT / "options.json"),
                 "--output",str(OUT / slot),"--checkpoint-dir",str(cache / slot)]
            with (OUT / "logs" / f"{slot}.stdout.log").open("x") as stdout,(OUT / "logs" / f"{slot}.stderr.log").open("x") as stderr:
                p=subprocess.Popen(cmd,cwd=REPO,stdout=stdout,stderr=stderr,
                    env=dict(os.environ,**original["numerical_environment"]),creationflags=subprocess.CREATE_NO_WINDOW)
            running[p]=(slot,cpu); event(dict(event="start",pid=p.pid,slot=slot,cpu=cpu,command=cmd))
        time.sleep(5)
        for p in list(running):
            if p.poll() is None:
                continue
            slot,cpu=running.pop(p);free.append(cpu)
            item=dict(slot=slot,exit_code=p.returncode);finished.append(item)
            event(dict(event="exit",pid=p.pid,**item))
            if p.returncode:
                failed=True;jobs.clear()
    verify()
    for rel,digest in audit["preserved_files"].items():
        if sha(OUT / rel) != digest or sha(OLD / rel) != digest:
            raise ValueError("Recovered artifact changed")
    (OUT / "completion.json").write_text(json.dumps(dict(status="stopped" if stopped else "failed" if failed else "completed",
        finished=finished,ended=time.time(),reused_candidates=11,recomputed_missing_candidates=4),indent=2),encoding="utf-8")
    for lock in locks:
        lock.close()


if __name__ == "__main__":
    main()
