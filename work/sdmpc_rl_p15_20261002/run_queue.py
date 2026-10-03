"""P15 paired bound test, reusing authenticated P13 controls without recomputation."""
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=REPO / "results/sdmpc_rl_p15_20261002"
OUT=ROOT / "wave1"
BASE=REPO / "results/sdmpc_rl_p13_20261002/pilot1"


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if OUT.exists():raise FileExistsError("Existing P15 wave; inspect instead of overwriting")
    original=read(BASE / "plan.json");report=read(BASE / "analysis.json")
    if report["status"]!="authenticated_neural_training_pilot" or read(BASE / "completion.json")["status"]!="completed":
        raise ValueError("P13 full authentication required")
    preflight=read(ROOT / "preflight.json")
    if preflight["status"]!="passed" or preflight["unchanged_decisions_checked"]!=675:
        raise ValueError("P15 preflight required")
    jobs=list(original["jobs"])
    roots=[REPO,ROOT,OUT,BASE,BASE.parent,REPO / "results/sdmpc_rl_balanced_goal_20260930",
           REPO / "results/sdmpc_rl_machine_b_20260930"]
    roots += [w / f"{s}_s{n}" for w in (OUT,BASE) for s,n in jobs]
    if any((r / "STOP").exists() for r in roots):raise RuntimeError("STOP before P15")
    specs=sorted((HERE / "specs").glob("*.json"))
    if {p.stem for p in specs}!={"half_np","restore_nuf","half_np_restore_nuf"}:raise ValueError("Three fixed factorial modes required")
    options=[dict(name="p15_actor",tag=p.stem,at=16,spec_path=str(p.relative_to(REPO)),spec_sha256=sha(p)) for p in specs]
    pins=dict(original["sources"])
    for path in list(HERE.glob("*.py"))+specs:
        pins[str(path.relative_to(REPO))]=sha(path)
    for rel,digest in preflight["sources"].items():
        if sha(REPO / rel)!=digest:raise ValueError("Preflight source changed")
    for rel,digest in preflight["specs"].items():
        if sha(REPO / rel)!=digest:raise ValueError("Preflight spec changed")
    def verify():
        for rel,digest in pins.items():
            if sha(REPO / rel)!=digest:raise ValueError("Pinned source/model changed: "+rel)
    verify()
    cache_hashes={}
    copies=[]
    for s,n in jobs:
        slot=f"{s}_s{n}"
        ctrl=next(r for r in report["rows"] if r["scenario"]==s and r["seed"]==n and r["policy"]=="carry")
        for file,expected in ((BASE / slot / "branches/k16_carry.json",ctrl["sha256"]),
                              (BASE / slot / "experience/carry.npz",ctrl["experience_sha256"])):
            if sha(file)!=expected:raise ValueError("P13 control changed")
        for file in ("carry.json","k16.pt"):
            cache_hashes[f"{slot}/{file}"]=sha(BASE / "cache" / slot / file)
        copies += [BASE / slot / "branches/k16_carry.json",BASE / slot / "experience/carry.npz",
                   BASE / slot / "experience/carry.json"]
    OUT.mkdir()
    locks=[]
    for wave in (BASE,OUT):
        lock=(wave / "queue.lock").open("a+b");lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);locks.append(lock)
    imported={}
    for path in copies:
        rel=path.relative_to(BASE);target=OUT / rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);imported[str(rel)]=sha(path)
        if sha(target)!=imported[str(rel)]:raise ValueError("Control copy failed")
    # probe recreates carry.json from the identical cache, adding reused_from.
    for s,n in jobs:
        slot=f"{s}_s{n}";shutil.copyfile(BASE / slot / "carry.json",OUT / slot / "carry.json")
    numeric=dict(original["numerical_environment"])
    plan=dict(pid=os.getpid(),started=time.time(),jobs=jobs,sources=pins,options=options,
        model_sha256=original["model_sha256"],numerical_environment=numeric,cpus=[0,1,2,3],
        source_wave=str(BASE),source_analysis_sha256=sha(BASE / "analysis.json"),
        imported_controls=imported,cache=str(BASE / "cache"),cache_hashes=cache_hashes,
        preflight_sha256=sha(ROOT / "preflight.json"),protocol_sha256=sha(ROOT / "protocol.md"),canonical_eligible=False,
        mode="paired_frozen_neural_strength_recovery_diagnostic",new_optimizer_updates=0,new_branches=15)
    (OUT / "plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    (OUT / "options.json").write_text(json.dumps(options,indent=2),encoding="utf-8")
    (OUT / "logs").mkdir()
    free=[0,1,2,3];running={};finished=[];stopped=failed=False
    def event(value):
        value["time"]=time.time()
        with (OUT / "events.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(value)+"\n")
        print(json.dumps(value),flush=True)
    while jobs or running:
        if any((r / "STOP").exists() for r in roots):
            stopped=True;jobs.clear()
            if not (OUT / "STOP").exists():(OUT / "STOP").write_text("Propagated STOP\n")
        while jobs and free and not failed:
            verify();s,n=jobs.pop(0);cpu=free.pop(0);slot=f"{s}_s{n}"
            for file in ("carry.json","k16.pt"):
                if sha(BASE / "cache" / slot / file)!=cache_hashes[f"{slot}/{file}"]:raise ValueError("Cache changed")
            cmd=[sys.executable,"-B","-u",str(HERE / "p15_worker.py"),"--scenario",s,"--seed",str(n),
                "--cpu-mask",str(1<<cpu),"--branch-steps","16","--options",str(OUT / "options.json"),
                "--output",str(OUT / slot),"--checkpoint-dir",str(BASE / "cache" / slot)]
            with (OUT / "logs" / f"{slot}.stdout.log").open("x") as stdout,(OUT / "logs" / f"{slot}.stderr.log").open("x") as stderr:
                p=subprocess.Popen(cmd,cwd=REPO,stdout=stdout,stderr=stderr,env=dict(os.environ,**numeric),creationflags=subprocess.CREATE_NO_WINDOW)
            running[p]=(slot,cpu);event(dict(event="start",pid=p.pid,slot=slot,cpu=cpu,command=cmd))
        time.sleep(5)
        for p in list(running):
            if p.poll() is None:continue
            slot,cpu=running.pop(p);free.append(cpu);item=dict(slot=slot,exit_code=p.returncode)
            finished.append(item);event(dict(event="exit",pid=p.pid,**item))
            if p.returncode:failed=True;jobs.clear()
    verify()
    for rel,digest in imported.items():
        if sha(OUT / rel)!=digest or sha(BASE / rel)!=digest:raise ValueError("Imported control changed")
    (OUT / "completion.json").write_text(json.dumps(dict(status="stopped" if stopped else "failed" if failed else "completed",finished=finished,ended=time.time()),indent=2),encoding="utf-8")
    for lock in locks:lock.close()


if __name__=="__main__":main()
