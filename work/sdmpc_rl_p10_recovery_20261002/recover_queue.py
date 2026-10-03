"""Recover authenticated P10 branches after shutdown, using the unchanged worker."""
import argparse
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
P10 = REPO / "work/sdmpc_rl_p10_20261001"
NUMERIC = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")}
NUMERIC.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
os.environ.update(NUMERIC)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def verify_files(root, pins):
    for rel, digest in pins.items():
        if sha(root / rel) != digest:
            raise RuntimeError("Pinned file changed: " + str(root / rel))


def stops(source, out, jobs):
    roots = [REPO, REPO / "results/sdmpc_rl_balanced_goal_20260930",
             REPO / "results/sdmpc_rl_machine_b_20260930", source, out]
    roots += [r / f"{s}_s{n}" for r in (source, out) for s, n in jobs]
    return [str(r / "STOP") for r in roots if (r / "STOP").exists()]


def lock_wave(wave):
    handle = (wave / "queue.lock").open("a+b")
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    return handle


def expected_options(plan):
    return {"k16_carry": {"name": "carry"}, **{
        "k16_p10_actor_" + s["tag"]: s for s in plan["options"]}}


def audit_branch(slot, file, carry, expected):
    import numpy as np
    sys.path.insert(0, str(P10))
    from p10_analysis import validate_experience
    from retention_actor import RetentionActor
    b = read(file)
    if b["option"] != expected or b["label"] != file.stem or b["branch_step"] != 16:
        raise ValueError("Unexpected branch identity: " + str(file))
    rows = b["rows"]
    if [r["control_step"] for r in rows] != list(range(15, 75)):
        raise ValueError("Incomplete branch: " + str(file))
    if b["ttt"] != rows[-1]["total_ttt"] or b["carry_ttt"] != carry["ttt"]:
        raise ValueError("TTT identity mismatch")
    if abs(b["prefix_ttt"] + sum(r["interval_ttt"] for r in rows) - b["ttt"]) > 1e-7:
        raise ValueError("TTT accounting mismatch")
    tag = expected.get("tag", "carry")
    if tag == "carry" and b["ttt"] != carry["ttt"]:
        raise ValueError("Carry reproduction mismatch")
    xp = slot / "experience" / f"{tag}.npz"
    meta_path = xp.with_suffix(".json")
    meta = read(meta_path)
    if sha(xp) != meta["file_sha256"]:
        raise ValueError("Experience hash mismatch")
    with np.load(xp, allow_pickle=False) as a:
        validate_experience(a, meta, b, carry)
        if tag != "carry":
            actor = RetentionActor(read(REPO / expected["spec_path"]), meta["observation_names"])
            for obs, act, mem, next_mem in zip(a["obs"], a["action"], a["memory"], a["next_memory"]):
                np.testing.assert_array_equal(actor.memory(), mem)
                np.testing.assert_array_equal(actor.act(obs), act)
                np.testing.assert_array_equal(actor.memory(), next_mem)
    return [file, xp, meta_path]


def prepare(source, out, cache):
    if out.exists():
        raise FileExistsError("Recovery output must be a new directory")
    source_lock = lock_wave(source)
    plan = read(source / "plan.json")
    if stops(source, out, plan["jobs"]):
        raise RuntimeError("STOP present")
    verify_files(REPO, plan["sources"])
    verify_files(cache, plan["checkpoint_hashes"])
    if read(source / "options.json") != plan["options"]:
        raise ValueError("Source wave options differ from plan")
    expected = expected_options(plan)
    copies, imported, pending, preserved_other = [], [], [], []
    for scenario, seed in plan["jobs"]:
        slot = source / f"{scenario}_s{seed}"
        files = sorted((slot / "branches").glob("*.json"))
        if any(f.stem not in expected for f in files):
            raise ValueError("Unexpected completed branch")
        carry_path = slot / "carry.json"
        if carry_path.exists():
            carry = read(carry_path)
            cached = read(cache / slot.name / "carry.json")
            actual = {k: v for k, v in carry.items() if k != "reused_from"}
            if actual != cached or carry["truncated"] or len(carry["rows"]) != 75:
                raise ValueError("Cached carry identity mismatch: " + slot.name)
            copies.append(carry_path)
        elif files:
            raise ValueError("Branch without carry provenance")
        for f in files:
            copies.extend(audit_branch(slot, f, carry, expected[f.stem]))
            imported.append(str(f.relative_to(source)))
        missing = sorted(set(expected) - {f.stem for f in files})
        if missing:
            if files and "k16_carry" not in {f.stem for f in files}:
                raise ValueError("Candidates without verified carry control")
            pending.append(dict(scenario=scenario, seed=seed, missing=missing))
        selected = set(copies)
        preserved_other.extend(str(p.relative_to(source)) for p in (slot / "experience").glob("*")
                               if p not in selected)
    out.mkdir(parents=True)
    copy_pins = {}
    for path in copies:
        rel = path.relative_to(source)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copy_pins[str(rel)] = sha(path)
    verify_files(out, copy_pins)
    new_plan = dict(plan, pid=None, started=None, argv=None,
        sources=dict(plan["sources"], **{str(Path(__file__).resolve().relative_to(REPO)): sha(__file__)}),
        numerical_environment=NUMERIC, cpus=[0, 1, 2, 3], cache=str(cache),
        recovery_source=str(source), recovery_source_plan_sha256=sha(source / "plan.json"),
        recovery_prepared=time.time(), recovery_jobs=pending, imported_branches=imported,
        imported_files=copy_pins, preserved_source_orphans=preserved_other,
        rerun_reason="User-reported shutdown. Reuse complete authenticated branches; restart missing branches from k16.")
    create_json(out / "options.json", plan["options"])
    create_json(out / "plan.json", new_plan)
    source_lock.close()
    print(json.dumps(dict(status="prepared", imported_branches=len(imported),
        remaining_branches=sum(len(j["missing"]) for j in pending), pending=pending,
        preserved_source_orphans=preserved_other), indent=2), flush=True)


def run(out):
    plan = read(out / "plan.json")
    source = Path(plan["recovery_source"])
    locks = [lock_wave(source), lock_wave(out)]
    if stops(source, out, plan["jobs"]):
        raise RuntimeError("STOP present")
    if sha(source / "plan.json") != plan["recovery_source_plan_sha256"]:
        raise RuntimeError("Source plan changed")
    verify_files(REPO, plan["sources"])
    verify_files(source, plan["imported_files"])
    verify_files(out, plan["imported_files"])
    verify_files(Path(plan["cache"]), plan["checkpoint_hashes"])
    if read(out / "options.json") != plan["options"]:
        raise RuntimeError("Recovery options changed")
    create_json(out / "run_started.json", dict(pid=os.getpid(), started=time.time(), argv=sys.argv))
    (out / "logs").mkdir()
    jobs = list(plan["recovery_jobs"])
    free = list(plan["cpus"])
    running, finished = {}, []
    stopped = failed = False
    def record(event):
        event["time"] = time.time()
        with (out / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
    while jobs or running:
        if stops(source, out, plan["jobs"]):
            stopped = True
            jobs.clear()
            # The unchanged worker observes the recovery-wave STOP each decision.
            if not (out / "STOP").exists():
                (out / "STOP").write_text("Propagated existing STOP from recovery source/scope\n")
        while jobs and free and not failed:
            verify_files(REPO, plan["sources"])
            job = jobs.pop(0)
            cpu = free.pop(0)
            slot = f"{job['scenario']}_s{job['seed']}"
            cmd = [sys.executable, "-B", "-u", str(P10 / "p10_worker.py"),
                "--scenario", job["scenario"], "--seed", str(job["seed"]),
                "--cpu-mask", str(1 << cpu), "--branch-steps", "16",
                "--options", str(out / "options.json"), "--output", str(out / slot),
                "--checkpoint-dir", str(Path(plan["cache"]) / slot)]
            with (out / "logs" / f"{slot}.stdout.log").open("x") as stdout, \
                 (out / "logs" / f"{slot}.stderr.log").open("x") as stderr:
                proc = subprocess.Popen(cmd, cwd=REPO, env=dict(os.environ, **NUMERIC),
                    stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            running[proc] = (slot, cpu)
            record(dict(event="start", pid=proc.pid, slot=slot, cpu=cpu, command=cmd,
                        missing=job["missing"]))
        time.sleep(5)
        for proc in list(running):
            if proc.poll() is None:
                continue
            slot, cpu = running.pop(proc)
            free.append(cpu)
            result = dict(slot=slot, exit_code=proc.returncode)
            finished.append(result)
            record(dict(event="exit", pid=proc.pid, **result))
            if proc.returncode:
                failed = True
                jobs.clear()
    verify_files(out, plan["imported_files"])
    verify_files(source, plan["imported_files"])
    status = "stopped" if stopped else "failed" if failed else "completed"
    if status == "completed":
        expected = set(expected_options(plan))
        for s, n in plan["jobs"]:
            if {f.stem for f in (out / f"{s}_s{n}" / "branches").glob("*.json")} != expected:
                raise RuntimeError("Recovery missing expected branches")
    create_json(out / "completion.json", dict(status=status, finished=finished,
        imported_branches=len(plan["imported_branches"]), ended=time.time()))
    for handle in locks:
        handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source is None or args.cache is None:
            parser.error("prepare requires --source and --cache")
        prepare(args.source.resolve(), args.output.resolve(), args.cache.resolve())
    else:
        run(args.output.resolve())
