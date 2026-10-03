"""Bounded P10 collection: paired carry and three candidates, three seeds per scenario."""
import argparse
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--options", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--cpus", default="0,1,2,3")
    args = p.parse_args()
    prerequisite = REPO / "results/sdmpc_rl_p9_20261001/wave2"
    prior = json.loads((prerequisite / "analysis.json").read_text())
    if prior["status"] != "authenticated_training_diagnostic":
        raise RuntimeError("P9 must be authenticated before P10")
    prior_plan = json.loads((prerequisite / "plan.json").read_text())
    if any(sha(REPO / rel) != digest for rel, digest in prior_plan["sources"].items()):
        raise RuntimeError("Preserved P9 source changed")
    preflight = REPO / "results/sdmpc_rl_p10_20261001/preflight.json"
    preflight_result = json.loads(preflight.read_text())
    if preflight_result["status"] != "passed" or preflight_result["torch_threads"] != 1:
        raise RuntimeError("P10 preflight missing or failed")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / "queue.lock", "a+b")
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (out / "plan.json").exists():
        raise RuntimeError("Existing wave; inspect durable progress before recovery in a new wave")
    jobs = [j.split(":") for j in args.jobs.split(",")]
    free = [int(c) for c in args.cpus.split(",")]
    if not 1 <= len(free) <= 4 or len(set(free)) != len(free):
        raise ValueError("One to four distinct worker CPUs required")
    for scenario, seed in jobs:
        cache = args.cache / f"{scenario}_s{seed}"
        for file in ("carry.json", "k16.pt"):
            if not (cache / file).is_file():
                raise ValueError(f"Missing verified carry checkpoint: {cache / file}")
    inventory = {r["path"]: r["sha256"] for r in json.loads((REPO /
        "artifacts/sdmpc_machine_b_20261001/inventory.json").read_text())}
    for scenario, seed in jobs:
        slot = f"{scenario}_s{seed}"
        for file in ("carry.json", "k16.pt"):
            if sha(args.cache / slot / file) != inventory[f"probe_checkpoint_cache/{slot}/{file}"]:
                raise RuntimeError("Cached artifact differs from archive inventory: " + slot + "/" + file)
    options = json.loads(args.options.read_text(encoding="utf-8"))
    numeric_env = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")}
    numeric_env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    sources = list(HERE.glob("*.py")) + [REPO / "work/sdmpc_rl_probe_b_20260930/probe.py",
              REPO / "work/sdmpc_rl_perimeter_b_20261001/perimeter_actor.py"]
    sources += list((REPO / "work/sdmpc_rl_multi_20260929").glob("*.py"))
    sources += list((REPO / "work/sdmpc_rl_p9_20261001").glob("*.py"))
    sources += [REPO / sp["spec_path"] for sp in options]
    pins = {str(s.relative_to(REPO)): sha(s) for s in sources}
    plan = dict(pid=os.getpid(), started=time.time(), argv=sys.argv, python=sys.executable,
                jobs=jobs, cpus=free.copy(), sources=pins, options=options, numerical_environment=numeric_env,
                experience_scope="training_only_decisions_16_to_75_true_terminal",
                rerun_reason="New observation/action/reward capture including matched carry; P9 stored compact rows only",
                preflight_sha256=sha(preflight), prior_analysis_sha256=sha(prerequisite / "analysis.json"),
                checkpoint_hashes={f"{s}_s{n}/{f}": sha(args.cache / f"{s}_s{n}" / f)
                                   for s, n in jobs for f in ("carry.json", "k16.pt")})
    (out / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (out / "options.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
    running, finished = {}, []
    (out / "logs").mkdir(exist_ok=True)
    def record(event):
        event["time"] = time.time()
        with open(out / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)
    stop_roots = (REPO, out, REPO / "results/sdmpc_rl_balanced_goal_20260930",
                  REPO / "results/sdmpc_rl_machine_b_20260930")
    stopped = False
    while jobs or running:
        if any((r / "STOP").exists() for r in stop_roots):
            stopped = True
            jobs.clear()
        while jobs and free:
            for rel, digest in pins.items():
                if sha(REPO / rel) != digest:
                    raise RuntimeError("Source changed during wave: " + rel)
            scenario, seed = jobs.pop(0)
            cpu = free.pop(0)
            slot = f"{scenario}_s{seed}"
            cmd = [sys.executable, "-B", "-u", str(HERE / "p10_worker.py"), "--scenario", scenario,
                   "--seed", seed, "--cpu-mask", str(1 << cpu), "--branch-steps", "16",
                   "--options", str(out / "options.json"), "--output", str(out / slot),
                   "--checkpoint-dir", str(args.cache / slot)]
            with open(out / "logs" / f"{slot}.stdout.log", "w") as stdout, open(
                    out / "logs" / f"{slot}.stderr.log", "w") as stderr:
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=stdout, stderr=stderr,
                                        env=dict(os.environ, **numeric_env),
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            running[proc] = (slot, cpu)
            record(dict(event="start", pid=proc.pid, slot=slot, cpu=cpu, command=cmd))
        time.sleep(5)
        for proc in list(running):
            if proc.poll() is None:
                continue
            slot, cpu = running.pop(proc)
            free.append(cpu)
            finished.append(dict(slot=slot, exit_code=proc.returncode))
            record(dict(event="exit", pid=proc.pid, **finished[-1]))
    (out / "completion.json").write_text(json.dumps(dict(
        status="stopped" if stopped else ("failed" if any(x["exit_code"] for x in finished) else "completed"),
        finished=finished, ended=time.time()), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
