"""Run probe jobs from a queue with at most one job per CPU (single-CPU affinity each).

Usage: queue_runner.py --name p8 --options options_p8.json --branch-steps 16 \
         --jobs sweet_155_w:8501,sweet_170_w:8502,... --cpus 0,1,2,3,4,5 --checkpoint-root D:\...\ckpt
Each job is probe.py --no-control with its cached carry/checkpoints. Progress lines go to stdout.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = REPO / ".venv-torch/Scripts/python.exe"
MACHINE = REPO / "results/sdmpc_rl_machine_b_20260930"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--options", required=True)
    p.add_argument("--branch-steps", required=True)
    p.add_argument("--jobs", required=True, help="comma list scenario:seed")
    p.add_argument("--cpus", default="0,1,2,3,4,5")
    p.add_argument("--checkpoint-root", required=True)
    args = p.parse_args()
    jobs = [tuple(j.split(":")) for j in args.jobs.split(",") if j.strip()]
    free = [int(c) for c in args.cpus.split(",")]
    options = (HERE / args.options).resolve()
    out_root = MACHINE / f"probe_{args.name}"
    logs = MACHINE / "logs"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "options.json").write_bytes(options.read_bytes())
    running = {}
    while jobs or running:
        if any((d / "STOP").exists() for d in (REPO, MACHINE, out_root)):
            print("STOP present: no new dispatch", flush=True)
            jobs = []
        while jobs and free:
            scenario, seed = jobs.pop(0)
            cpu = free.pop(0)
            slot = out_root / f"{scenario}_s{seed}"
            log = logs / f"probe_{args.name}_{scenario}_s{seed}"
            cmd = [str(PY), "-B", "-u", str(HERE / "probe.py"), "--scenario", scenario, "--seed", seed,
                   "--cpu-mask", str(1 << cpu), "--branch-steps", args.branch_steps, "--options", str(options),
                   "--output", str(slot), "--no-control",
                   "--checkpoint-dir", str(Path(args.checkpoint_root) / f"{scenario}_s{seed}")]
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=open(f"{log}.stdout.log", "w"),
                                    stderr=open(f"{log}.stderr.log", "w"),
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            running[proc] = (scenario, seed, cpu)
            print(f"started {scenario} s{seed} on CPU {cpu} pid {proc.pid}", flush=True)
        time.sleep(20)
        for proc in list(running):
            if proc.poll() is not None:
                scenario, seed, cpu = running.pop(proc)
                free.append(cpu)
                print(f"finished {scenario} s{seed} exit {proc.returncode}", flush=True)
    print("ALL JOBS DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
