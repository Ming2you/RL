"""Counterfactual budget probes on TRAINING profiles (machine B); never trains, never touches canonical.

One process = one (scenario, training seed). It runs the zero-action carry trajectory once with the
unchanged frozen BudgetEnv, keeps in-memory env checkpoints at the requested control steps, then for
every (branch step, option) restores the checkpoint, applies the option, continues with carry (zero
action) to the true terminal and records the full-episode TTT. A zero-action control branch from the
first checkpoint must reproduce the carry TTT bit-for-bit (restore determinism check).

Options are budget-level: at each step of the option window the requested budget target is converted
into the env's residual action a = clip((target - anchor) / [50, 1000], -1, 1). After the window the
option either keeps carrying (hold, the env's zero action) or steers back to the pre-option anchor.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MULTI = REPO / "work/sdmpc_rl_multi_20260929"
sys.path.insert(0, str(MULTI))
from budget_runtime import boot, DEFAULT_SNAPSHOT, plain  # noqa: E402

SCALE = (50., 1000.)
SCENARIOS = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=1, allow_nan=True), encoding="utf-8")
    for attempt in range(20):  # Windows: a concurrent reader holding the target makes replace fail briefly
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    tmp.replace(path)


def compact(row):
    q = row["queue_state"]
    return dict(
        control_step=row["control_step"], B_requested=list(row["B_requested"][0]), B_executed=list(row["B_executed"]),
        G_achieved=list(row["G_achieved"]), action=[float(x) for x in row["action_requested"]],
        anchor=list(row["action_anchor"]), reference_budget=list(row["reference_budget"]),
        source=row["selection_source"], fallback_reasons=list(row["fallback_reasons"]),
        converged=bool(row["converged"]), interval_ttt=row["interval_ttt"], total_ttt=row["total_ttt"],
        freeway_ttt=row["freeway_ttt"], urban_ttt=row["urban_ttt"], inventory=row["inventory"],
        ramp_queue=float(sum(q["ramp_queue"].values())), boundary_queue=float(sum(q["boundary_queue"].values())),
        origin_queue=float(sum(q["mainline_origin_queue"].values())),
        urban_queue=float(sum(q["urban_movement_queue"].values())),
        decision_wall_seconds=row["decision_wall_seconds"])


def action_for_target(target, anchor):
    import numpy as np
    a = (np.asarray(target, float) - np.asarray(anchor, float)) / np.asarray(SCALE)
    return np.clip(a, -1., 1.).astype(np.float32)


class Option:
    """Budget-target option over a window of control steps, then hold or return."""

    def __init__(self, spec):
        self.spec = dict(spec)
        self.name = spec["name"]
        self.window = int(spec.get("window", 0))
        self.after = spec.get("after", "hold")  # "hold" (carry) or "return" (steer back to pre-option anchor)

    def begin(self, env):
        import numpy as np
        self.t = 0
        self.pre_anchor = np.asarray(env.controller.action_anchor, float).copy()

    def act(self, env, obs=None):
        import numpy as np
        anchor = np.asarray(env.controller.action_anchor, float)
        achieved = np.asarray(env.last_executed, float) - np.asarray(env.last_slack, float)
        t, self.t = self.t, self.t + 1
        if self.name == "carry":
            return np.zeros(2, np.float32)
        if self.name == "actor":
            # the exact frozen actor used for canonical evaluation, driven by the observation only
            if not hasattr(self, "actor"):
                sys.path.insert(0, str(REPO / "work/sdmpc_rl_perimeter_b_20261001"))
                from perimeter_actor import PerimeterActor
                self.actor = PerimeterActor.load(REPO / self.spec["spec_path"], env.observer.names)
                if self.actor.sha256 != self.spec["spec_sha256"]:
                    raise ValueError("Actor spec hash differs")
            return self.actor.act(obs)
        if self.name == "np_feedback":
            # Stateless perimeter feedback on the urban queue (sum of urban movement queues, veh):
            # tighten NP below the last achieved net inflow while the urban queue is high, keep NP
            # non-binding (margin above achieved) while it is low, hold in between (hysteresis).
            uq = float(sum(env.sim.state.urban_movement_queue.values()))
            target = anchor.copy()
            if uq >= float(self.spec["uq_on"]):
                delta = float(self.spec["delta"]) + float(self.spec.get("gain", 0.)) * (uq - float(self.spec["uq_on"]))
                target[0] = achieved[0] - min(delta, float(self.spec.get("delta_max", delta)))
            elif uq <= float(self.spec.get("uq_off", self.spec["uq_on"])):
                target[0] = max(anchor[0], achieved[0] + float(self.spec.get("margin", 30.)))
            return action_for_target(target, anchor)
        if t < self.window:
            target = anchor.copy()
            if "np_delta" in self.spec:     # bind NP below last achieved net inflow by delta (veh over H3)
                target[0] = achieved[0] - float(self.spec["np_delta"])
            if "nuf_delta" in self.spec:    # bind NUF below last achieved metering sum by delta (veh/h)
                target[1] = achieved[1] - float(self.spec["nuf_delta"])
            if "np_shift" in self.spec:     # shift NP relative to the current anchor
                target[0] = anchor[0] + float(self.spec["np_shift"])
            if "np_margin" in self.spec:    # keep NP at least margin above last achieved (relax only if needed)
                target[0] = max(anchor[0], achieved[0] + float(self.spec["np_margin"]))
            if "nuf_shift" in self.spec:
                target[1] = anchor[1] + float(self.spec["nuf_shift"])
            return action_for_target(target, anchor)
        if self.after == "return":
            return action_for_target(self.pre_anchor, anchor)
        if self.after == "margin":  # stateless return: NP at least after_margin above last achieved
            target = anchor.copy()
            target[0] = max(anchor[0], achieved[0] + float(self.spec["after_margin"]))
            return action_for_target(target, anchor)
        if self.after == "level":   # stateless return: relax NP up to a fixed level (never tighten)
            target = anchor.copy()
            target[0] = max(anchor[0], float(self.spec["after_level"]))
            return action_for_target(target, anchor)
        return np.zeros(2, np.float32)


def run_to_end(env, obs, option, rows, deadline, max_step=75):
    terminal = False
    while not terminal and env.k - 5 < max_step:
        if time.time() > deadline:
            raise TimeoutError("probe wall-clock limit")
        action = option.act(env, obs)
        obs, _, terminal, row = env.step(action, "rl", 0., actor_cpu_seconds=0.)
        rows.append(compact(row))
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--seed", type=int, required=True, help="training demand seed (canonical is refused)")
    p.add_argument("--cpu-mask", type=int, required=True)
    p.add_argument("--branch-steps", type=str, required=True, help="comma list of control steps (1..74)")
    p.add_argument("--options", type=Path, required=True, help="JSON list of option specs")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-hours", type=float, default=30.)
    p.add_argument("--max-steps", type=int, default=75, help="smoke tests only: truncate every run here")
    p.add_argument("--no-control", action="store_true",
                   help="skip the zero-action restore control (determinism already established)")
    p.add_argument("--checkpoint-dir", type=Path,
                   help="store/reuse the carry trajectory and branch checkpoints (per scenario+seed)")
    args = p.parse_args()
    try:  # keep the desktop responsive when all CPUs run probes; numerics are unaffected
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 3600. * args.max_hours
    branch_steps = sorted({int(x) for x in args.branch_steps.split(",") if x.strip()})
    if not branch_steps or min(branch_steps) < 1 or max(branch_steps) > 74:
        raise ValueError("branch steps must be in 1..74")
    specs = json.loads(args.options.read_text(encoding="utf-8"))
    save(out / "plan.json", dict(scenario=args.scenario, seed=args.seed, branch_steps=branch_steps,
                                 options=specs, pid=os.getpid(), started=time.time(), argv=sys.argv))
    rt = boot(DEFAULT_SNAPSHOT, args.cpu_mask)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    torch.set_num_threads(1)
    env = BudgetEnv(rt, scenario=args.scenario, training_seed=args.seed, guard_mode="physical")
    try:
        # 1. carry trajectory with checkpoints before the step at each branch control step; with
        #    --checkpoint-dir the carry and its checkpoints are stored once and reused by later probes
        carry_path = out / "carry.json"
        store = args.checkpoint_dir.resolve() if args.checkpoint_dir else None
        cached = store is not None and (store / "carry.json").exists() and all(
            (store / f"k{k:02d}.pt").exists() for k in branch_steps)
        checkpoints = {}
        if cached:
            carry = json.loads((store / "carry.json").read_text(encoding="utf-8"))
            if carry["scenario"] != args.scenario or carry["seed"] != args.seed or carry["profile_sha256"] != env.profile_hash:
                raise ValueError("Cached carry does not match scenario/seed/profile")
            for k in branch_steps:
                checkpoints[k] = torch.load(store / f"k{k:02d}.pt", map_location="cpu", weights_only=False)
            rows, carry_ttt = carry["rows"], carry["ttt"]
            save(carry_path, dict(carry, reused_from=str(store)))
            print(f"carry {args.scenario} seed {args.seed} TTT={carry_ttt:.6f} (cached)", flush=True)
        else:
            obs = env.reset()
            rows = []
            t0 = time.time()
            while env.k < 80 and env.k - 5 < args.max_steps:
                cstep = env.k - 4  # control step about to be executed (1-based)
                if cstep in branch_steps:
                    checkpoints[cstep] = env.checkpoint()
                    if store is not None:
                        store.mkdir(parents=True, exist_ok=True)
                        torch.save(checkpoints[cstep], store / f"k{cstep:02d}.pt.tmp")
                        (store / f"k{cstep:02d}.pt.tmp").replace(store / f"k{cstep:02d}.pt")
                obs, _, terminal, row = env.step(np.zeros(2, np.float32), "rl", 0., actor_cpu_seconds=0.)
                rows.append(compact(row))
                print(f"carry {len(rows)}/75 TTT={rows[-1]['total_ttt']:.6f}", flush=True)
            carry_ttt = rows[-1]["total_ttt"]
            carry = dict(scenario=args.scenario, seed=args.seed, profile_sha256=env.profile_hash,
                         warmup_ttt=env.warmup_ttt, ttt=carry_ttt, rows=rows, wall_seconds=time.time() - t0,
                         truncated=len(rows) < 75)
            save(carry_path, carry)
            if store is not None and not carry["truncated"]:
                save(store / "carry.json", carry)
            print(f"carry {args.scenario} seed {args.seed} TTT={carry_ttt:.6f}", flush=True)
        # 2. branches
        # determinism control from the cheapest (latest) checkpoint; option specs may pin their own
        # branch step with "at", otherwise they run at every branch step
        jobs = ([] if args.no_control else [(branch_steps[-1], dict(name="carry"))]) + [
            (k, sp) for sp in specs for k in branch_steps if sp.get("at", k) == k]
        for k, spec in jobs:
            if "tag" in spec:
                label = f"k{k:02d}_{spec['name']}_{spec['tag']}"
            else:
                label = f"k{k:02d}_{spec['name']}" + ("" if spec["name"] == "carry" else
                        "_" + "_".join(f"{key}{spec[key]}" for key in sorted(spec) if key not in ("name", "at")))
            path = out / "branches" / f"{label}.json"
            if path.exists():
                continue
            t0 = time.time()
            obs = env.restore(checkpoints[k])
            option = Option(spec)
            option.begin(env)
            brows = []
            run_to_end(env, obs, option, brows, deadline, args.max_steps)
            ttt = brows[-1]["total_ttt"]
            prefix_ttt = rows[k - 2]["total_ttt"] if k >= 2 else env.warmup_ttt
            result = dict(branch_step=k, option=spec, label=label, ttt=ttt, carry_ttt=carry_ttt,
                          delta_ttt=ttt - carry_ttt, delta_pct=100. * (ttt - carry_ttt) / carry_ttt,
                          prefix_ttt=prefix_ttt, rows=brows, wall_seconds=time.time() - t0,
                          fallbacks=sum(r["source"] == "reference_fallback" for r in brows))
            save(path, result)
            print(f"branch {label}: TTT={ttt:.6f} delta={ttt - carry_ttt:+.6f} ({result['delta_pct']:+.3f}%)",
                  flush=True)
            save(out / "status.json", dict(phase="branches", done=label))
        save(out / "status.json", dict(phase="completed"))
    except BaseException:
        save(out / "failure.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
