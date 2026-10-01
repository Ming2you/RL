"""Canonical evaluation of one frozen PerimeterActor spec on machine B: one scenario per process.

Physical contract (unchanged from the goal): frozen snapshot, canonical profile (training_seed=None),
physical feasibility guard, initial/recovery PFO, 5 warmup + 75 controlled intervals (14400 s), one
lower candidate, previous-executed-budget anchor, no exploration/learning/Q/performance gate.

Checked here before the run: the goal's frozen contract gate and digest, source pins, runtime
versions, the machine-B fingerprint (centers are machine specific), config/options digests,
coordinator contract, canonical profile hash with training_seed None, the 2367-entry observation
schema, and that the (spec, evaluator source) pair was registered in canonical_registry.json
BEFORE the run. Checked per interval: physical guard mode, one lower candidate, execution check
valid and budget feasible, NUF within capacity, timeline. After the run: 75 intervals / 14400 s,
cumulative and area TTT accounting, a replay of the recorded observations through a fresh actor
(identical actions), and unchanged sources/pins. Deeper physical row rules stay enforced by the
pinned environment code itself. Baseline: the machine-B carry center, exact pinned TTT.
Acceptance (goal rule): center TTT - candidate TTT > max(1e-6, 1e-8 * center TTT), per scenario;
the goal needs all five with the same spec (see readout.py).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MULTI = REPO / "work/sdmpc_rl_multi_20260929"
sys.path.insert(0, str(MULTI))
from budget_runtime import boot, DEFAULT_SNAPSHOT, read, plain, digest  # noqa: E402
from run_budget import pins, runtime_versions, observation_schema, exclusive_run  # noqa: E402
sys.path.insert(0, str(HERE))
from perimeter_actor import PerimeterActor  # noqa: E402

GOAL = REPO / "results/sdmpc_rl_balanced_goal_20260930"
MACHINE_ROOT = REPO / "results/sdmpc_rl_machine_b_20260930"
REGISTRY = MACHINE_ROOT / "canonical_registry.json"
GATE = GOAL / "value_audit_v1/projection/completion.json"
GATE_SHA = "b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3"
CONTRACT_SHA = "d6d14c4c18a7f8cfc88969bf0e9b8231ba5c452600593db49b125c1e5730eb70"
CENTER = MACHINE_ROOT / "center_repro_v1"
SCENARIOS = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")
MASKS = dict(zip(SCENARIOS, (1, 2, 4, 8, 16)))
BASE_TTT = dict(zip(SCENARIOS, (3101.4778926950567, 3934.6182377626324, 5546.224358313904,
                                4244.34789709013, 6593.685996544476)))
# identical to the machine-B evaluator preflight work/sdmpc_rl_return_eval_b_20260930/evidence/9f7515aa
MACHINE_B = dict(processor="AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD", machine="AMD64", cpu_count=6,
                 torch_cpu_capability="AVX2", platform="Windows-10-10.0.19045-SP0")
FORMAT = "sdmpc-perimeter-canonical-b1"
OUTPUTS = ("settings.json", "trace.json", "summary.json", "observations.npz", "completion.json")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, data: bytes):
    """Exclusive create: never overwrites, even under a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        stream.write(data)


def save_once(path, value):
    write_new(path, json.dumps(plain(value), ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8"))


def tag_nonfinite(value):
    """Keep raw diagnostic infinities (e.g. rejected-candidate stationarity) as explicit strict-JSON tags."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, dict):
        return {k: tag_nonfinite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tag_nonfinite(v) for v in value]
    return value


def same(a, b, label, tol=1e-8):
    if not (abs(float(a) - float(b)) <= tol):
        raise ValueError(f"Accounting mismatch: {label}: {a!r} vs {b!r}")


def stop_requested(out):
    return any((p / "STOP").exists() for p in (REPO, GOAL, MACHINE_ROOT, out.parent, out))


def machine_fingerprint():
    import torch
    return dict(processor=platform.processor(), machine=platform.machine(), cpu_count=os.cpu_count(),
                torch_cpu_capability=torch.backends.cpu.get_cpu_capability(), platform=platform.platform())


def own_sources():
    return {q.name: file_hash(q) for q in sorted(HERE.glob("*.py"))}


def contract():
    if file_hash(GATE) != GATE_SHA:
        raise ValueError("Physical contract gate changed")
    value = read(GATE)["settings"]["contract"]
    if digest(value) != CONTRACT_SHA:
        raise ValueError("Physical contract digest differs")
    return value


def registered(spec_sha, sources_sha):
    if not REGISTRY.exists():
        raise ValueError("No canonical registry: register the spec before any canonical run")
    entries = read(REGISTRY)["entries"]
    match = [e for e in entries if e["spec_sha256"] == spec_sha and e["evaluator_sources_sha256"] == sources_sha]
    if len(match) != 1:
        raise ValueError("Spec/evaluator pair is not registered exactly once in canonical_registry.json")
    return match[0]


def center(scenario, phys):
    folder = CENTER / scenario
    done, summary = read(folder / "completion.json"), read(folder / "episode_00_summary.json")
    settings = read(folder / "settings.json")
    if (done["status"] != "completed" or done["mode"] != "center" or settings["mode"] != "center" or
            settings["seeds"] != [None] or settings["model_sha256"] is not None or
            settings["source_pins"] != phys["source_pins"] or settings["runtime_versions"] != phys["runtime_versions"] or
            settings["environment_contract"] != phys["environment_contract"] or summary["control_steps"] != 75 or
            summary["simulation_seconds"] != 14400):
        raise ValueError("Machine-B center identity differs")
    if summary["ttt"] != BASE_TTT[scenario] or done["episodes"][0]["ttt"] != BASE_TTT[scenario]:
        raise ValueError("Machine-B center TTT differs from the pinned baseline")
    return dict(ttt=summary["ttt"], warmup_ttt=summary["warmup_ttt"], files={
        p.name: file_hash(p) for p in (folder / "completion.json", folder / "episode_00_summary.json",
                                       folder / "settings.json", folder / "episode_00_trace.json")})


def verify_env(env, rt, phys, scenario):
    ec = phys["environment_contract"]
    if (digest(rt["rc"].to_plain_dict(rt["cfg"])) != ec["config_sha256"] or
            digest(rt["rc"].to_plain_dict(rt["options"])) != ec["options_sha256"]):
        raise ValueError("Physical configuration/options differ")
    actual = {k: v for k, v in env.contract().items() if k not in ("cfg", "options", "observation", "source_snapshot")}
    profile = next(r["profile_sha256"] for r in ec["scenarios"] if r["scenario"] == scenario)
    if actual != ec["coordinator"] or env.training_seed is not None or env.profile_hash != profile:
        raise ValueError("Canonical environment/profile/seed differs")


def check_row(row, i):
    check = row["execution_check"]
    if (row["selection_source"] not in ("lower_solution", "reference_fallback") or len(row["candidates"]) != 1 or
            row["guard_mode"] != "physical" or row["h3_guard_enabled"] is not False or
            check["physical_control_valid"] is not True or check["budget_feasible"] is not True or
            row["control_step"] != i or row["step"] != i + 5 or row["time_sec"] != (i + 6) * 180 or
            row["terminated"] is not (i == 74) or row["truncated"] is not False):
        raise ValueError(f"Physical/timeline contract differs at control step {i}")
    if not 0 <= row["B_executed"][1] <= 6000.:
        raise ValueError("Executed NUF outside capacity")


def run(args, out, smoke):
    phys = contract()
    base = None if smoke else center(args.scenario, phys)
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    sources = own_sources()
    fingerprint = machine_fingerprint()
    registration = None
    if not smoke:
        if fingerprint != MACHINE_B:
            raise ValueError(f"Not machine B: {fingerprint}")
        from perimeter_actor import spec_sha256
        registration = registered(spec_sha256(spec), digest(sources))
    started, cpu0 = time.perf_counter(), time.process_time()
    source_pins = pins(DEFAULT_SNAPSHOT)
    if source_pins != phys["source_pins"]:
        raise ValueError("Frozen source pins differ from the physical contract")
    if runtime_versions() != phys["runtime_versions"]:
        raise ValueError("Runtime versions differ from the physical contract")
    mask = (args.smoke_cpu_mask if smoke and args.smoke_cpu_mask else
            args.cpu_mask if args.cpu_mask else MASKS[args.scenario])
    rt = boot(DEFAULT_SNAPSHOT, mask)
    import numpy as np
    import torch
    from budget_env import BudgetEnv
    torch.set_num_threads(1)
    env = BudgetEnv(rt, scenario=args.scenario, training_seed=args.smoke_training_seed, guard_mode="physical")
    try:
        if not smoke:
            verify_env(env, rt, phys, args.scenario)
        obs = env.reset()
        if observation_schema(env) != phys["observation_schema"]:
            raise ValueError("Observation schema differs from the physical contract")
        actor = PerimeterActor(spec, env.observer.names)
        if not smoke:
            same(env.warmup_ttt, base["warmup_ttt"], "matched warmup")
        settings = dict(format=FORMAT + ("-SMOKE-NOT-CANONICAL" if smoke else ""), scenario=args.scenario,
                        training_seed=args.smoke_training_seed, guard_mode="physical",
                        evaluation=True, exploration=False, learning=False, policy_q=None, gate=False,
                        actor_format=spec["format"], actor_spec=spec, actor_spec_sha256=actor.sha256,
                        actor_spec_path=str(spec_path), cpu_mask=mask, machine_fingerprint=fingerprint,
                        registration=registration, contract_sha256=CONTRACT_SHA, gate_sha256=GATE_SHA,
                        source_pins=source_pins, runtime_versions=runtime_versions(), evaluator_sources=sources,
                        evaluator_sources_sha256=digest(sources), profile_sha256=env.profile_hash,
                        warmup_ttt=env.warmup_ttt, baseline=base, pid=os.getpid(), argv=sys.argv)
        save_once(out / "settings.json", settings)
        trace, actions, observations = [], [], []
        while env.k < 80 and env.k - 5 < args.smoke_max_steps:
            if stop_requested(out):
                raise SystemExit("STOP requested; partial run abandoned (rerun in a new root)")
            observations.append(obs.copy())
            tick, cpu = time.perf_counter(), time.process_time()
            action = actor.act(obs)
            actor_wall, actor_cpu = time.perf_counter() - tick, time.process_time() - cpu
            if action.dtype != np.float32 or action.shape != (2,) or not np.isfinite(action).all():
                raise ValueError("Invalid actor action")
            obs, reward, terminal, row = env.step(action, "rl", actor_wall, actor_cpu_seconds=actor_cpu)
            row.update(reward=reward, scenario=args.scenario, profile_sha256=env.profile_hash,
                       actor_spec_sha256=actor.sha256, policy_q=None, learning=[], learning_wall_seconds=0.)
            check_row(row, len(trace))
            trace.append(plain(row))
            actions.append([float(x) for x in action])
            print(args.scenario, f"{len(trace)}/75", f"TTT={env.sim.total_ttt:.6f}", flush=True)
        if not smoke and (len(trace) != 75 or env.sim.state.time_sec != 14400.):
            raise ValueError("Not a full canonical run")
        total = env.warmup_ttt
        for row in trace:
            total += row["interval_ttt"]
            same(row["total_ttt"], total, "cumulative TTT")
            same(row["freeway_ttt"] + row["urban_ttt"], total, "area TTT")
        # the actor is a pure function: replaying the recorded observations must give identical actions
        replay = PerimeterActor(spec, env.observer.names)
        for o, a in zip(observations, actions):
            if replay.act(o).tolist() != a:
                raise ValueError("Actor replay differs: hidden state or nondeterminism")
        if own_sources() != sources or pins(DEFAULT_SNAPSHOT) != source_pins:
            raise ValueError("Evaluator sources or frozen pins changed during the run")
        ttt = trace[-1]["total_ttt"]
        improvement = None if smoke else base["ttt"] - ttt
        threshold = None if smoke else max(1e-6, 1e-8 * base["ttt"])
        summary = dict(format=settings["format"], scenario=args.scenario, actor_spec_sha256=actor.sha256,
                       evaluator_sources_sha256=digest(sources), machine_fingerprint=fingerprint, ttt=ttt,
                       center_ttt=None if smoke else base["ttt"], improvement=improvement,
                       improvement_pct=None if smoke else 100. * improvement / base["ttt"],
                       threshold=threshold, passes=(not smoke) and improvement > threshold,
                       freeway_ttt=trace[-1]["freeway_ttt"], urban_ttt=trace[-1]["urban_ttt"],
                       warmup_ttt=env.warmup_ttt, terminal_inventory=trace[-1]["inventory"],
                       peak_inventory=max(r["inventory"] for r in trace),
                       fallback_steps=[i + 1 for i, r in enumerate(trace) if r["selection_source"] == "reference_fallback"],
                       pfo_calls=sum(r["pfo_calls"] for r in trace),
                       pfo_recovery_count=sum(r["reference_recovery"] for r in trace),
                       decision_wall_seconds=sum(r["decision_wall_seconds"] for r in trace),
                       actor_wall_seconds=sum(r["actor_wall_seconds"] for r in trace),
                       elapsed_wall_seconds=time.perf_counter() - started, elapsed_cpu_seconds=time.process_time() - cpu0,
                       actions=actions, control_steps=len(trace), simulation_seconds=env.sim.state.time_sec)
        import io
        buffer = io.BytesIO()
        np.savez_compressed(buffer, observations=np.stack(observations), actions=np.asarray(actions, dtype=np.float32))
        write_new(out / "observations.npz", buffer.getvalue())
        save_once(out / "trace.json", tag_nonfinite(trace))
        save_once(out / "summary.json", summary)
        save_once(out / "completion.json", dict(format=settings["format"], status="smoke" if smoke else "completed",
                  scenario=args.scenario, actor_spec_sha256=actor.sha256, evaluator_sources_sha256=digest(sources),
                  outputs_sha256={n: file_hash(out / n) for n in OUTPUTS if n != "completion.json"}))
        if smoke:
            print(f"SMOKE {args.scenario} steps={len(trace)} TTT={ttt!r} (training profile, no baseline)", flush=True)
        else:
            print(f"COMPLETED {args.scenario} TTT={ttt!r} center={base['ttt']!r} improvement={improvement:+.6f} "
                  f"({summary['improvement_pct']:+.4f}%) passes={summary['passes']}", flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=SCENARIOS, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="run root; the scenario folder is created inside")
    p.add_argument("--smoke-training-seed", type=int,
                   help="CODE-PATH SMOKE TEST ONLY: use this training profile (never canonical), no baseline claim")
    p.add_argument("--smoke-max-steps", type=int, default=75)
    p.add_argument("--smoke-cpu-mask", type=int, help="smoke only: CPU mask override")
    p.add_argument("--cpu-mask", type=int, help="single-CPU affinity mask (default: per-scenario 1,2,4,8,16)")
    args = p.parse_args()
    smoke = args.smoke_training_seed is not None
    if not smoke and (args.smoke_max_steps != 75 or args.smoke_cpu_mask is not None):
        raise ValueError("Smoke options require --smoke-training-seed")
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass
    out = args.output.resolve() / args.scenario
    if any((out / n).exists() for n in OUTPUTS):
        raise FileExistsError("Scenario slot already used; evaluate in a new run root")
    if stop_requested(out):
        raise SystemExit("STOP requested")
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out):
        try:
            run(args, out, smoke)
        except BaseException:
            write_new(out / f"failure-{time.strftime('%Y%m%d_%H%M%S')}-{os.getpid()}.txt",
                      traceback.format_exc().encode("utf-8"))
            raise


if __name__ == "__main__":
    main()
