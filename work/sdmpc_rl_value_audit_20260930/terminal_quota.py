"""Bounded paired terminal-quota ablation on authenticated training replay only."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import terminal_fit as terminal

HERE = Path(__file__).resolve().parent
REPO, BASE, BASE_HASH = terminal.REPO, terminal.BASE, terminal.BASE_HASH
GOAL_ROOT = terminal.GOAL_ROOT
PROJECTION = GOAL_ROOT / "value_audit_v1/projection/completion.json"
_spec = spec_from_file_location(
    "terminal_quota_critic", HERE.parent / "sdmpc_rl_recovery_20260929/critic_diagnostic.py")
critic = module_from_spec(_spec)
_spec.loader.exec_module(critic)
torch, np, SCENARIOS = terminal.torch, terminal.np, terminal.SCENARIOS
load_base, exclusive_run = terminal.load_base, terminal.exclusive_run
fingerprint, file_hash, json_save = terminal.fingerprint, terminal.file_hash, terminal.json_save
SEEDS = (6529, 6530)
UPDATES = 3750
LOG_UPDATES = (0, 375, 750, 1500, 3750)
ARMS = ("uniform", "terminal_quota")
TAU = 0.005
INITIAL_MAE = 5.701837813854217
EVALUATION_SEED = 86529
CAVEAT = (
    "Observed behavior returns are not current fixed-policy Q ground truth. "
    "Exact terminal rewards are ground truth only at the ten stored state-actions. "
    "The old saturated actor is frozen; this diagnostic cannot justify deployment, "
    "policy admission, traffic improvement, holdout or generalization claims. "
    "Metrics never stop, select or export a policy. If the predeclared criterion "
    "fails, do not keep increasing this same update budget.")


def input_paths():
    paths = [BASE / "completion.json", BASE / "train_round0/model_final.pt",
             BASE / "train_round1/model_final.pt"]
    for episode in (0, 1):
        for scenario in SCENARIOS:
            folder = BASE / f"collect_round{episode}" / scenario
            paths.extend(folder / name for name in (
                "completion.json", "settings.json", "experience.pt", "episode_00_trace.json",
                "episode_00_summary.json", "observation_schema.json", "runtime_versions.json"))
    return paths


def capture_identity(stop):
    """Read the prior audit as evidence; never import or run its physical runtime."""
    stop()
    identity = terminal.capture_identity(stop)
    sources = (Path(__file__), HERE / "terminal_quota_brief.md", Path(terminal.__file__),
               Path(terminal.common.__file__), Path(critic.__file__), Path(critic._td3.__file__),
               *(REPO / "work/sdmpc_rl_multi_20260929" / name for name in
                 ("td3.py", "run_budget.py", "budget_runtime.py", "freeze_runtime.py")))
    for path in sources:
        stop()
        identity["source_sha256"][str(path)] = file_hash(path)
    stop()
    payload = PROJECTION.read_bytes()
    completion_hash = hashlib.sha256(payload).hexdigest()
    completion = json.loads(payload)
    settings = completion.get("settings", {})
    if completion.get("status") != "completed" or settings.get("base_sha256") != BASE_HASH:
        raise ValueError("Successful projection audit with exact base hash required")
    expected = {str(path.relative_to(REPO)): path for path in input_paths()}
    manifest = settings.get("input_sha256", {})
    if len(manifest) != 73 or set(manifest) != set(expected):
        raise ValueError("Projection audit must contain the exact 73-file input manifest")
    for name, path in expected.items():
        stop()
        if file_hash(path) != manifest[name]:
            raise ValueError("Projection input changed: " + name)
    model_key = str((BASE / "train_round1/model_final.pt").relative_to(REPO))
    if manifest[model_key] != BASE_HASH:
        raise ValueError("Projection manifest base hash differs")
    projection_sources = settings.get("source_sha256", {})
    expected_sources = (HERE / "projection_audit.py", Path(terminal.common.__file__))
    if set(projection_sources) != {str(path.relative_to(REPO)) for path in expected_sources}:
        raise ValueError("Projection source manifest differs")
    for path in expected_sources:
        stop()
        digest = file_hash(path)
        if digest != projection_sources[str(path.relative_to(REPO))]:
            raise ValueError("Projection source changed: " + str(path))
        identity["source_sha256"][str(path)] = digest
    contract = settings.get("contract", {})
    if (contract.get("source_pins") != identity["base_source_pins"] or
            contract.get("runtime_versions") != identity["runtime"]["versions"]):
        raise ValueError("Projection source/runtime contract differs")
    stop()
    if file_hash(PROJECTION) != completion_hash:
        raise ValueError("Projection completion changed while reading")
    identity.update(projection_completion=dict(path=str(PROJECTION), sha256=completion_hash),
                    projection_input_sha256=manifest, projection_contract=contract)
    stop()
    return identity


def smoothing_noise(rows, generator):
    return torch.randn((rows, 2), generator=generator, device="cpu", dtype=torch.float32).mul(
        critic.TD3.policy_noise).clamp(-critic.TD3.noise_clip, critic.TD3.noise_clip)


def paired_indices(uniform_rng, replacement_rng):
    uniform = np.stack([uniform_rng.integers(150, size=8) for _ in SCENARIOS])
    quota = uniform.copy()
    quota[:, 0] = np.asarray(terminal.TERMINALS)[replacement_rng.integers(2, size=5)]
    offsets = np.arange(5, dtype=np.int64)[:, None] * 150
    return {name: torch.from_numpy((values + offsets).reshape(-1))
            for name, values in zip(ARMS, (uniform, quota))}


def frozen_fingerprints(learner):
    result = {}
    for name in ("actor", "actor_target"):
        module = getattr(learner, name)
        result[name] = fingerprint(dict(
            state=module.state_dict(), modes={k: m.training for k, m in module.named_modules()},
            controls={k: (p.requires_grad, p.grad) for k, p in module.named_parameters()}))
    result["actor_optimizer"] = fingerprint(learner.actor_optimizer.state_dict())
    return result


def clone_arms(original):
    arms = {}
    source = fingerprint(original.state_dict())
    for name in ARMS:
        learner = deepcopy(original)
        if fingerprint(learner.state_dict()) != source:
            raise ValueError("Full initial learner copy differs")
        learner.actor_target.load_state_dict(original.actor.state_dict())
        for module in (learner.actor, learner.actor_target):
            module.requires_grad_(False).eval()
            module.zero_grad(set_to_none=True)
        arms[name] = learner
    if terminal.learner_fingerprint(arms[ARMS[0]]) != terminal.learner_fingerprint(arms[ARMS[1]]):
        raise ValueError("Paired initial learner states differ")
    return arms


def finite_state(learner):
    for name in ("actor", "actor_target", "critics", "critic_targets"):
        for p in getattr(learner, name).parameters():
            if not torch.isfinite(p).all() or (p.grad is not None and not torch.isfinite(p.grad).all()):
                raise ValueError("Nonfinite parameter/gradient: " + name)
    for optimizer in (learner.actor_optimizer, learner.critic_optimizer):
        for state in optimizer.state.values():
            if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                   for value in state.values()):
                raise ValueError("Nonfinite Adam state")


def checked_step(learner, batch, noise, update, stop):
    stop()
    finite_state(learner)

    def before_adam(optimizer, args, kwargs):
        stop()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learner.critics.parameters()):
            raise ValueError("Missing/nonfinite critic gradient before Adam")

    hook = learner.critic_optimizer.register_step_pre_hook(before_adam)
    try:
        loss = critic.critic_step(learner, batch, noise, TAU, update)
    finally:
        hook.remove()
    finite_state(learner)
    stop()
    return loss


def terminal_metrics(learner, terminals, update):
    result = terminal.metrics(learner.critics, terminals, update, [])
    # Sampling counts come from actual minibatches, not the supervised fit helper.
    for key in ("diagnostic_training_rows", "training_draws_by_scenario", "total_training_draws"):
        result.pop(key)
    result["per_scenario"] = {}
    for scenario in SCENARIOS:
        rows = [row for row in result["rows"] if row["scenario"] == scenario]
        result["per_scenario"][scenario] = {
            name: dict(mae=sum(row["abs_errors"][name] for row in rows) / len(rows),
                       max_abs_error=max(row["abs_errors"][name] for row in rows))
            for name in ("q1", "q2", "min_q")}
    return result


def fit_seed(original, state, seed, stop, record, *, updates=UPDATES, log_updates=LOG_UPDATES):
    """Internal shorter budgets are for synthetic tests only."""
    if (type(updates) is not int or updates < 1 or not log_updates or
            tuple(sorted(set(log_updates))) != tuple(log_updates) or
            log_updates[0] != 0 or log_updates[-1] != updates):
        raise ValueError("Invalid update/log budget")
    for key, expected in dict(gamma=1., tau=TAU, policy_delay=2, learning_rate=0.0003,
                              policy_noise=0.2, noise_clip=0.5).items():
        if (state["spec"][key] != expected or getattr(original, key) != expected or
                getattr(critic.TD3, key) != expected):
            raise ValueError("Original TD3 specification differs: " + key)
    stop()
    terminals = terminal.extract_terminals(state)
    data = critic.replay_tensors(state)
    data_before = fingerprint(data)
    arms = clone_arms(original)
    initial = terminal.learner_fingerprint(arms[ARMS[0]])
    uniform_rng = np.random.default_rng(seed)
    replacement_rng = np.random.default_rng(seed + 100000)
    noise_rng = torch.Generator(device="cpu").manual_seed(seed)
    eval_noise = smoothing_noise(750, torch.Generator(device="cpu").manual_seed(EVALUATION_SEED))
    digests = {name: hashlib.sha256() for name in (*ARMS, "noise")}
    result = dict(seed=seed, replacement_seed=seed + 100000, noise_seed=seed,
                  evaluation_seed=EVALUATION_SEED, evaluation_noise_sha256=fingerprint(eval_noise),
                  original_state_sha256=fingerprint(original.state_dict()),
                  initial_state_sha256=initial, arms={})
    frozen = {name: frozen_fingerprints(value) for name, value in arms.items()}
    unchanged = {name: fingerprint({k: v for k, v in value.state_dict().items()
                                   if k not in ("critics", "critic_targets", "critic_optimizer")})
                 for name, value in arms.items()}
    for name, value in arms.items():
        finite_state(value)
        result["arms"][name] = dict(
            initial_state_sha256=initial, frozen_initial=frozen[name],
            initial_critic_sha256=fingerprint(value.critics.state_dict()),
            initial_critic_target_sha256=fingerprint(value.critic_targets.state_dict()),
            initial_critic_optimizer_sha256=fingerprint(value.critic_optimizer.state_dict()),
            online_critic_updates=0, target_critic_updates=0, actor_updates=0, actor_target_updates=0,
            sample_counts=dict.fromkeys(SCENARIOS, 0), terminal_counts=dict.fromkeys(SCENARIOS, 0),
            total_samples=0, checkpoints=[])
    loss_sums = dict.fromkeys(ARMS, 0.)
    losses = dict.fromkeys(ARMS, None)

    def snapshot(update):
        for name, value in arms.items():
            stop()
            arm = result["arms"][name]
            arm["index_stream_sha256"] = digests[name].hexdigest()
            arm["noise_stream_sha256"] = digests["noise"].hexdigest()
            arm["checkpoints"].append(dict(
                update=update, online_critic_updates=arm["online_critic_updates"],
                target_critic_updates=arm["target_critic_updates"],
                sample_counts=deepcopy(arm["sample_counts"]), total_samples=arm["total_samples"],
                terminal_counts=deepcopy(arm["terminal_counts"]),
                index_stream_sha256=arm["index_stream_sha256"], noise_stream_sha256=arm["noise_stream_sha256"],
                last_batch_td_loss=losses[name], mean_training_td_loss=loss_sums[name] / update if update else None,
                diagnostics=critic.diagnostics(value, data, eval_noise),
                terminals=terminal_metrics(value, terminals, update)))
        json.dumps(result, allow_nan=False)
        stop()
        record(result)

    started = time.perf_counter()
    snapshot(0)
    for update in range(1, updates + 1):
        stop()
        indices = paired_indices(uniform_rng, replacement_rng)
        noise = smoothing_noise(40, noise_rng)
        noise_before = fingerprint(noise)
        digests["noise"].update(noise.numpy().tobytes())
        for name, value in arms.items():
            stop()
            batch = {key: tensor[indices[name]] for key, tensor in data.items()}
            losses[name] = checked_step(value, batch, noise, update, stop)
            if fingerprint(noise) != noise_before:
                raise ValueError("Shared smoothing noise changed")
            loss_sums[name] += losses[name]
            arm = result["arms"][name]
            arm["online_critic_updates"] += 1
            arm["target_critic_updates"] += int(update % 2 == 0)
            arm["total_samples"] += 40
            for i, scenario in enumerate(SCENARIOS):
                arm["sample_counts"][scenario] += 8
                arm["terminal_counts"][scenario] += int(batch["terminated"][i * 8:(i + 1) * 8].sum())
            digests[name].update(indices[name].numpy().tobytes())
        if update in log_updates:
            snapshot(update)
    for name, value in arms.items():
        stop()
        arm = result["arms"][name]
        arm["frozen_final"] = frozen_fingerprints(value)
        after = fingerprint({k: v for k, v in value.state_dict().items()
                             if k not in ("critics", "critic_targets", "critic_optimizer")})
        if arm["frozen_final"] != frozen[name] or after != unchanged[name]:
            raise ValueError("Frozen actor, optimizer, replay or learner controls changed")
        arm["final_critic_sha256"] = fingerprint(value.critics.state_dict())
        arm["final_critic_target_sha256"] = fingerprint(value.critic_targets.state_dict())
        arm["final_critic_optimizer_sha256"] = fingerprint(value.critic_optimizer.state_dict())
    if fingerprint(data) != data_before:
        raise ValueError("Diagnostic replay changed")
    result["elapsed_seconds"] = time.perf_counter() - started
    stop()
    record(result)
    return result


def interpretation(seeds):
    finals = {name: [seed["arms"][name]["checkpoints"][-1]["terminals"]["aggregate"]["min_q"]["mae"]
                     for seed in seeds] for name in ARMS}
    pooled = {name: sum(values) / len(values) for name, values in finals.items()}
    checks = dict(lower_than_uniform_both_seeds=all(q < u for q, u in zip(finals[ARMS[1]], finals[ARMS[0]])),
                  pooled_mae_at_least_50_percent_lower=pooled[ARMS[1]] <= 0.5 * pooled[ARMS[0]],
                  below_initial_both_seeds=all(q < INITIAL_MAE for q in finals[ARMS[1]]))
    return dict(final_min_q_terminal_mae=finals, pooled_min_q_terminal_mae=pooled,
                initial_threshold=INITIAL_MAE, checks=checks, supports_sampling_hypothesis=all(checks.values()),
                caveat=CAVEAT)


def require_empty(output, *, locked=False):
    if output.exists() and (not output.is_dir() or any(
            not (locked and path.name == "runner.lock" and path.is_file()) for path in output.iterdir())):
        raise FileExistsError("Refusing nonempty output: " + str(output))


def run_diagnostic(output, *, updates=UPDATES, log_updates=LOG_UPDATES, command=None):
    output = Path(output).resolve()
    paths = tuple(folder / "STOP" for folder in (REPO, GOAL_ROOT, output))
    stop = lambda: terminal.check_stop(paths)
    started = time.perf_counter()
    run = dict(pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), output=str(output),
               command=list(command or [sys.executable, *sys.argv]), stop_paths=list(map(str, paths)))
    seeds = []
    try:
        stop()
        require_empty(output)
        stop()
        with exclusive_run(output):
            stop()
            require_empty(output, locked=True)

            def write(name, value):
                stop()
                json_save(output / name, value)

            def record(value):
                if seeds and seeds[-1]["seed"] == value["seed"]:
                    seeds[-1] = value
                else:
                    seeds.append(value)
                write("metrics.json", dict(seeds=seeds, caveat=CAVEAT))

            rng_before = terminal.global_rng_fingerprint()
            try:
                with terminal.preserve_rng():
                    stop()
                    terminal.single_thread()
                    identity_before = capture_identity(stop)
                    stop()
                    model, learner = load_base()
                    stop()
                    learner_before = terminal.learner_fingerprint(learner)
                    model_before = fingerprint(model)
                    if capture_identity(stop) != identity_before:
                        raise ValueError("Input/source/runtime identity changed during load")
                    settings = dict(
                        format="sdmpc-terminal-quota-v1", **run, identity_before=identity_before,
                        seeds=list(SEEDS), updates_per_arm_per_seed=updates, log_updates=list(log_updates),
                        arms=list(ARMS), samples_per_scenario=8, batch_size=40, replay_transitions=750,
                        sampling="with replacement; quota replaces first draw only, at least one terminal per scenario",
                        terminal_indices=list(terminal.TERMINALS), evaluation_seed=EVALUATION_SEED,
                        gamma=1., critic_target_tau=TAU, critic_target_period=2,
                        actor_updates=0, actor_target_updates=0, actor_optimizer_updates=0,
                        initialization="independent full final TD3 copies including critic Adam and critic targets; both actors = final online actor",
                        learned_weights_export=False, original_spec=deepcopy(model["learner"]["spec"]),
                        original_learner_before=learner_before, original_model_before=model_before,
                        global_rng_before=rng_before, caveat=CAVEAT,
                        criterion="quota final min-Q terminal MAE lower in both seeds, pooled >=50% lower, both <5.701837813854217")
                    write("settings.json", settings)
                    write("status.json", dict(**run, status="running"))
                    for seed in SEEDS:
                        stop()
                        fit_seed(learner, model["learner"], seed, stop, record,
                                 updates=updates, log_updates=log_updates)
                    write("status.json", dict(**run, status="verifying"))
                    stop()
                    identity_after = capture_identity(stop)
                    if identity_after != identity_before:
                        raise ValueError("Input/source/runtime identity changed during diagnostic")
                    stop()
                    learner_after = terminal.learner_fingerprint(learner)
                    model_after = fingerprint(model)
                    if learner_after != learner_before or model_after != model_before:
                        raise ValueError("Original learner/model changed")
                rng_after = terminal.global_rng_fingerprint()
                if rng_after != rng_before:
                    raise ValueError("Global RNG state changed")
                result = dict(**run, status="completed", elapsed_seconds=time.perf_counter() - started,
                              ended_at=datetime.now(timezone.utc).isoformat(),
                              interpretation=interpretation(seeds), original_learner_before=learner_before,
                              original_learner_after=learner_after, original_model_before=model_before,
                              original_model_after=model_after, global_rng_before=rng_before, global_rng_after=rng_after,
                              identity_after=identity_after,
                              seeds=[dict(seed=s["seed"], arms={name: {k: v for k, v in arm.items() if k != "checkpoints"}
                                                             for name, arm in s["arms"].items()}) for s in seeds])
                stop()
                # Completion is the last write; a stopped/failed attempt never publishes it.
                write("completion.json", result)
                return result
            except terminal.Stopped:
                raise
            except Exception as exc:
                write("status.json", dict(**run, status="failed", reason=f"{type(exc).__name__}: {exc}"))
                raise
    except terminal.Stopped as exc:
        # STOP forbids further writes, including status; completion is authoritative.
        return dict(**run, status="stopped", reason=str(exc), elapsed_seconds=time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    result = run_diagnostic(args.output, command=command)
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
