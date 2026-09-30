"""Paired critic-only replay diagnostic; never exports or evaluates a policy."""

import argparse
from copy import deepcopy
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TD3_SOURCE = ROOT / "work/sdmpc_rl_multi_20260929/td3.py"
MODEL = ROOT / "results/sdmpc_rl_multi_20260929/pilot_v1/train_round1/model_final.pt"
MODEL_SHA256 = "3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650"
UPDATES = 3750
LOG_UPDATES = (0, 375, 750, 1500, 3750)
ARMS = {"A_tau_0.005": 0.005, "B_tau_1": 1.0}
CAVEAT = (
    "Behavior reward-to-go uses exploratory stored actions and is explicitly not "
    "current-policy Q truth. Terminal errors have no continuation ambiguity. "
    "This fixed-replay diagnostic does not establish causal policy superiority."
)

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
sys.path.insert(0, str(ROOT / ".deps-budget"))

import numpy as np
import torch
from torch.nn import functional as F

# Resolve the old implementation by file, independent of other tasks' td3 imports.
_spec = spec_from_file_location("critic_diagnostic_original_td3", TD3_SOURCE)
_td3 = module_from_spec(_spec)
_spec.loader.exec_module(_td3)
TD3, SCENARIOS = _td3.TD3, _td3.SCENARIOS


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_model(path=MODEL):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Frozen final model SHA-256 mismatch")
    # Deserialize precisely the bytes verified above, without reopening the input.
    return torch.load(BytesIO(payload), map_location="cpu", weights_only=False)["learner"], digest


def clone_arms(state):
    arms = {}
    for name in ARMS:
        learner = TD3(state["observation_dim"], state["spec"]["seed"], state["spec"]["hidden"])
        learner.load_state_dict(state)
        learner.actor_target.load_state_dict(state["actor"])
        learner.actor.requires_grad_(False).eval()
        learner.actor_target.requires_grad_(False).eval()
        arms[name] = learner
    return arms


def behavior_returns(rewards, terminated):
    """Undiscounted observed reward-to-go, resetting only at true terminals."""
    result = torch.empty_like(rewards)
    value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        value = float(rewards[index]) + (0.0 if terminated[index] else value)
        result[index] = value
    return result


def replay_tensors(state):
    groups = state["replay"]
    if set(groups) != set(SCENARIOS):
        raise ValueError("Replay must contain the five training scenarios")
    for scenario in SCENARIOS:
        group = groups[scenario]
        done = group["terminated"]
        if (len(group["rewards"]) != 150 or done.dtype != torch.bool
                or done.nonzero().flatten().tolist() != [74, 149]):
            raise ValueError("Need 150 transitions and two complete 75-step episodes per scenario")
        if not torch.equal(group["next_observations"][:-1][~done[:-1]],
                           group["observations"][1:][~done[:-1]]):
            raise ValueError("Stored training episode continuity differs")
    data = {key: torch.cat([groups[s][key] for s in SCENARIOS]).clone()
            for key in groups[SCENARIOS[0]]}
    data["behavior_returns"] = torch.cat([
        behavior_returns(groups[s]["rewards"], groups[s]["terminated"]) for s in SCENARIOS
    ])
    return data


def draw_batch(data, numpy_rng, torch_rng):
    indices = torch.from_numpy(np.concatenate([
        numpy_rng.integers(150, size=8) + 150 * i for i in range(len(SCENARIOS))
    ]))
    batch = {key: value[indices] for key, value in data.items()}
    noise = torch.randn((40, 2), generator=torch_rng, device="cpu", dtype=torch.float32).mul(TD3.policy_noise).clamp(
        -TD3.noise_clip, TD3.noise_clip)
    return indices, batch, noise


@torch.no_grad()
def target_values(learner, batch, noise):
    actions = (learner.actor_target(batch["next_observations"]) + noise).clamp(-1.0, 1.0)
    inputs = torch.cat((batch["next_observations"], actions), dim=1)
    minimum_q = torch.minimum(*(critic(inputs).flatten() for critic in learner.critic_targets))
    return batch["rewards"] + TD3.gamma * torch.where(
        batch["terminated"], torch.zeros_like(minimum_q), minimum_q)


def critic_step(learner, batch, noise, tau, update):
    targets = target_values(learner, batch, noise)
    inputs = torch.cat((batch["observations"], batch["actions"]), dim=1)
    loss = sum(F.mse_loss(critic(inputs).flatten(), targets) for critic in learner.critics)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite critic loss")
    learner.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    learner.critic_optimizer.step()
    if update % 2 == 0:
        with torch.no_grad():
            for online, target in zip(learner.critics.parameters(), learner.critic_targets.parameters()):
                target.mul_(1.0 - tau).add_(online, alpha=tau)
    return float(loss.detach())


@torch.no_grad()
def diagnostics(learner, data, noise):
    inputs = torch.cat((data["observations"], data["actions"]), dim=1)
    q1, q2 = [critic(inputs).flatten() for critic in learner.critics]
    targets = target_values(learner, data, noise)

    def summarize(mask):
        terminal = mask & data["terminated"]
        heads = {}
        for name, q in (("q1", q1), ("q2", q2), ("minimum_q", torch.minimum(q1, q2))):
            gap = q[mask] - data["behavior_returns"][mask]
            error = (q[terminal] - data["rewards"][terminal]).abs()
            heads[name] = dict(
                q_mean=float(q[mask].mean()), behavior_return_gap_mean=float(gap.mean()),
                behavior_return_gap_rmse=float(gap.square().mean().sqrt()),
                terminal_abs_error_mean=float(error.mean()) if len(error) else None,
                terminal_abs_error_max=float(error.max()) if len(error) else None)
        return dict(transitions=int(mask.sum()), true_terminals=int(terminal.sum()),
                    total_td_loss=float(sum((q[mask] - targets[mask]).square().mean() for q in (q1, q2))),
                    behavior_return_mean=float(data["behavior_returns"][mask].mean()), critics=heads)

    index = torch.arange(len(q1))
    local_index = index % 150
    time_index = local_index % 75
    per_scenario = {}
    for i, scenario in enumerate(SCENARIOS):
        mask = index // 150 == i
        row = summarize(mask)
        row["time_slices"] = {
            name: summarize(mask & (time_index >= start) & (time_index < stop))
            for name, start, stop in (("early_0_24", 0, 25), ("middle_25_49", 25, 50), ("late_50_74", 50, 75))
        }
        row["episodes"] = {str(ep): summarize(mask & (local_index // 75 == ep)) for ep in (0, 1)}
        per_scenario[scenario] = row
    return dict(total=summarize(torch.ones_like(index, dtype=torch.bool)), per_scenario=per_scenario)


def runtime_record():
    return dict(python=platform.python_version(), executable=sys.executable,
                platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                numpy_path=np.__file__, torch_path=torch.__file__, device="cpu", dtype="float32",
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                thread_environment={key: os.environ[key] for key in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


def run_diagnostic(state, output, seed=6529, stop_paths=(), *, updates=UPDATES,
                   log_updates=LOG_UPDATES, provenance=None):
    """The CLI fixes the full budget; shorter budgets exist only for synthetic tests."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = deepcopy(state)
    arms = clone_arms(state)
    data = replay_tensors(state)
    numpy_rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device="cpu").manual_seed(seed)
    # Fixed evaluation noise uses its own stream and never consumes training RNG.
    eval_rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    eval_noise = torch.randn((750, 2), generator=eval_rng, device="cpu", dtype=torch.float32).mul(TD3.policy_noise).clamp(
        -TD3.noise_clip, TD3.noise_clip)
    source_hashes = {str(path.resolve()): file_hash(path) for path in (Path(__file__), TD3_SOURCE)}
    metadata = dict(
        format="sdmpc-critic-temporal-diagnostic-v1", input=provenance, source_sha256=source_hashes,
        runtime=runtime_record(), seed=seed, evaluation_noise_seed=seed + 1,
        parameters=dict(updates=updates, log_updates=list(log_updates), batch_size=40,
                        samples_per_scenario_per_batch=8, sampling="with replacement",
                        gamma=TD3.gamma, learning_rate=TD3.learning_rate,
                        policy_noise=TD3.policy_noise, noise_clip=TD3.noise_clip,
                        arms=ARMS, critic_target_period=2, actor_updates=0, actor_target_updates=0,
                        actor_target_initialization="final actor in both arms",
                        critic_optimizer_initialization="frozen final checkpoint in both arms",
                        initial_learner_updates=state["updates"], original_td3_spec=state["spec"]),
        replay_counts=dict.fromkeys(SCENARIOS, 150), true_terminals=dict.fromkeys(SCENARIOS, 2),
        td_loss_definition="Sum of twin-critic MSEs over all 750 transitions with one fixed shared smoothing-noise draw.",
        time_slices="Zero-based control index within each 75-step episode, pooled over both collection rounds.",
        caveat=CAVEAT)
    stop_paths = tuple(map(Path, stop_paths)) + tuple(
        folder / "STOP" for folder in (output, output.parent, output.parent.parent))
    completed, last_logged = 0, -1
    losses = dict.fromkeys(ARMS, None)
    loss_sums = dict.fromkeys(ARMS, 0.0)
    batch_digest = hashlib.sha256()
    started = time.perf_counter()
    with (output / "diagnostics.jsonl").open("x", encoding="utf-8") as stream:
        def write(row):
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

        def snapshot():
            nonlocal last_logged
            row = dict(kind="checkpoint", updates=completed, target_updates=completed // 2,
                       sample_counts=dict.fromkeys(SCENARIOS, completed * 8),
                       total_samples=completed * 40, paired_batch_sha256=batch_digest.hexdigest(), arms={})
            for name, learner in arms.items():
                row["arms"][name] = dict(
                    **diagnostics(learner, data, eval_noise), last_batch_td_loss=losses[name],
                    mean_training_td_loss=loss_sums[name] / completed if completed else None)
            write(row)
            last_logged = completed

        write(dict(kind="metadata", **metadata))
        snapshot()
        status = "completed"
        while completed < updates:
            if any(path.exists() for path in stop_paths):
                status = "stopped"
                break
            indices, batch, noise = draw_batch(data, numpy_rng, torch_rng)
            for name, tau in ARMS.items():
                losses[name] = critic_step(arms[name], batch, noise, tau, completed + 1)
                loss_sums[name] += losses[name]
            completed += 1
            batch_digest.update(indices.numpy().tobytes())
            batch_digest.update(noise.numpy().tobytes())
            if completed in log_updates:
                snapshot()
        if completed != last_logged:
            snapshot()
        for learner in arms.values():
            for module in (learner.actor, learner.actor_target):
                if any(not torch.equal(value, state["actor"][key]) for key, value in module.state_dict().items()):
                    raise ValueError("Fixed actor changed")
        if source_hashes != {path: file_hash(path) for path in source_hashes}:
            raise ValueError("Diagnostic source changed during execution")
        if provenance and file_hash(provenance["path"]) != provenance["sha256"]:
            raise ValueError("Frozen model changed during execution")
        summary = dict(status=status, updates=completed, target_updates=completed // 2,
                       sample_counts=dict.fromkeys(SCENARIOS, completed * 8), total_samples=completed * 40,
                       paired_batch_sha256=batch_digest.hexdigest(), actors_unchanged=True,
                       source_unchanged=True, input_unchanged=True if provenance else None,
                       elapsed_wall_seconds=time.perf_counter() - started, caveat=CAVEAT)
        write(dict(kind="summary", **summary))
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=6529)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state, digest = load_frozen_model()
    result = run_diagnostic(state, args.output, args.seed,
                            stop_paths=() if args.stop_file is None else (args.stop_file,),
                            provenance=dict(path=str(MODEL.resolve()), sha256=digest))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
