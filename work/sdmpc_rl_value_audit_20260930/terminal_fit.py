"""Bounded terminal-reward fitting only; no plant, policy export, or evaluation."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"

_spec = spec_from_file_location(
    "terminal_fit_common", HERE.parent / "sdmpc_rl_recovery_20260929/common.py")
common = module_from_spec(_spec)
_spec.loader.exec_module(common)
load_base, BASE, BASE_HASH, REPO = common.load_base, common.BASE, common.BASE_HASH, common.REPO
TD3, SCENARIOS, torch = common.TD3, common.SCENARIOS, common.torch
save, file_hash = common.save, common.file_hash
runtime_versions, exclusive_run = common.runtime_versions, common.exclusive_run
import numpy as np

GOAL_ROOT = REPO / "results/sdmpc_rl_balanced_goal_20260930"
UPDATES = 1000
LOG_UPDATES = (0, 50, 250, 1000)
FRESH_SEED = 8100
LEARNING_RATE = 0.0003
TERMINALS = (74, 149)
TOLERANCE = 0.01
ARMS = ("original_critics", "fresh_critics")
HOLDOUT_NOTES = {
    "original_critics": (
        "The original critic previously saw both folds during joint TD3 training. "
        "Diagnostic holdout excludes only these extra supervised updates."),
    "fresh_critics": (
        "Fresh-critic holdout is untrained in this diagnostic. There is only one "
        "sample per scenario; no out-of-sample generalization claim."),
}
CAVEAT = (
    "Train max absolute error <=0.01 reward units is descriptive per critic. "
    "Holdout errors are diagnostic only, never used for updates, stopping, arm "
    "selection, or policy admission. This experiment does not test traffic control.")


def json_save(path, value):
    # The existing atomic writer permits NaN; validate before it touches a file.
    json.dumps(value, allow_nan=False)
    save(path, value)


def fingerprint(value):
    """Hash in-memory state without serializing a model or advancing any RNG."""
    def plain(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            return dict(dtype=str(item.dtype), shape=list(item.shape),
                        sha256=hashlib.sha256(item.tobytes()).hexdigest())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(v) for v in item]
        return item
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def learner_fingerprint(learner):
    controls = {}
    for name in ("actor", "actor_target", "critics", "critic_targets"):
        module = getattr(learner, name)
        controls[name] = dict(
            modes={key: child.training for key, child in module.named_modules()},
            parameters={key: dict(requires_grad=p.requires_grad, grad=p.grad)
                        for key, p in module.named_parameters()})
    return fingerprint(dict(state=learner.state_dict(), controls=controls))


def global_rng_fingerprint():
    return fingerprint((random.getstate(), np.random.get_state(), torch.get_rng_state()))


@contextmanager
def preserve_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def single_thread():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"


def extract_terminals(state):
    """Validate the stored replay schema and copy only its ten true terminals."""
    dim = state["observation_dim"]
    if type(dim) is not int or dim < 1 or set(state["replay"]) != set(SCENARIOS):
        raise ValueError("Expected a valid observation dimension and exactly five scenarios")
    inputs, targets, rows = [], [], []
    for scenario in SCENARIOS:
        group = state["replay"][scenario]
        shapes = {"observations": (150, dim), "actions": (150, 2), "rewards": (150,),
                  "next_observations": (150, dim), "terminated": (150,)}
        for name, shape in shapes.items():
            value = group.get(name)
            dtype = torch.bool if name == "terminated" else torch.float32
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != dtype or value.device.type != "cpu"
                    or value.layout != torch.strided or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid finite CPU replay shape/dtype: {scenario}/{name}")
        if group["terminated"].nonzero().flatten().tolist() != list(TERMINALS):
            raise ValueError(f"Expected true terminals at rows 74 and 149: {scenario}")
        for round_index, index in enumerate(TERMINALS):
            inputs.append(torch.cat((group["observations"][index], group["actions"][index])))
            targets.append(group["rewards"][index].detach().clone())
            rows.append(dict(scenario=scenario, round=round_index, row_index=index))
    return dict(inputs=torch.stack(inputs).detach().clone(),
                targets=torch.stack(targets).reshape(10, 1), rows=rows)


def fold_data(data, train_round):
    if train_round not in (0, 1):
        raise ValueError("Only collection rounds 0 and 1 are available")
    def select(round_index):
        indices = [i for i, row in enumerate(data["rows"]) if row["round"] == round_index]
        rows = [deepcopy(data["rows"][i]) for i in indices]
        if [row["scenario"] for row in rows] != list(SCENARIOS):
            raise ValueError("Each fold must have exactly one row per scenario")
        return dict(inputs=data["inputs"][indices].clone(),
                    targets=data["targets"][indices].clone(), rows=rows)
    return select(train_round), select(1 - train_round)


def fresh_critics(learner):
    with preserve_rng():
        return deepcopy(TD3(learner.observation_dim, FRESH_SEED, learner.hidden).critics)


def critic_copy(template):
    critics = deepcopy(template).requires_grad_(True).train()
    for parameter in critics.parameters():
        parameter.grad = None
    optimizer = torch.optim.Adam(critics.parameters(), lr=LEARNING_RATE)
    return critics, optimizer


def predictions(critics, data):
    if len(critics) != 2:
        raise ValueError("Expected twin critics")
    values = [critic(data["inputs"]) for critic in critics]
    if any(value.shape != data["targets"].shape or not torch.isfinite(value).all()
           for value in values):
        raise ValueError("Invalid/nonfinite critic predictions")
    return values


def critic_step(critics, optimizer, train):
    values = predictions(critics, train)
    loss = sum(torch.nn.functional.mse_loss(value, train["targets"]) for value in values)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite supervised loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in critics.parameters()):
        raise ValueError("Missing/nonfinite critic gradient")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in critics.parameters()):
        raise ValueError("Nonfinite critic parameter after update")
    if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
           for state in optimizer.state.values() for v in state.values()):
        raise ValueError("Nonfinite Adam state")
    return float(loss.detach())


@torch.no_grad()
def metrics(critics, data, update, train_rows):
    q1, q2 = predictions(critics, data)
    heads = {"q1": q1, "q2": q2, "min_q": torch.minimum(q1, q2)}
    rows = [dict(**row, target=float(data["targets"][i, 0]), predictions={}, errors={},
                 abs_errors={}) for i, row in enumerate(data["rows"])]
    aggregate = {}
    for name, values in heads.items():
        error = values.double() - data["targets"].double()
        aggregate[name] = dict(mse=float(error.square().mean()), mae=float(error.abs().mean()),
                               max_abs_error=float(error.abs().max()))
        for i, row in enumerate(rows):
            row["predictions"][name] = float(values[i, 0])
            row["errors"][name] = float(error[i, 0])
            row["abs_errors"][name] = float(error[i, 0].abs())
    result = dict(update=update, sample_count=len(rows), rows=rows, aggregate=aggregate,
                  diagnostic_training_rows=deepcopy(train_rows),
                  training_draws_by_scenario=dict.fromkeys(SCENARIOS, update),
                  total_training_draws=5 * update)
    json.dumps(result, allow_nan=False)
    return result


class Stopped(RuntimeError):
    pass


def check_stop(paths):
    found = [str(path) for path in paths if path.exists()]
    if found:
        raise Stopped("STOP present: " + ", ".join(found))


def capture_identity(stop):
    """Only frozen training input, source pins, and local runtime identities."""
    stop()
    model_path = BASE / "train_round1/model_final.pt"
    model_hash = file_hash(model_path)
    if model_hash != BASE_HASH:
        raise ValueError("Frozen base model SHA-256 differs")
    stop()
    expected_folder = REPO / "work/sdmpc_rl_multi_20260929"
    for helper in (TD3, save, file_hash, runtime_versions, exclusive_run):
        if Path(sys.modules[helper.__module__].__file__).resolve().parent != expected_folder:
            raise ValueError("Mixed runtime helper modules")
    sources = (Path(__file__), Path(common.__file__), HERE / "terminal-brief.md")
    runtime = dict(
        versions=runtime_versions(), executable=sys.executable,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        thread_environment={name: os.environ[name] for name in THREAD_VARIABLES},
        entrypoint_sha256={str(Path(path).resolve()): file_hash(path) for path in
                           (sys.executable, torch.__file__, torch._C.__file__, np.__file__)})
    identity = dict(
        input_sha256={str(model_path): model_hash,
                      str(BASE / "completion.json"): file_hash(BASE / "completion.json")},
        source_sha256={str(path): file_hash(path) for path in sources},
        base_source_pins=common.pins(common.DEFAULT_SNAPSHOT),
        runtime=runtime, runtime_sha256=fingerprint(runtime))
    stop()
    return identity


def fit(learner, state, result, stop, record, *, updates=UPDATES, log_updates=LOG_UPDATES):
    """Only tests inject a smaller budget; the CLI exposes no budget override."""
    if type(updates) is not int or updates < 1 or tuple(sorted(set(log_updates))) != tuple(log_updates):
        raise ValueError("Invalid fixed update/log budget")
    if not log_updates or log_updates[0] != 0 or log_updates[-1] != updates:
        raise ValueError("Metrics must include update zero and the final update")
    if state["spec"]["learning_rate"] != LEARNING_RATE or learner.learning_rate != LEARNING_RATE:
        raise ValueError("Original learning rate differs from 0.0003")
    stop()
    data = extract_terminals(state)
    result["original_all_terminals"] = metrics(deepcopy(learner.critics), data, 0, [])
    record()
    stop()
    templates = {"original_critics": deepcopy(learner.critics), "fresh_critics": fresh_critics(learner)}
    for train_round in (0, 1):
        stop()
        train, holdout = fold_data(data, train_round)
        fold = dict(train_round=train_round, holdout_round=1 - train_round,
                    training_rows=train["rows"], holdout_rows=holdout["rows"], arms={})
        result["folds"].append(fold)
        for name in ARMS:
            stop()
            critics, optimizer = critic_copy(templates[name])
            arm = dict(holdout_note=HOLDOUT_NOTES[name], updates_completed=0,
                       initialization_sha256=fingerprint(critics.state_dict()),
                       trainable_parameters=[key for key, _ in critics.named_parameters()],
                       training_draws_by_scenario=dict.fromkeys(SCENARIOS, 0), checkpoints=[])
            fold["arms"][name] = arm

            def snapshot(update):
                arm["checkpoints"].append(dict(
                    update=update, train=metrics(critics, train, update, train["rows"]),
                    holdout=metrics(critics, holdout, update, train["rows"])))
                record()

            snapshot(0)
            for update in range(1, updates + 1):
                stop()
                critic_step(critics, optimizer, train)
                arm["updates_completed"] = update
                arm["training_draws_by_scenario"] = dict.fromkeys(SCENARIOS, update)
                stop()
                if update in log_updates:
                    snapshot(update)
            final = arm["checkpoints"][-1]["train"]["aggregate"]
            arm["descriptive_train_max_abs_error_le_0_01"] = {
                head: final[head]["max_abs_error"] <= TOLERANCE for head in ("q1", "q2")}
    stop()
    record()


def run_diagnostic(output, *, updates=UPDATES, log_updates=LOG_UPDATES, command=None):
    output = Path(output).resolve()
    paths = tuple(folder / "STOP" for folder in (output, GOAL_ROOT, REPO))
    run = dict(command=list(command or [sys.executable, *sys.argv]), pid=os.getpid(),
               cwd=str(Path.cwd()), started_at=datetime.now(timezone.utc).isoformat(),
               output=str(output), stop_paths=[str(path) for path in paths])
    try:
        check_stop(paths)
    except Stopped as exc:
        # An occupied output with STOP is left completely untouched.
        return dict(**run, status="stopped", reason=str(exc), output_created=False,
                    ended_at=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True, exist_ok=False)
    with exclusive_run(output):
        status, error, reason = "running", None, None
        identity_before = identity_after = learner_before = learner_after = None
        state_before = state_after = None
        learner = state = None
        result = dict(original_all_terminals=None, folds=[], caveat=CAVEAT)
        settings = dict(
            format="sdmpc-terminal-fit-v1", **run,
            updates_per_arm_per_fold=updates, log_updates=list(log_updates),
            arms=list(ARMS), folds=[dict(train_round=0, holdout_round=1),
                                    dict(train_round=1, holdout_round=0)],
            rows_per_scenario=150, terminal_indices=list(TERMINALS),
            fullbatch_size=5, scenario_contribution=dict.fromkeys(SCENARIOS, 0.2),
            learning_rate=LEARNING_RATE, optimizer="fresh Adam in each arm and fold",
            fresh_seed=FRESH_SEED, fresh_initialization="same TD3 architecture; identical in both folds",
            target="exact stored immediate reward", loss="sum of twin-critic fullbatch MSE",
            bootstrap=False, reward_normalization=False, reward_clipping=False,
            augmentation=False, noise=False, actor_updates=0, target_updates=0,
            learned_weights_export=False, holdout_notes=HOLDOUT_NOTES, caveat=CAVEAT)
        json_save(output / "settings.json", settings)

        def record():
            json_save(output / "metrics.json", result)
            progress = [dict(train_round=fold["train_round"], arm=name,
                             updates_completed=arm["updates_completed"],
                             training_draws_by_scenario=arm["training_draws_by_scenario"])
                        for fold in result["folds"] for name, arm in fold["arms"].items()]
            json_save(output / "status.json", dict(**run, status="running", progress=progress))

        stop = lambda: check_stop(paths)
        rng_before = global_rng_fingerprint()
        try:
            with preserve_rng():
                stop()
                single_thread()
                record()
                identity_before = capture_identity(stop)
                settings["identity_before"] = identity_before
                json_save(output / "settings.json", settings)
                stop()
                model, learner = load_base()
                state = model["learner"]
                learner_before, state_before = learner_fingerprint(learner), fingerprint(state)
                settings["original_spec"] = deepcopy(state["spec"])
                settings["original_learner_sha256"] = learner_before
                json_save(output / "settings.json", settings)
                fit(learner, state, result, stop, record, updates=updates, log_updates=log_updates)
                status = "completed"
        except Stopped as exc:
            status, reason = "stopped", str(exc)
        except Exception as exc:
            status, error, reason = "failed", exc, f"{type(exc).__name__}: {exc}"
        try:
            if identity_before is not None:
                # Finish read-only verification even on STOP; never resume fitting.
                identity_after = capture_identity(lambda: None)
                if identity_before != identity_after:
                    raise ValueError("Input/source/runtime identity changed during diagnostic")
            if learner_before is not None:
                learner_after, state_after = learner_fingerprint(learner), fingerprint(state)
                if learner_before != learner_after or state_before != state_after:
                    raise ValueError("Original learner or model state changed during diagnostic")
            if rng_before != global_rng_fingerprint():
                raise ValueError("Global RNG state changed during diagnostic")
            if status == "completed":
                stop()
        except Stopped as exc:
            status, reason = "stopped", str(exc)
        except Exception as exc:
            status, error, reason = "failed", exc, f"{type(exc).__name__}: {exc}"
        summary = dict(
            **run, status=status, reason=reason, ended_at=datetime.now(timezone.utc).isoformat(),
            identity_after=identity_after,
            immutability=dict(
                input_source_runtime_unchanged=(identity_before == identity_after)
                if identity_before is not None and identity_after is not None else None,
                learner_before_sha256=learner_before, learner_after_sha256=learner_after,
                model_state_before_sha256=state_before, model_state_after_sha256=state_after,
                original_learner_unchanged=(learner_before == learner_after)
                if learner_before is not None and learner_after is not None else None,
                original_model_state_unchanged=(state_before == state_after)
                if state_before is not None and state_after is not None else None,
                global_rng_unchanged=rng_before == global_rng_fingerprint()),
            progress=[dict(train_round=fold["train_round"], arm=name,
                           updates_completed=arm["updates_completed"],
                           training_draws_by_scenario=arm["training_draws_by_scenario"])
                      for fold in result["folds"] for name, arm in fold["arms"].items()],
            caveat=CAVEAT)
        json_save(output / "metrics.json", result)
        json_save(output / "status.json", summary)
        if status == "completed":
            # Recheck after final JSON writes so a late STOP cannot claim completion.
            try:
                stop()
            except Stopped as exc:
                summary.update(status="stopped", reason=str(exc))
                json_save(output / "status.json", summary)
            else:
                json_save(output / "completion.json", summary)
        if error is not None:
            raise error
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    summary = run_diagnostic(args.output, command=command)
    print(json.dumps(summary, allow_nan=False))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
