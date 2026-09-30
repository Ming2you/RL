"""Centralized stratified TD3 updates after five complete collection episodes."""
import argparse
import copy
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from budget_runtime import boot, DEFAULT_SNAPSHOT, read, save
from run_budget import (EXPERIENCE_FORMAT, POLICY_FORMAT, checkpoint_save, exclusive_run,
                        file_hash, pins, runtime_versions, stopped, validate_policy, verify_pins)

TRAIN_FORMAT = "sdmpc-multi-training-round-v1"


def verify_experience(payload, settings, schema, trace):
    import numpy as np
    if payload.get("format") != EXPERIENCE_FORMAT or payload.get("settings") != settings:
        raise ValueError("Experience provenance mismatch")
    expected = dict(source_pins=settings["source_pins"], runtime_versions=settings["runtime_versions"],
        environment_contract=settings["environment_contract"], observation_schema=schema,
        reward_divisor=100., gamma=1.)
    if payload.get("contract") != expected or len(payload["transitions"]) != 75 or len(trace) != 75:
        raise ValueError("Experience contract/trajectory length differs")
    previous_next = None
    for transition, row in zip(payload["transitions"], trace):
        obs, action, reward, next_obs, terminal = transition
        for array in (obs, next_obs):
            if np.asarray(array).shape != (len(schema["names"]),) or not np.isfinite(array).all():
                raise ValueError("Experience observation shape/value differs")
        if previous_next is not None:
            np.testing.assert_array_equal(obs, previous_next)
        previous_next = next_obs
        np.testing.assert_array_equal(action, np.asarray(row["action_requested"], dtype=np.float32))
        if (type(terminal) is not bool or terminal != row["terminated"] or
                not np.isfinite(reward) or abs(reward - row["reward"]) > 1e-8):
            raise ValueError("Experience reward/terminal differs from actual trace")
    return expected


def load_collections(folders, round_index, source_pins, runtime, model_hash):
    import torch
    from td3 import SCENARIOS
    from compare_runs import load_completed_run
    if len(folders) != 5 or len({p.resolve() for p in folders}) != 5:
        raise ValueError("Need five distinct collection outputs")
    contracts, runs, run_ids = [], {}, set()
    for folder in folders:
        settings = read(folder / "settings.json")
        scenario = settings["scenario"]
        if scenario not in SCENARIOS or scenario in runs:
            raise ValueError("Missing/duplicate scenario collection")
        seed = (6301 if round_index == 0 else 6401) + SCENARIOS.index(scenario)
        result, settings, traces, schema = load_completed_run(
            folder, "collect", [seed], source_pins, runtime, model_hash, scenario)
        if (settings["round"] != round_index or settings["model_sha256"] != model_hash or
                settings["run_id"] in run_ids or settings["policy_seed"] != 6300):
            raise ValueError("Collection round/shared policy identity differs")
        run_ids.add(settings["run_id"])
        payload = torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)
        contract = verify_experience(payload, settings, schema, traces[0])
        contracts.append(contract)
        runs[scenario] = dict(transitions=payload["transitions"],
            provenance=dict(scenario=scenario, seed=seed, profile_sha256=settings["profile_sha256"][0],
                run_id=settings["run_id"], experience_sha256=result["experience_sha256"]),
            identity=dict(folder=str(folder.resolve()), completion_sha256=file_hash(folder / "completion.json"),
                          experience_sha256=result["experience_sha256"], settings_sha256=file_hash(folder / "settings.json")))
    if set(runs) != set(SCENARIOS) or any(c != contracts[0] for c in contracts):
        raise ValueError("Collectors do not share one policy contract/schema")
    return contracts[0], runs


def validate_metric(row, updates):
    from td3 import SCENARIOS
    expected = {"updates", "sampled_per_scenario", "critic_loss"}
    if updates % 2 == 0:
        expected.add("actor_loss")
    if (not isinstance(row, dict) or set(row) != expected or
            type(row["updates"]) is not int or row["updates"] != updates or
            row["sampled_per_scenario"] != dict.fromkeys(SCENARIOS, 8) or
            any(type(v) is not int for v in row["sampled_per_scenario"].values()) or
            any(type(row[k]) not in (int, float) or not math.isfinite(row[k])
                for k in expected if k.endswith("_loss"))):
        raise ValueError("Training metric phase/sample/loss contract differs")


def reconcile_training(learner, expected_replay, round_index, completed, metrics):
    import torch
    from td3 import SCENARIOS
    if type(completed) is not int or not 0 <= completed <= 375 or len(metrics) != completed:
        raise ValueError("Invalid training progress")
    updates = 375 * round_index + completed
    if (learner.seed != 6300 or learner.updates != updates or
            learner.sample_counts != dict.fromkeys(SCENARIOS, updates * 8) or
            learner.replay_counts() != dict.fromkeys(SCENARIOS, 75 * (round_index + 1))):
        raise ValueError("Training seed/replay/update/sample contract differs")
    actual = learner.state_dict()["replay"]
    for scenario in SCENARIOS:
        for column, expected in expected_replay[scenario].items():
            value = actual[scenario][column]
            if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
                raise ValueError("Training replay differs from admitted inputs: " + scenario + "/" + column)
    for i, row in enumerate(metrics):
        validate_metric(row, 375 * round_index + i + 1)


def training_inputs(contract, runs, round_index, model_path):
    import torch
    from td3 import TD3, SCENARIOS
    learner, provenance = TD3(len(contract["observation_schema"]["names"]), 6300), []
    if round_index == 1:
        model_path = Path(model_path)
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        validate_policy(model, contract)
        if model.get("training_round") != 0 or len(model["training_profiles"]) != 5:
            raise ValueError("Wrong predecessor training round")
        if model_path.resolve() != (model_path.parent / "model_final.pt").resolve():
            raise ValueError("Predecessor must be the admitted training output")
        validate_training_output(model_path.parent, 0, contract["source_pins"], contract["runtime_versions"])
        learner.load_state_dict(model["learner"])
        provenance = copy.deepcopy(model["training_profiles"])
    elif round_index != 0 or model_path is not None:
        raise ValueError("Wrong predecessor training round")
    for scenario in SCENARIOS:
        for transition in runs[scenario]["transitions"]:
            learner.add(*transition, scenario=scenario)
        provenance.append(copy.deepcopy(runs[scenario]["provenance"]))
    return learner, provenance


def validate_training_output(folder, round_index, source_pins, runtime, model_path=None, collections=None):
    import torch
    from td3 import SCENARIOS
    folder = Path(folder)
    result, settings = read(folder / "completion.json"), read(folder / "settings.json")
    previous_hash = None if model_path is None else file_hash(model_path)
    if collections is None:
        collections = [Path(settings["collections"][s]["folder"]) for s in SCENARIOS]
    contract, runs = load_collections(collections, round_index, source_pins, runtime, previous_hash)
    if (result["format"] != TRAIN_FORMAT or result["status"] != "completed" or result["settings"] != settings or
            settings["format"] != TRAIN_FORMAT or not settings["run_id"] or settings["policy_seed"] != 6300 or
            settings["round"] != round_index or settings["collections"] != {s: runs[s]["identity"] for s in SCENARIOS} or
            settings["contract"] != contract or settings["model_sha256"] != previous_hash or
            settings["source_pins"] != source_pins or settings["runtime_versions"] != runtime or
            settings["updates"] != 375 or settings["batch_size"] != 40 or settings["updates_per_new_transition"] != 1 or
            result["model_sha256"] != file_hash(folder / "model_final.pt")):
        raise ValueError("Completed central training contract differs")
    learner, provenance = training_inputs(contract, runs, round_index, model_path)
    expected_replay = learner.state_dict()["replay"]
    model = torch.load(folder / "model_final.pt", map_location="cpu", weights_only=False)
    validate_policy(model, contract)
    if (model.get("training_run_id") != settings["run_id"] or model.get("training_round") != round_index or
            model.get("training_profiles") != provenance):
        raise ValueError("Completed training provenance differs from admitted inputs")
    learner.load_state_dict(model["learner"])
    reconcile_training(learner, expected_replay, round_index, 375, read(folder / "metrics.json"))
    if (result["replay_counts"] != learner.replay_counts() or result["learner_updates"] != learner.updates or
            result["sampled_per_scenario"] != dict.fromkeys(SCENARIOS, 3000)):
        raise ValueError("Completed replay/update balance differs")
    return result


def q_diagnostics(learner):
    import numpy as np
    import torch
    from td3 import SCENARIOS
    result = {}
    for scenario in SCENARIOS:
        data = learner.state_dict()["replay"][scenario]
        with torch.no_grad():
            inputs = torch.cat((data["observations"], data["actions"]), dim=1)
            q = torch.minimum(*(c(inputs) for c in learner.critics)).ravel()
            action = learner.actor(data["observations"])
        rewards, done = data["rewards"].numpy(), data["terminated"].numpy()
        returns, value = np.zeros(len(rewards)), 0.
        for i in range(len(rewards)-1, -1, -1):
            value = float(rewards[i]) + (0. if done[i] else value)
            returns[i] = value
        terminal_errors = abs(q.numpy()[done] - rewards[done])
        result[scenario] = dict(transitions=len(rewards), true_terminals=int(done.sum()),
            q_mean=float(q.mean()), behavior_return_mean=float(returns.mean()),
            q_minus_behavior_return_mean=float(np.mean(q.numpy()-returns)),
            terminal_abs_error_mean=float(terminal_errors.mean()),
            terminal_abs_error_max=float(terminal_errors.max()),
            actor_mean=action.mean(dim=0).tolist(),
            actor_abs_ge_095_fraction=(action.abs() >= .95).float().mean(dim=0).tolist())
    return dict(per_scenario=result,
        caveat="Exploratory behavior returns are not current-policy Q ground truth; terminal reward errors have no continuation ambiguity.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--collections", type=Path, nargs=5, required=True)
    p.add_argument("--round", type=int, choices=(0, 1), required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    if (args.round == 1) != (args.model is not None):
        p.error("Only round1 continues the prior shared model")
    with exclusive_run(args.output):
        if (args.output / "completion.json").exists():
            raise ValueError("Training round already completed")
        if (args.output / "checkpoint.pt").exists() != args.resume:
            raise ValueError("Training resume checkpoint mismatch")
        source_pins = pins(args.root)
        boot(args.root, 1)
        import torch
        from td3 import SCENARIOS
        versions = runtime_versions()
        model_hash = None if args.model is None else file_hash(args.model)
        contract, runs = load_collections(args.collections, args.round, source_pins, versions, model_hash)
        existing_settings = (args.output / "settings.json").exists()
        settings = dict(format=TRAIN_FORMAT, round=args.round, policy_seed=6300,
            run_id=read(args.output / "settings.json")["run_id"] if existing_settings else uuid.uuid4().hex,
            source_pins=source_pins, runtime_versions=versions, contract=contract, model_sha256=model_hash,
            collections={s: runs[s]["identity"] for s in SCENARIOS},
            updates=375, batch_size=40, updates_per_new_transition=1)
        if existing_settings and read(args.output / "settings.json") != settings:
            raise ValueError("Training settings changed")
        learner, provenance = training_inputs(contract, runs, args.round, args.model)
        expected_replay = learner.state_dict()["replay"]
        completed, metrics, elapsed_before = 0, [], 0.
        if args.resume:
            ck = torch.load(args.output / "checkpoint.pt", map_location="cpu", weights_only=False)
            if (ck.get("format") != TRAIN_FORMAT or ck.get("settings") != settings or
                    ck.get("training_run_id") != settings["run_id"] or ck.get("training_profiles") != provenance):
                raise ValueError("Training checkpoint contract differs")
            completed, metrics, elapsed_before = ck["completed"], ck["metrics"], ck["elapsed_wall_seconds"]
            learner.load_state_dict(ck["learner"])
        else:
            save(args.output / "settings.json", settings)
        reconcile_training(learner, expected_replay, args.round, completed, metrics)
        save(args.output / "process.json", dict(pid=os.getpid(), command=sys.argv, started=time.time()))
        started = time.perf_counter()

        def checkpoint():
            checkpoint_save(torch, args.output / "checkpoint.pt", dict(format=TRAIN_FORMAT,
                settings=settings, completed=completed, metrics=metrics, learner=learner.state_dict(),
                training_run_id=settings["run_id"], training_profiles=provenance,
                elapsed_wall_seconds=elapsed_before+time.perf_counter()-started))

        try:
            checkpoint()
            while completed < 375:
                if stopped(args.output):
                    checkpoint()
                    save(args.output / "status.json", dict(status="paused", updates=completed))
                    return
                row = learner.update(40)
                validate_metric(row, 375 * args.round + completed + 1)
                metrics.append(row)
                completed += 1
                if completed % 25 == 0:
                    verify_pins(args.root, source_pins)
                    checkpoint()
                    save(args.output / "status.json", dict(status="running", updates=completed))
            verify_pins(args.root, source_pins)
            if args.model is not None and file_hash(args.model) != model_hash:
                raise ValueError("Predecessor checkpoint changed")
            reconcile_training(learner, expected_replay, args.round, completed, metrics)
            model = dict(format=POLICY_FORMAT, learner=learner.state_dict(), contract=contract,
                training_run_id=settings["run_id"], training_profiles=provenance, training_round=args.round)
            validate_policy(model, contract)
            checkpoint_save(torch, args.output / "model_final.pt", model)
            model_hash = file_hash(args.output / "model_final.pt")
            save(args.output / "metrics.json", metrics)
            save(args.output / "q_diagnostics.json", q_diagnostics(learner))
            save(args.output / "completion.json", dict(format=TRAIN_FORMAT, status="completed", settings=settings,
                model_sha256=model_hash, replay_counts=learner.replay_counts(), learner_updates=learner.updates,
                sampled_per_scenario={s: sum(row["sampled_per_scenario"][s] for row in metrics) for s in SCENARIOS},
                elapsed_wall_seconds=elapsed_before+time.perf_counter()-started))
            save(args.output / "status.json", dict(status="completed"))
            print("SHARED_TRAINING_COMPLETE", args.round, learner.replay_counts(), learner.updates)
        except Exception:
            save(args.output / "failure.json", dict(traceback=traceback.format_exc(), completed_updates=completed))
            save(args.output / "status.json", dict(status="failed"))
            raise


if __name__ == "__main__":
    main()
