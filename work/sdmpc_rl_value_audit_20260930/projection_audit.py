"""Audit preserved training transitions and Q on identical projected requests."""
import argparse
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "work/sdmpc_rl_recovery_20260929"))
from common import (BASE, BASE_HASH, REPO, DEFAULT_SNAPSHOT, SCENARIOS, torch,
                    load_base, read, save, file_hash, verify_pins, runtime_versions,
                    exclusive_run)
import numpy as np
from budget_controller import residual_budget
from budget_runtime import boot, digest
from train_round import load_collections

GOAL_ROOT = REPO / "results/sdmpc_rl_balanced_goal_20260930"


def stopped(output):
    return any((folder / "STOP").exists() for folder in (output, GOAL_ROOT, REPO))


def input_manifest():
    paths = [BASE / "completion.json", BASE / "train_round0/model_final.pt",
             BASE / "train_round1/model_final.pt"]
    for round_index in (0, 1):
        for scenario in SCENARIOS:
            folder = BASE / f"collect_round{round_index}" / scenario
            paths.extend(folder / name for name in (
                "completion.json", "settings.json", "experience.pt", "episode_00_trace.json",
                "episode_00_summary.json", "observation_schema.json", "runtime_versions.json"))
    return {str(path.relative_to(REPO)): file_hash(path) for path in paths}


def verify_input_manifest(manifest):
    for name, expected in manifest.items():
        try:
            actual = file_hash(REPO / name)
        except OSError as exc:
            raise ValueError("Input missing or changed: " + name) from exc
        if actual != expected:
            raise ValueError("Input changed: " + name)


def verify_collection_provenance(model, runs, round_index):
    profiles = model.get("training_profiles")
    start = round_index * len(SCENARIOS)
    if (not isinstance(profiles, list) or len(profiles) != 2 * len(SCENARIOS) or
            profiles[start:start + len(SCENARIOS)] != [runs[s]["provenance"] for s in SCENARIOS]):
        raise ValueError("Collection provenance differs from authenticated final model")


def column(names, name):
    if names.count(name) != 1:
        raise ValueError("Missing/duplicate observation field: " + name)
    return names.index(name)


def validate_sequence(transitions, trace, names, capacity):
    if len(transitions) != 75 or len(trace) != 75:
        raise ValueError("Need a complete 75-interval training episode")
    remaining = column(names, "memory/remaining/0")
    clock = column(names, "state/time_sec")
    anchor_columns = [column(names, f"memory/action_anchor/{i}") for i in range(2)]
    memory_columns = {key: [column(names, f"memory/{key}/{i}") for i in range(2)]
                      for key in ("previous_requested", "previous_executed")}
    for i, ((obs, action, reward, next_obs, terminal), row) in enumerate(zip(transitions, trace)):
        if (np.asarray(obs).shape != (len(names),) or np.asarray(next_obs).shape != (len(names),)
                or not np.isfinite(obs).all() or not np.isfinite(next_obs).all()
                or not np.isfinite(reward)):
            raise ValueError("Invalid replay observation or reward")
        if type(terminal) is not bool or terminal != (i == 74) or row["terminated"] != terminal:
            raise ValueError("True terminal differs")
        if row["control_step"] != i or row["time_sec"] != 1080. + 180. * i:
            raise ValueError("Interval clock differs")
        np.testing.assert_equal(obs[remaining], np.float32((75 - i) / 75.))
        np.testing.assert_equal(obs[clock], np.float32((900. + 180. * i) / 14400.))
        np.testing.assert_array_equal(obs[anchor_columns],
            np.asarray(np.asarray(row["action_anchor"]) / [1000., 10000.], dtype=np.float32))
        for key, source in (("previous_requested", "B_requested"), ("previous_executed", "B_executed")):
            expected = np.zeros(2) if i == 0 else np.asarray(trace[i-1][source])
            if i and source == "B_requested":
                expected = expected[0]
            np.testing.assert_array_equal(obs[memory_columns[key]], np.asarray(expected / [1000., 10000.], dtype=np.float32))
        if row["interval_ttt"] < 0 or abs(reward + row["interval_ttt"] / 100.) > 1e-8:
            raise ValueError("Reward differs from actual interval accounting")
        np.testing.assert_array_equal(action, np.asarray(row["action_requested"], dtype=np.float32))
        raw, request = residual_budget(action, row["action_anchor"], capacity)
        np.testing.assert_allclose(raw, row["B_raw"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(request, row["B_requested"][0], rtol=0, atol=1e-10)
        if terminal:
            np.testing.assert_array_equal(next_obs, np.zeros(len(names), dtype=np.float32))
        else:
            np.testing.assert_array_equal(next_obs, transitions[i+1][0])
    return dict(transitions=75, true_terminals=1, horizon_and_clock="matched",
                previous_budget_memory="matched", reward_accounting="matched", projection="matched")


def alias_actions(anchor, capacity, np_action):
    """Use the physical transform itself to prove equality, never a rounded key."""
    threshold = (capacity - float(anchor[1])) / 1000.
    if threshold > 1.:
        return []
    low = np.float32(max(-1., threshold))
    # Select a representable boundary action using the actual physical transform.
    if residual_budget(np.array([np_action, low], dtype=np.float32), anchor, capacity)[1][1] < capacity:
        low = np.nextafter(low, np.float32(1.))
    candidates = [np.array([np_action, value], dtype=np.float32)
                  for value in (low, (float(low)+1.)/2., 1.)]
    groups = []
    for action in candidates:
        _, request = residual_budget(action, anchor, capacity)
        for group in groups:
            if np.array_equal(request, group["request"]):
                if not any(np.array_equal(action, other) for other in group["actions"]):
                    group["actions"].append(action)
                break
        else:
            groups.append(dict(request=request, actions=[action]))
    return [group for group in groups if len(group["actions"]) >= 2]


@torch.no_grad()
def q_aliases(learner, obs, anchor, capacity):
    records = []
    for np_action in (-1., 0., 1.):
        for group in alias_actions(anchor, capacity, np_action):
            actions = torch.unique(torch.tensor(np.stack(group["actions"]), dtype=torch.float32), dim=0)
            if len(actions) < 2:
                continue
            # Conversion to float32 must preserve the physical equivalence.
            requests = [residual_budget(a.numpy(), anchor, capacity)[1] for a in actions]
            if not all(np.array_equal(r, requests[0]) for r in requests):
                continue
            inputs = torch.cat((torch.tensor(obs).repeat(len(actions), 1), actions), dim=1)
            q = torch.stack([critic(inputs).flatten() for critic in learner.critics], dim=1)
            if not torch.isfinite(q).all():
                raise ValueError("Nonfinite alias Q")
            records.append(dict(np_action=np_action, projected_request=requests[0].tolist(),
                nominal_actions=actions.tolist(), twin_q=q.tolist(),
                twin_q_spread=(q.max(dim=0).values-q.min(dim=0).values).tolist()))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if stopped(output):
        print("PROJECTION_AUDIT_STOPPED", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    with exclusive_run(output):
        if stopped(output):
            print("PROJECTION_AUDIT_STOPPED", flush=True)
            return
        if any(path.name != "runner.lock" for path in output.iterdir()):
            raise FileExistsError(output)
        started = time.time()
        save(output / "process.json", dict(pid=os.getpid(), started=started, command=sys.argv))
        source = {str(path.relative_to(REPO)): file_hash(path) for path in (
            Path(__file__), REPO / "work/sdmpc_rl_recovery_20260929/common.py")}
        inputs = input_manifest()
        model, learner = load_base()
        rt = boot(DEFAULT_SNAPSHOT, 1)
        contract = model["contract"]
        if (digest(rt["rc"].to_plain_dict(rt["cfg"])) != contract["environment_contract"]["config_sha256"] or
                digest(rt["rc"].to_plain_dict(rt["options"])) != contract["environment_contract"]["options_sha256"]):
            raise ValueError("Physical configuration differs")
        capacity = rt["cfg"].network.total_ramp_capacity
        settings = dict(base_sha256=BASE_HASH, source_sha256=source, contract=contract,
                        input_sha256=inputs, capacity=capacity, scope="training_replay_only_no_new_simulation")
        save(output / "settings.json", settings)
        save(output / "status.json", dict(status="running"))
        names = contract["observation_schema"]["names"]
        data, episodes, alias_rows, identities = model["learner"]["replay"], [], [], []
        for round_index in (0, 1):
            if stopped(output):
                save(output / "status.json", dict(status="stopped"))
                return
            predecessor = None if round_index == 0 else inputs[str((BASE / "train_round0/model_final.pt").relative_to(REPO))]
            loaded_contract, runs = load_collections(
                [BASE / f"collect_round{round_index}" / s for s in SCENARIOS], round_index,
                contract["source_pins"], contract["runtime_versions"], predecessor)
            if loaded_contract != contract:
                raise ValueError("Collection and final model contracts differ")
            verify_collection_provenance(model, runs, round_index)
            for scenario in SCENARIOS:
                if stopped(output):
                    save(output / "status.json", dict(status="stopped"))
                    return
                run = runs[scenario]
                folder = Path(run["identity"]["folder"])
                trace = read(folder / "episode_00_trace.json")
                transitions = run["transitions"]
                check = validate_sequence(transitions, trace, names, capacity)
                for index, (transition, row) in enumerate(zip(transitions, trace)):
                    i = round_index*75 + index
                    for key, value in zip(("observations", "actions", "rewards", "next_observations", "terminated"), transition):
                        expected = torch.as_tensor(value, dtype=data[scenario][key].dtype)
                        torch.testing.assert_close(data[scenario][key][i], expected, rtol=0, atol=0)
                    aliases = q_aliases(learner, transition[0], row["action_anchor"], capacity)
                    alias_rows.extend(dict(scenario=scenario, round=round_index, control_step=index,
                                           **record) for record in aliases)
                episodes.append(dict(scenario=scenario, round=round_index, **check))
                identities.append(run["identity"])
        if stopped(output):
            save(output / "status.json", dict(status="stopped"))
            return
        spreads = np.array([r["twin_q_spread"] for r in alias_rows]).reshape(-1, 2)
        save(output / "aliases.json", alias_rows)
        result = dict(status="completed", settings=settings, episodes=episodes, collections=identities,
            alias_groups=len(alias_rows), aliased_states=len({(r["scenario"],r["round"],r["control_step"]) for r in alias_rows}),
            q_spread_mean=spreads.mean(axis=0).tolist() if len(spreads) else None,
            q_spread_max=spreads.max(axis=0).tolist() if len(spreads) else None,
            alias_groups_above_1e_6=int(np.any(spreads > 1e-6, axis=1).sum()),
            aliases_sha256=file_hash(output / "aliases.json"), elapsed_seconds=time.time()-started,
            caveat="Same projected request at the same saved training state; no counterfactual solver run. "
                   "Nominal actions enter the physical controller only through this projection. "
                   "A Q difference violates action equivalence, but does not measure optimal ranking or traffic gain.")
        verify_pins(DEFAULT_SNAPSHOT, contract["source_pins"])
        if file_hash(BASE / "train_round1/model_final.pt") != BASE_HASH or runtime_versions() != contract["runtime_versions"]:
            raise ValueError("Input/runtime changed")
        if any(file_hash(REPO / key) != value for key, value in source.items()):
            raise ValueError("Diagnostic source changed")
        verify_input_manifest(inputs)
        if stopped(output):
            save(output / "status.json", dict(status="stopped"))
            return
        save(output / "completion.json", result)
        save(output / "status.json", dict(status="completed"))
        print("PROJECTION_AUDIT_PASS", result["aliased_states"], result["q_spread_max"], flush=True)


if __name__ == "__main__":
    main()
