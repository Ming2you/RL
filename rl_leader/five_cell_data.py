"""Strict, balanced five-scenario replay for one shared response DDQN policy.

This opt-in wrapper preserves the existing single-scenario merge contract.
Every source episode is a complete normal-reset 75-interval evaluation. Inputs
and their original contracts are never rewritten to impersonate another cell.
"""
from __future__ import annotations

import copy
from dataclasses import fields
import hashlib
from typing import Mapping

import numpy as np

from rl_leader.experiment_contract import ExperimentContract, canonical_contract_json
from rl_leader.response_dqn_catalog import StructuredActionCatalog
from rl_leader.response_dqn_data import FrozenResponseReplay, merge_frozen_response_replays


FIVE_CELL_SCENARIOS = (
    "sweet_155_w60", "sweet_170_w60", "sweet_170_incident_w60",
    "sweet_170_skew15_w60", "sweet_190_w60",
)
FIVE_CELL_FORMAT = "five_cell_common_response_replay_v1"
FIVE_CELL_EQUIVALENCE = "post_commit_continuation_five_cell_v1"
CONTROL_STEPS = 75
_ARRAY_FIELDS = tuple(field.name for field in fields(FrozenResponseReplay) if field.name != "manifest")
_SHARED_FIELDS = (
    "format_version", "collection_phase", "training_phase_simulator_interaction",
    "catalog_fingerprint", "catalog", "action_count", "observation_schema",
    "response_contract", "response_equivalence_mode", "reward_semantics",
    "done_semantics", "t_total_sec", "response_evaluation_mode", "collector_format",
)


def _contract_map(contracts: Mapping[str, dict]) -> dict[str, ExperimentContract]:
    if set(contracts) != set(FIVE_CELL_SCENARIOS):
        raise ValueError("contracts must contain exactly the five required scenarios")
    verified = {}
    common = None
    required = {"scenario_name", "scenario", "resolved_config", "pstack_options",
                "warmup", "supervisor", "dynamic_far", "simulation"}
    for scenario in FIVE_CELL_SCENARIOS:
        payload = contracts[scenario]
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(f"incomplete experiment contract for {scenario}")
        contract = ExperimentContract.from_payload(payload)
        payload = contract.payload
        if payload["scenario_name"] != scenario or payload["scenario"].get("name") != scenario:
            raise ValueError(f"contract scenario identity mismatch for {scenario}")
        if (payload["simulation"].get("T_total_sec") != 14400.0
                or payload["simulation"].get("control_interval_sec") != 180.0
                or payload["warmup"].get("steps") != 5
                or payload["warmup"].get("control") != "uncontrolled"
                or payload["warmup"].get("included_in_total_ttt") is not True):
            raise ValueError("five-cell contract requires normal reset, warmup 5, and 14400 seconds")
        shared = {key: value for key, value in payload.items() if key not in {"scenario_name", "scenario"}}
        if common is not None and shared != common:
            raise ValueError("five-cell contracts differ in shared physical/configuration semantics")
        common = shared
        verified[scenario] = contract
    return verified


def _validate_common_replay(replay: FrozenResponseReplay) -> None:
    replay.validate()
    for name in ("episode", "control_step", "action_id", "option_steps"):
        if np.asarray(getattr(replay, name)).dtype.kind not in "iu":
            raise ValueError(f"five-cell replay requires integer {name}")
    manifest = replay.manifest
    if manifest.get("reward_semantics") != "interval_negative_ttt":
        raise ValueError("five-cell replay requires raw interval_negative_ttt rewards")
    if manifest.get("done_semantics") != "environment_terminal":
        raise ValueError("five-cell replay requires environment_terminal done flags")
    if manifest.get("response_equivalence_mode") != FIVE_CELL_EQUIVALENCE:
        raise ValueError(f"five-cell replay requires {FIVE_CELL_EQUIVALENCE} equivalence")
    if manifest.get("t_total_sec") != 14400.0 or not np.all(replay.option_steps == 1):
        raise ValueError("five-cell replay requires 14400-second one-interval transitions")
    if np.any(replay.reward >= 0):
        raise ValueError("five-cell interval rewards must be strictly negative")
    if any(manifest.get(key) is not None for key in ("mc_gamma", "reward_mode", "recovery_config")):
        raise ValueError("transformed/recovery labels cannot be five-cell interval data")
    catalog = StructuredActionCatalog.from_manifest(manifest.get("catalog", {}))
    if catalog.fingerprint != manifest["catalog_fingerprint"] or catalog.size != replay.action_count:
        raise ValueError("replay catalog content does not match its declared fingerprint/shape")
    schema = manifest.get("observation_schema", {})
    names = schema.get("names", [])
    if len(names) != replay.observation_dim or names.count("time.phase") != 1:
        raise ValueError("five-cell observation schema must identify exactly one time.phase")


def _episode_rows(replay: FrozenResponseReplay) -> list[tuple[int, np.ndarray]]:
    phase_index = replay.manifest["observation_schema"]["names"].index("time.phase")
    episodes = []
    for episode in np.unique(replay.episode):
        indices = np.flatnonzero(replay.episode == episode)
        indices = indices[np.argsort(replay.control_step[indices], kind="stable")]
        if len(indices) != CONTROL_STEPS or not np.array_equal(replay.control_step[indices], np.arange(CONTROL_STEPS)):
            raise ValueError(f"episode {episode} must contain each control step 0..74 exactly once")
        expected_done = np.zeros(CONTROL_STEPS, dtype=np.float32)
        expected_done[-1] = 1
        if not np.array_equal(replay.done[indices], expected_done):
            raise ValueError(f"episode {episode} must have exactly one true terminal at step 74")
        for next_name, current_name in (("next_observation", "observation"),
                                        ("next_response_features", "response_features"),
                                        ("next_action_mask", "action_mask")):
            if not np.array_equal(getattr(replay, next_name)[indices[:-1]],
                                  getattr(replay, current_name)[indices[1:]]):
                raise ValueError(f"episode {episode} has a sequential boundary mismatch in {next_name}")
        for name, offset in (("observation", 5), ("next_observation", 6)):
            expected = ((np.arange(CONTROL_STEPS) + offset) / 80.0).astype(np.float32)
            if not np.allclose(getattr(replay, name)[indices, phase_index], expected, rtol=0, atol=1e-7):
                raise ValueError(f"episode {episode} has incorrect normal-reset time.phase")
        episodes.append((int(episode), indices))
    return episodes


def _aggregate_fingerprint(manifest: dict) -> str:
    # Fingerprint the declared contract, not mutable provenance or array order.
    payload = {
        "format": FIVE_CELL_FORMAT,
        "multi_scenario_contracts": manifest["multi_scenario_contracts"],
        "shared_replay_contract": {key: manifest.get(key) for key in _SHARED_FIELDS},
    }
    return hashlib.sha256(canonical_contract_json(payload).encode("utf-8")).hexdigest()


def merge_five_cell_replays(
    replays: list[FrozenResponseReplay], contracts: dict[str, dict], source: str,
) -> FrozenResponseReplay:
    """Merge equal numbers of complete episodes; contracts maps name to payload.

    Payload means ``ExperimentContract.payload``, not an artifact-fields wrapper.
    A source replay may contain multiple complete episodes of one scenario.
    Original episode IDs may collide across source files and are renumbered with
    explicit provenance. Equal-length/equal-count episodes give every cell 20%
    of rows and terminals without synthetic copies or sampling weights.
    """
    verified = _contract_map(contracts)
    if not replays:
        raise ValueError("at least one replay for each scenario is required")
    grouped = {scenario: [] for scenario in FIVE_CELL_SCENARIOS}
    prepared = []
    for source_index, replay in enumerate(replays):
        _validate_common_replay(replay)
        scenario = replay.manifest.get("scenario")
        if scenario not in verified:
            raise ValueError(f"unexpected source scenario: {scenario}")
        if replay.manifest.get("experiment_contract_sha256") != verified[scenario].sha256:
            raise ValueError(f"source experiment contract hash mismatch for {scenario}")
        if (replay.manifest.get("experiment_contract") is not None
                and replay.manifest["experiment_contract"] != verified[scenario].payload):
            raise ValueError(f"source experiment contract payload mismatch for {scenario}")
        if replay.manifest.get("multi_scenario_contracts") is not None:
            raise ValueError("source replay must be single-scenario")
        episodes = _episode_rows(replay)
        grouped[scenario].append(replay)
        prepared.append((source_index, replay, scenario, episodes))
    if any(not values for values in grouped.values()):
        raise ValueError("replays must cover all five required scenarios")
    # Existing strict merge remains authoritative within each scenario.
    for scenario, values in grouped.items():
        merge_frozen_response_replays(values, source=f"{source}:{scenario}:validation")
    reference = replays[0]
    for replay in replays[1:]:
        for key in _SHARED_FIELDS:
            if replay.manifest.get(key) != reference.manifest.get(key):
                raise ValueError(f"five-cell shared replay contract mismatch: {key}")
        if (replay.observation_dim, replay.response_feature_dim) != (reference.observation_dim, reference.response_feature_dim):
            raise ValueError("five-cell observation/response dimensions differ")
    buffers = {name: [] for name in _ARRAY_FIELDS}
    episode_scenarios, episode_sources = {}, {}
    for scenario in FIVE_CELL_SCENARIOS:
        for source_index, replay, name, episodes in prepared:
            if name != scenario:
                continue
            for original_episode, indices in episodes:
                episode = len(episode_scenarios)
                episode_scenarios[str(episode)] = scenario
                episode_sources[str(episode)] = {
                    "input_index": source_index, "source": str(replay.manifest.get("source", "")),
                    "original_episode": original_episode,
                    "experiment_contract_sha256": verified[scenario].sha256,
                }
                for field in _ARRAY_FIELDS:
                    values = getattr(replay, field)[indices].copy()
                    if field == "episode":
                        values = np.full(CONTROL_STEPS, episode, dtype=np.int64)
                    elif field == "event_group":
                        values = np.asarray([f"five-cell:{episode}:{value}" for value in values])
                    buffers[field].append(values)
    manifest = copy.deepcopy(reference.manifest)
    for key in ("experiment_contract_sha256", "experiment_contract", "experiment_contract_version"):
        manifest.pop(key, None)
    manifest.update({
        "scenario": "five_cell_common_policy", "multi_scenario_format": FIVE_CELL_FORMAT,
        "source": str(source), "transition_count": len(episode_scenarios) * CONTROL_STEPS,
        "multi_scenario_contracts": {name: verified[name].artifact_fields() for name in FIVE_CELL_SCENARIOS},
        "episode_scenarios": episode_scenarios, "episode_sources": episode_sources,
        "merged_batch_count": len(replays),
    })
    manifest["aggregate_contract_sha256"] = _aggregate_fingerprint(manifest)
    merged = FrozenResponseReplay(**{name: np.concatenate(values) for name, values in buffers.items()}, manifest=manifest)
    validate_five_cell_replay(merged)
    return merged


def validate_five_cell_replay(replay: FrozenResponseReplay) -> dict:
    """Validate an aggregate, including contract identities and 20% cell balance."""
    _validate_common_replay(replay)
    manifest = replay.manifest
    if (manifest.get("multi_scenario_format") != FIVE_CELL_FORMAT
            or manifest.get("scenario") != "five_cell_common_policy"):
        raise ValueError("not a versioned five-cell common-policy replay")
    if "experiment_contract_sha256" in manifest or "experiment_contract" in manifest:
        raise ValueError("aggregate must not impersonate a single experiment contract")
    artifacts = manifest.get("multi_scenario_contracts", {})
    verified = _contract_map({name: artifact.get("experiment_contract") for name, artifact in artifacts.items()})
    for name, contract in verified.items():
        if artifacts[name].get("experiment_contract_sha256") != contract.sha256:
            raise ValueError(f"aggregate experiment contract hash mismatch for {name}")
    if manifest.get("aggregate_contract_sha256") != _aggregate_fingerprint(manifest):
        raise ValueError("aggregate contract fingerprint mismatch")
    episodes = _episode_rows(replay)
    scenario_map, sources = manifest.get("episode_scenarios", {}), manifest.get("episode_sources", {})
    keys = {str(episode) for episode, _ in episodes}
    if set(scenario_map) != keys or set(sources) != keys:
        raise ValueError("episode scenario/provenance maps must cover exactly the replay episodes")
    counts = {name: 0 for name in FIVE_CELL_SCENARIOS}
    for episode, _ in episodes:
        scenario = scenario_map[str(episode)]
        if scenario not in verified:
            raise ValueError("episode refers to an unknown scenario")
        if sources[str(episode)].get("experiment_contract_sha256") != verified[scenario].sha256:
            raise ValueError("episode provenance contract differs from its scenario")
        counts[scenario] += 1
    if min(counts.values()) < 1 or len(set(counts.values())) != 1:
        raise ValueError("five-cell replay requires equal complete episode counts in all scenarios")
    return {
        "format": FIVE_CELL_FORMAT, "total_rows": replay.size, "total_episodes": len(episodes),
        "true_terminals": int(replay.done.sum()), "verified_links": replay.size - len(episodes),
        "aggregate_contract_sha256": manifest["aggregate_contract_sha256"],
        "scenario_counts": {name: {"episodes": count, "rows": count * CONTROL_STEPS,
                                  "terminals": count, "row_fraction": .2, "terminal_fraction": .2}
                            for name, count in counts.items()},
        "sampling": "equal complete episodes; uniform rows and terminal strata each allocate 20% per scenario",
    }
