"""Repackage a frozen response replay with a smaller action catalog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rl_leader.response_dqn_catalog import (
    DiscreteLeaderAction,
    StructuredActionCatalog,
)
from rl_leader.response_dqn_data import FrozenResponseReplay, load_frozen_response_replay


SUBSET_REPLAY_FORMAT = "response_aware_catalog_subset_replay_v1"


def _parse_csv(value: str, cast=str) -> tuple:
    return tuple(cast(item) for item in str(value).split(",") if item)


def _keep_action(
    action: DiscreteLeaderAction,
    *,
    families: set[str] | None,
    domains: set[str] | None,
    owners: set[str] | None,
    templates: set[str] | None,
    magnitudes: set[float] | None,
) -> bool:
    if action.action_id == 0:
        return True
    if families is not None and action.family not in families:
        return False
    if domains is not None and action.domain not in domains:
        return False
    if owners is not None and action.owner not in owners:
        return False
    if templates is not None and action.template not in templates:
        return False
    if magnitudes is not None and float(action.magnitude) not in magnitudes:
        return False
    return True


def subset_replay_catalog(
    replay: FrozenResponseReplay,
    *,
    source: str,
    families: set[str] | None = None,
    domains: set[str] | None = None,
    owners: set[str] | None = None,
    templates: set[str] | None = None,
    magnitudes: set[float] | None = None,
) -> FrozenResponseReplay:
    """Keep selected catalog actions and remap action IDs contiguously."""
    replay.validate()
    if "catalog" not in replay.manifest:
        raise ValueError("replay manifest does not contain an action catalog")
    old_catalog = StructuredActionCatalog.from_manifest(replay.manifest["catalog"])
    selected_old_ids = [
        action.action_id
        for action in old_catalog.actions
        if _keep_action(
            action,
            families=families,
            domains=domains,
            owners=owners,
            templates=templates,
            magnitudes=magnitudes,
        )
    ]
    if selected_old_ids[:1] != [0]:
        raise ValueError("subset must preserve action 0 as the anchor")
    if len(selected_old_ids) <= 1:
        raise ValueError("catalog subset removed every non-anchor action")

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected_old_ids)}
    selected_set = set(selected_old_ids)
    row_mask = np.asarray([
        int(action_id) in selected_set
        for action_id in replay.action_id
    ], dtype=bool)
    if not np.any(row_mask):
        raise ValueError("catalog subset removed every replay row")

    idx = np.flatnonzero(row_mask)
    new_actions = [
        DiscreteLeaderAction(
            action_id=new_id,
            key=old_catalog.action(old_id).key,
            domain=old_catalog.action(old_id).domain,
            owner=old_catalog.action(old_id).owner,
            template=old_catalog.action(old_id).template,
            magnitude=old_catalog.action(old_id).magnitude,
            family=old_catalog.action(old_id).family,
            residual=old_catalog.action(old_id).residual,
        )
        for new_id, old_id in enumerate(selected_old_ids)
    ]
    new_catalog = StructuredActionCatalog(old_catalog.action_names, new_actions)
    selected_ids = np.asarray(selected_old_ids, dtype=np.int64)
    action_id = np.asarray([
        old_to_new[int(old_id)]
        for old_id in replay.action_id[idx]
    ], dtype=np.int64)

    manifest = dict(replay.manifest)
    manifest.update({
        "subset_format_version": SUBSET_REPLAY_FORMAT,
        "source": str(source),
        "source_dataset": str(replay.manifest.get("source", "")),
        "source_catalog_fingerprint": str(replay.manifest.get("catalog_fingerprint", "")),
        "transition_count": int(idx.size),
        "action_count": int(new_catalog.size),
        "catalog_fingerprint": new_catalog.fingerprint,
        "catalog": new_catalog.as_manifest(),
        "catalog_subset": {
            "old_action_ids": [int(value) for value in selected_old_ids],
            "families": sorted(families) if families is not None else None,
            "domains": sorted(domains) if domains is not None else None,
            "owners": sorted(owners) if owners is not None else None,
            "templates": sorted(templates) if templates is not None else None,
            "magnitudes": sorted(magnitudes) if magnitudes is not None else None,
            "kept_rows": int(idx.size),
            "dropped_rows": int(replay.size - idx.size),
        },
    })
    return FrozenResponseReplay(
        observation=replay.observation[idx],
        action_id=action_id,
        reward=replay.reward[idx],
        next_observation=replay.next_observation[idx],
        done=replay.done[idx],
        option_steps=replay.option_steps[idx],
        action_mask=replay.action_mask[idx][:, selected_ids],
        next_action_mask=replay.next_action_mask[idx][:, selected_ids],
        response_features=replay.response_features[idx][:, selected_ids, :],
        next_response_features=replay.next_response_features[idx][:, selected_ids, :],
        event_group=replay.event_group[idx],
        episode=replay.episode[idx],
        control_step=replay.control_step[idx],
        manifest=manifest,
    ).validate()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default="")
    parser.add_argument("--domains", default="")
    parser.add_argument("--owners", default="")
    parser.add_argument("--templates", default="")
    parser.add_argument("--magnitudes", default="")
    args = parser.parse_args(argv)

    replay = load_frozen_response_replay(args.input)
    subset = subset_replay_catalog(
        replay,
        source="rl_leader.subset_response_replay_catalog",
        families=set(_parse_csv(args.families)) or None,
        domains=set(_parse_csv(args.domains)) or None,
        owners=set(_parse_csv(args.owners)) or None,
        templates=set(_parse_csv(args.templates)) or None,
        magnitudes=set(_parse_csv(args.magnitudes, float)) or None,
    )
    subset.save(args.out)
    print(json.dumps({
        "input": str(args.input),
        "output": str(args.out),
        "transitions": subset.size,
        "action_count": subset.action_count,
        "action_support_counts": subset.action_support_counts().tolist(),
        "catalog_fingerprint": subset.manifest["catalog_fingerprint"],
        "old_action_ids": subset.manifest["catalog_subset"]["old_action_ids"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
