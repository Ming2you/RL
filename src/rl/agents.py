from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from src.models.state import ExperimentConfig


@dataclass(frozen=True)
class RLAgentSpec:
    """분산 RL actor의 소유권 경계를 명시하는 불변 사양."""

    agent_id: str
    family: str
    link: str | None = None
    segment_index: int | None = None
    signal: str | None = None
    ramps: Tuple[str, ...] = field(default_factory=tuple)
    off_ramps: Tuple[str, ...] = field(default_factory=tuple)
    movements: Tuple[str, ...] = field(default_factory=tuple)
    urban_links: Tuple[str, ...] = field(default_factory=tuple)
    boundary_links: Tuple[str, ...] = field(default_factory=tuple)
    connected_movements: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_freeway(self) -> bool:
        return self.family == "freeway_segment"

    @property
    def is_urban(self) -> bool:
        return self.family == "urban_intersection"


def _sorted_unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def build_rl_agent_specs(cfg: ExperimentConfig) -> Dict[str, RLAgentSpec]:
    """Spec 19의 actor 경계를 config topology에서 유도한다.

    E 노드는 plant 상태와 이웃 요약에는 남기지만, controlled signal actor로 만들지 않는다.
    """

    specs: Dict[str, RLAgentSpec] = {}
    specs.update(_build_freeway_segment_specs(cfg))
    specs.update(_build_urban_intersection_specs(cfg))
    return specs


def _build_freeway_segment_specs(cfg: ExperimentConfig) -> Dict[str, RLAgentSpec]:
    net = cfg.network
    specs: Dict[str, RLAgentSpec] = {}
    for link in net.freeway_links:
        for segment_index in range(net.freeway_segments_per_link):
            ramps = _sorted_unique(
                ramp
                for ramp in net.ramps
                if net.ramp_to_freeway.get(ramp) == link
                and int(net.ramp_merge_segment_index.get(ramp, -1)) == segment_index
            )
            off_ramps = _sorted_unique(
                off_ramp
                for off_ramp in net.off_ramps
                if net.off_ramp_from_freeway.get(off_ramp) == link
                and int(net.off_ramp_segment_index.get(off_ramp, -1)) == segment_index
            )
            connected_movements = _sorted_unique(
                movement
                for ramp in ramps
                for movement in net.on_ramp_to_movement.get(ramp, ())
            ) + _sorted_unique(
                movement
                for off_ramp in off_ramps
                for movement in net.off_ramp_to_movement.get(off_ramp, ())
            )
            agent_id = f"freeway_{link}_seg{segment_index}"
            specs[agent_id] = RLAgentSpec(
                agent_id=agent_id,
                family="freeway_segment",
                link=link,
                segment_index=segment_index,
                ramps=ramps,
                off_ramps=off_ramps,
                connected_movements=_sorted_unique(connected_movements),
                urban_links=_sorted_unique(
                    net.off_ramp_storage_link.get(off_ramp, "")
                    for off_ramp in off_ramps
                ),
            )
    return specs


def _build_urban_intersection_specs(cfg: ExperimentConfig) -> Dict[str, RLAgentSpec]:
    net = cfg.network
    uncontrolled = {str(node) for node in net.uncontrolled_nodes}
    storage_links = set(net.urban_link_storage_veh)
    boundary_links = set(net.boundary_in_links) | set(net.boundary_out_links)
    specs: Dict[str, RLAgentSpec] = {}

    for signal in net.signals:
        if signal in uncontrolled:
            continue
        movements = _sorted_unique(
            movement
            for movement, movement_spec in net.urban_movements.items()
            if str(movement_spec.get("intersection", "")) == signal
        )
        movement_specs = [net.urban_movements[movement] for movement in movements]
        urban_links = []
        local_boundary_links = []
        ramps = []
        off_ramps = []
        for movement_spec in movement_specs:
            for key in ("origin", "destination", "receiving_link"):
                value = str(movement_spec.get(key, ""))
                if value in storage_links:
                    urban_links.append(value)
                if value in boundary_links:
                    local_boundary_links.append(value)
            if movement_spec.get("ramp"):
                ramps.append(str(movement_spec["ramp"]))
            if movement_spec.get("off_ramp"):
                off_ramps.append(str(movement_spec["off_ramp"]))

        agent_id = f"urban_{signal}"
        specs[agent_id] = RLAgentSpec(
            agent_id=agent_id,
            family="urban_intersection",
            signal=signal,
            ramps=_sorted_unique(ramps),
            off_ramps=_sorted_unique(off_ramps),
            movements=movements,
            urban_links=_sorted_unique(urban_links),
            boundary_links=_sorted_unique(local_boundary_links),
        )
    return specs
