"""원 결합 plant의 적분 시점에서 player별 순수 TTT를 분할하는 선택적 회계.

제어기가 없는 stock도 PASSIVE 비용으로 포함한다. Dynamics·기존 비용·상태는
수정하지 않으며 observer 없이 실행하는 기존 controller의 수치 경로는 그대로다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from src.models.state import ExperimentConfig, TrafficState


def freeway_buffer_vehicle_counts(state: TrafficState, cfg: ExperimentConfig) -> dict[tuple[str, str, int], float]:
    """과거 METANET의 실제 상·하류 완충 셀 점유[veh]; 도시 도착 예약은 제외."""
    net = cfg.network
    # d6241bb METANET은 완충을 nominal lanes와 동일 segment 길이로 적분한다.
    # 코어 effective-lane 배열/total_freeway_vehicles에는 이 별도 셀이 포함되지 않는다.
    if int(getattr(net, "freeway_buffer_segments", 0)) <= 0:
        return {}
    return {(link, side, i): float(rho) * net.freeway_segment_length_km * net.freeway_lanes
            for side in ("up", "down")
            for link in net.freeway_links
            for i, rho in enumerate(getattr(state, f"freeway_buffer_{side}_density", {}).get(link, []))}


def freeway_buffer_state_values(state: TrafficState) -> tuple[list[float], list[float]]:
    """기존 물리 gate가 완충의 밀도·속도도 검사하도록 원 상태값만 읽는다."""
    return tuple([value for side in ("up", "down")
                  for values in getattr(state, f"freeway_buffer_{side}_{kind}", {}).values()
                  for value in values] for kind in ("density", "speed"))


@dataclass
class PlayerCostOwnership:
    active_player_ids: tuple[str, ...]
    segment_owner: dict[tuple[str, int], str]
    origin_owner: dict[str, str]
    ramp_owner: dict[str, str]
    offramp_storage_owner: dict[str, str]
    movement_owner: dict[str, str]
    urban_storage_owner: dict[str, str]
    passive_cost_owners: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: ExperimentConfig, *, freeway_segments: bool = False,
                    passive_cost_owners: Mapping[str, str] | None = None) -> "PlayerCostOwnership":
        """실제 기본 player는 freeway link와 controlled signal; segment 옵션도 명시 지원."""
        net = cfg.network
        segments = {
            (link, i): f"FW:{link}:seg{i}" if freeway_segments else f"FW:{link}"
            for link in net.freeway_links for i in range(net.freeway_segments_per_link)
        }

        def freeway_owner(link: str, index: int) -> str:
            index = min(max(index, 0), net.freeway_segments_per_link - 1)
            return segments.get((link, index), "PASSIVE:UNMAPPED_F")

        def urban_owner(node: str) -> str:
            return f"URBAN:{node}" if node in net.signals else f"PASSIVE:{node or 'UNMAPPED_U'}"

        origins = {link: freeway_owner(link, 0) for link in net.freeway_links}
        ramps = {r: freeway_owner(net.ramp_to_freeway.get(r, ""),
                                 int(net.ramp_merge_segment_index.get(r, net.freeway_segments_per_link // 2)))
                 for r in net.ramps}
        offramps: dict[str, str] = {}
        for ramp, storage in net.off_ramp_storage_link.items():
            owner = freeway_owner(net.off_ramp_from_freeway.get(ramp, ""),
                                  int(net.off_ramp_segment_index.get(ramp, net.freeway_segments_per_link - 1)))
            if storage in offramps and offramps[storage] != owner:
                raise ValueError(f"Ambiguous off-ramp storage owner: {storage}")
            offramps[storage] = owner
        movements = {movement: urban_owner(str(spec.get("intersection", "")))
                     for movement, spec in net.urban_movements.items()}
        storage_owners: dict[str, str] = {}
        for storage in net.urban_link_storage_veh:
            if storage in offramps:
                continue
            # Directed transit는 downstream intersection, sink는 final upstream intersection.
            # 이름을 추측하지 않고 movement의 origin/receiving 관계만 사용한다.
            downstream = {str(spec.get("intersection", "")) for spec in net.urban_movements.values()
                          if str(spec.get("origin", "")) == storage}
            upstream = {str(spec.get("intersection", "")) for spec in net.urban_movements.values()
                        if str(spec.get("receiving_link", "")) == storage}
            nodes = downstream or upstream
            if len(nodes) > 1:
                raise ValueError(f"Ambiguous urban storage owner: {storage}: {sorted(nodes)}")
            storage_owners[storage] = urban_owner(next(iter(nodes), "UNMAPPED_U"))
        active = tuple(dict.fromkeys(segments.values())) + tuple(f"URBAN:{s}" for s in net.signals)
        ownership = cls(active, segments, origins, ramps, offramps, movements, storage_owners)
        # PASSIVE:E는 제어 권한 없이 raw ledger에 유지한다. Player 비용합=총TTT를
        # 위해 북쪽 인접 controlled B에 비용만 귀속하는 명시적 확장이다.
        fallback = next((f"URBAN:{s}" for s in net.signals), active[0])
        supplied = dict(passive_cost_owners or {})
        for bucket in ownership.passive_cost_ids:
            owner = supplied.get(bucket, "URBAN:B" if bucket == "PASSIVE:E" and "B" in net.signals else fallback)
            if owner not in active:
                raise ValueError(f"Passive cost owner is not an active player: {bucket}: {owner}")
            ownership.passive_cost_owners[bucket] = owner
        return ownership

    @property
    def passive_cost_ids(self) -> tuple[str, ...]:
        """최적화 player를 추가하지 않고 남겨 둔 비통제 비용 bucket 목록."""
        mappings = (self.segment_owner, self.origin_owner, self.ramp_owner,
                    self.offramp_storage_owner, self.movement_owner, self.urban_storage_owner)
        return tuple(sorted({owner for mapping in mappings for owner in mapping.values()
                             if owner not in self.active_player_ids}))


@dataclass
class PlayerCostLedger:
    cfg: ExperimentConfig
    ownership: PlayerCostOwnership | None = None
    raw_costs: dict[str, float] = field(default_factory=dict, init=False)
    regional_costs: dict[str, float] = field(default_factory=lambda: {"F": 0.0, "U": 0.0}, init=False)
    substeps: dict[str, int] = field(default_factory=lambda: {"urban": 0, "freeway": 0}, init=False)
    max_abs_substep_residual: float = field(default=0.0, init=False)
    terms: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.ownership is None:
            self.ownership = PlayerCostOwnership.from_config(self.cfg)
        self.raw_costs = {key: 0.0 for key in self.ownership.active_player_ids + self.ownership.passive_cost_ids}

    def inventory(self, stage: str, state: TrafficState) -> dict[str, tuple[str, str, float]]:
        """해당 적분 시점의 (물리 stock key -> owner, F/U region, veh)만 반환한다."""
        net, own = self.cfg.network, self.ownership
        assert own is not None
        result: dict[str, tuple[str, str, float]] = {}
        if stage == "freeway":
            # Effective lanes를 사용해야 capacity-drop 상태에서도 차량이 보존된다.
            for link, counts in state.freeway_vehicle_count_by_link(net).items():
                for i, count in enumerate(counts):
                    result[f"segment:{link}:{i}"] = (own.segment_owner.get((link, i), "PASSIVE:UNMAPPED_F"), "F", count)
            # 완충은 무제어지만 물리 TTT가 존재하므로 같은 freeway player에 한 번 귀속한다.
            for (link, side, i), count in freeway_buffer_vehicle_counts(state, self.cfg).items():
                result[f"buffer:{link}:{side}:{i}"] = (own.origin_owner[link], "F", count)
            for link, queue in state.mainline_origin_queue.items():
                result[f"origin:{link}"] = (own.origin_owner.get(link, "PASSIVE:UNMAPPED_F"), "F", max(0.0, queue))
            for ramp, queue in state.ramp_queue.items():
                result[f"reservoir:{ramp}"] = (own.ramp_owner.get(ramp, "PASSIVE:UNMAPPED_F"), "F", queue)
        elif stage == "urban":
            for movement, queue in state.urban_movement_queue.items():
                result[f"movement:{movement}"] = (own.movement_owner.get(movement, "PASSIVE:UNMAPPED_U"), "U", queue)
            for link, capacity in net.urban_link_storage_veh.items():
                occupancy = max(0.0, capacity - state.urban_link_storage.get(link, capacity))
                if link in own.offramp_storage_owner:
                    result[f"storage:{link}"] = (own.offramp_storage_owner[link], "F", occupancy)
                else:
                    result[f"storage:{link}"] = (own.urban_storage_owner.get(link, "PASSIVE:UNMAPPED_U"), "U", occupancy)
        else:
            raise ValueError(f"Unknown integration stage: {stage}")
        return result

    def observe(self, stage: str, state: TrafficState, dt_h: float,
                freeway_increment: float, urban_increment: float) -> None:
        """coupling observer API. 전달된 state는 읽기만 하고 복사/수정하지 않는다."""
        inventory = self.inventory(stage, state)
        subtotals = {"F": 0.0, "U": 0.0}
        for stock, (owner, region, vehicles) in inventory.items():
            cost = vehicles * dt_h
            self.raw_costs[owner] = self.raw_costs.get(owner, 0.0) + cost
            self.terms[stock] = self.terms.get(stock, 0.0) + cost
            self.regional_costs[region] += cost
            subtotals[region] += cost
        residual = max(abs(subtotals["F"] - freeway_increment), abs(subtotals["U"] - urban_increment))
        self.max_abs_substep_residual = max(self.max_abs_substep_residual, residual)
        scale = max(1.0, abs(freeway_increment) + abs(urban_increment))
        if residual > 1.0e-10 * scale:
            # 누락/중복을 arbitrary player 보정으로 숨기지 않고 즉시 실패시킨다.
            raise ValueError(f"Player TTT accounting mismatch at {stage}: {residual}")
        self.substeps[stage] += 1

    @property
    def total_ttt(self) -> float:
        return sum(self.raw_costs.values())

    @property
    def player_costs(self) -> dict[str, float]:
        """기존 active player에 passive 비용을 정확히 한 번 포함한 local cost."""
        assert self.ownership is not None
        result = {key: 0.0 for key in self.ownership.active_player_ids}
        for bucket, cost in self.raw_costs.items():
            owner = bucket if bucket in result else self.ownership.passive_cost_owners.get(bucket)
            if owner not in result:
                raise ValueError(f"Unassigned passive cost bucket: {bucket}")
            result[owner] += cost
        return result

    @property
    def costs(self) -> dict[str, float]:
        """Local controller에 전달하는 비용은 passive 귀속을 포함한 player 비용이다."""
        return self.player_costs

    def cost_vector(self, ids: tuple[str, ...] | None = None) -> list[float]:
        """기존 active player 순서의 비용 벡터; raw passive 비용도 한 번 포함된다."""
        costs = self.player_costs
        return [costs[key] for key in (ids or tuple(costs))]
