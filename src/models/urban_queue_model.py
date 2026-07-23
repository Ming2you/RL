from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Mapping, Tuple

import numpy as np

from .demand import DemandStep
from .state import ControlAction, ExperimentConfig, TrafficState


# inflow_outflow_allocation은 perimeter(경계/램프) 제어 전용 신호다. 내부 urban
# movement는 green×saturation으로만 제어돼야 하므로 allocation을 cap으로 적용하지 않는다.
PERIMETER_MOVEMENT_KINDS = {"boundary_in", "off_ramp", "boundary_out", "on_ramp"}


def safe_balance_index(values: Iterable[float], eps: float = 1.0e-9) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or float(np.sum(np.abs(arr))) <= eps:
        return 0.0
    l1 = float(np.sum(np.abs(arr)))
    l2_sq = float(np.sum(arr * arr))
    return max(0.0, l2_sq / max(l1 * l1, eps) - 1.0 / arr.size)


def boundary_indices(values: Iterable[float], queue_max: float, eps: float = 1.0e-9) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"CV_boundary": 0.0, "MaxMin_boundary": 0.0, "OverflowRatio_boundary": 0.0}
    mean = float(np.mean(arr))
    return {
        "CV_boundary": float(np.std(arr) / max(mean, eps)) if mean > eps else 0.0,
        "MaxMin_boundary": float(np.max(arr) - np.min(arr)),
        "OverflowRatio_boundary": float(np.mean(arr > queue_max)),
    }


def movement_specs(cfg: ExperimentConfig) -> Dict[str, Dict[str, object]]:
    return {key: dict(value) for key, value in cfg.network.urban_movements.items()}


# _sync_legacy_queues/ensure_urban_state 전용 정적 캐시 — link↔movement 토폴로지
# (origin/destination)와 키 집합 크기는 런 중 불변 전제(위반 시 캐시 무효화 필요).
# network 객체 참조 동일성으로 재사용을 판정한다. 합산 순서는 urban_movements 삽입
# 순서 그대로 보존해 기존 구현과 부동소수 비트동일을 유지한다.
_SYNC_INDEX_NET = None
_SYNC_INDEX: tuple = ()
_SYNC_COUNTS: tuple = (0, 0, 0)


def _legacy_sync_index(cfg: ExperimentConfig) -> tuple:
    global _SYNC_INDEX_NET, _SYNC_INDEX, _SYNC_COUNTS
    net = cfg.network
    if net is not _SYNC_INDEX_NET:
        index = []
        for link in net.movement_links:
            moves = tuple(
                movement
                for movement, spec in net.urban_movements.items()
                if spec.get("origin") == link or spec.get("destination") == link
            )
            index.append((link, moves))
        _SYNC_INDEX = tuple(index)
        _SYNC_COUNTS = (
            len(net.urban_movements),
            len(approach_routing(cfg)),
            len(net.urban_link_storage_veh),
        )
        _SYNC_INDEX_NET = net
    return _SYNC_INDEX


def movement_forecast_arrivals_veh(
    cfg: ExperimentConfig,
    forecast: Iterable[DemandStep],
) -> Dict[str, float]:
    """forecast 구간 동안 movement별 외생 도착량[veh] — boundary_in(게이트)·on_ramp(램프)만.

    off_ramp/boundary_out는 외생 도착이 없고(freeway·내부 흐름에서 유입) 큐로만 잡히므로 0.
    leader reachability 경계와 coordinator follower가 동일한 도착을 보도록 단일 출처로 둔다."""
    net = cfg.network
    dt_h = float(cfg.simulation.T_c_h)
    specs = movement_specs(cfg)
    onramp_by_movement = {
        movement: ramp
        for ramp, movements in net.on_ramp_to_movement.items()
        for movement in movements
    }
    arrivals: Dict[str, float] = {}
    for step in forecast:
        for movement, spec in specs.items():
            kind = str(spec.get("kind", ""))
            if kind == "boundary_in":
                origin = str(spec.get("origin", ""))
                beta = float(spec.get("beta", 1.0))
                arrivals[movement] = arrivals.get(movement, 0.0) + (
                    max(0.0, step.urban_boundary.get(origin, 0.0)) * beta * dt_h
                )
            elif kind == "on_ramp":
                ramp = onramp_by_movement.get(movement, "")
                if not ramp:
                    continue
                share = 1.0 / max(len(net.on_ramp_to_movement.get(ramp, [])), 1)
                arrivals[movement] = arrivals.get(movement, 0.0) + (
                    max(0.0, step.ramp_arrival.get(ramp, 0.0)) * share * dt_h
                )
    return arrivals


def approach_routing(cfg: ExperimentConfig) -> Dict[str, list[tuple[str, float]]]:
    """approach source(링크/게이트/off-ramp storage) → [(movement, β)] 매핑.

    spec §3.3.5: 링크 끝(큐 꼬리)에 도착한 차량은 단일 next_movement가 아니라
    β[o,s,·]로 그 교차로의 (o,s,d) movement 큐들에 분할 주입된다."""
    net = cfg.network
    routing: Dict[str, list[tuple[str, float]]] = {}
    for movement, spec in cfg.network.urban_movements.items():
        kind = str(spec.get("kind", ""))
        if kind == "off_ramp":
            source = net.off_ramp_storage_link.get(str(spec.get("origin", "")), "")
        else:
            # boundary_in은 게이트 in링크, 그 외는 내부 incoming link가 origin이다.
            source = str(spec.get("origin", ""))
        if source:
            routing.setdefault(source, []).append((movement, float(spec.get("beta", 0.0))))
    return routing


def sink_storage_links(cfg: ExperimentConfig) -> set[str]:
    """boundary_out 게이트로 나가는 storage 링크 — 링크 끝 도착 = system sink 이탈."""
    out_links = set(cfg.network.boundary_out_links)
    return {
        str(spec.get("receiving_link", ""))
        for spec in cfg.network.urban_movements.values()
        if str(spec.get("destination", "")) in out_links
    }


def movement_storage_capacity(
    cfg: ExperimentConfig,
    movement: str,
    spec: Mapping[str, object] | None = None,
) -> float:
    """movement queue를 density로 정규화할 때 쓰는 저장용량을 반환한다."""
    spec = movement_specs(cfg).get(movement, {}) if spec is None else spec
    if "storage_capacity_veh" in spec:
        return float(spec["storage_capacity_veh"])
    kind = spec.get("kind")
    if kind == "off_ramp":
        off_ramp = str(spec.get("off_ramp", ""))
        storage = cfg.network.off_ramp_storage_link.get(off_ramp, "")
        return float(cfg.network.urban_link_storage_veh.get(storage, cfg.network.boundary_queue_max_veh))
    if kind == "on_ramp":
        return float(cfg.network.boundary_queue_max_veh)
    receiving = str(spec.get("receiving_link", ""))
    return float(cfg.network.urban_link_storage_veh.get(receiving, cfg.network.boundary_queue_max_veh))


# spec §3.3 (397행): "receiving space belongs to a directed link." movement 점큐 x[o,s,d]는
# 그 movement가 올라온 origin 링크 l_{o,s} 위 점유로 존재하므로, 그 링크의 가용공간 S를
# 줄여야 한다. cfg 객체별로 storage 링크 → 그 링크를 origin으로 하는 movement 목록을 캐시한다.
# 값에 cfg 참조를 함께 보관해 id 재사용(GC 후 동일 id의 새 cfg) 시 stale 캐시를
# 반환하지 않도록 한다(`cached_cfg is cfg` 검증). 직전 구현은 id(cfg)만 키로 써,
# 한 cfg가 GC된 뒤 다른 network의 cfg가 같은 id를 받으면 stale movement 목록을
# 반환했다(테스트 순서 의존 버그). WU-CD-F probe가 _effective_available_space를
# 호출하면서 이 캐시를 더 빈번히 채워 그 잠복 버그가 표면화됨.
_ORIGIN_STORAGE_MOVEMENTS_CACHE: Dict[int, tuple[ExperimentConfig, Dict[str, list[str]]]] = {}


def _origin_storage_movements(cfg: ExperimentConfig) -> Dict[str, list[str]]:
    """storage 링크 → 그 링크를 origin으로 점유하는 movement 목록 매핑.

    internal·boundary_out·on_ramp movement는 origin이 실제 storage 링크 l_{o,s}이고
    그 점큐가 urban_movement_queue에 쌓인다. boundary_in(origin=게이트 in링크)·
    off_ramp(origin=ramp 이름, 점큐는 storage occupancy로 직접 잡힘)는 origin이 storage
    링크가 아니라 자연히 제외된다."""
    key = id(cfg)
    cached = _ORIGIN_STORAGE_MOVEMENTS_CACHE.get(key)
    if cached is not None and cached[0] is cfg:
        return cached[1]
    storage_links = set(cfg.network.urban_link_storage_veh.keys())
    mapping: Dict[str, list[str]] = {}
    for movement, spec in cfg.network.urban_movements.items():
        origin = str(spec.get("origin", ""))
        if origin in storage_links:
            mapping.setdefault(origin, []).append(movement)
    _ORIGIN_STORAGE_MOVEMENTS_CACHE[key] = (cfg, mapping)
    return mapping


def _effective_available_space(
    state: TrafficState,
    cfg: ExperimentConfig,
    link: str,
) -> float:
    """링크 L의 유효 가용공간 S_eff(L).

    S_eff(L) = urban_link_storage[L] − Σ(urban_movement_queue[m] : origin(m)==L), 0 이상 clamp.
    그 링크 끝(교차로)에서 대기 중인 movement 점큐를 그 링크의 점유로 잡아, 하류가 막혀
    점큐가 쌓이면 그 링크 S_eff가 줄어 상류 movement가 receiving 제약에 걸리고 backup이
    전파된다(spec §3.3.2, 397행). 게이팅 계산에만 쓰고 storage 값 자체는 바꾸지 않으므로
    차량보존 불변이다."""
    available = max(0.0, state.urban_link_storage.get(link, 0.0))
    point_queue = 0.0
    for movement in _origin_storage_movements(cfg).get(link, ()):  # noqa: B007
        point_queue += max(0.0, state.urban_movement_queue.get(movement, 0.0))
    return max(0.0, available - point_queue)


def movement_density_values(
    state: TrafficState,
    cfg: ExperimentConfig,
    accepted_kinds: set[str],
) -> list[float]:
    """§3.2 allocation objective와 동일한 movement-level density vector를 만든다."""
    specs = movement_specs(cfg)
    out: list[float] = []
    for movement, spec in specs.items():
        if str(spec.get("kind", "")) not in accepted_kinds:
            continue
        queue = max(0.0, state.urban_movement_queue.get(movement, 0.0))
        out.append(queue / max(movement_storage_capacity(cfg, movement, spec), 1.0e-9))
    return out


def boundary_group_key(spec: Mapping[str, object]) -> str:
    """movement가 속한 물리적 경계 요소 key — 게이트 in링크 / off-ramp / ramp.

    balance 지표는 경계 요소 간 공평성이 목적이다. 그리드 라우팅으로 한 게이트가
    β분할 movement 3~4개로 쪼개진 뒤에는 movement 단위 B가 "구조적 공큐 조각"에
    지배되므로(B_in 바닥 ≈0.1), round-9 당시의 차원(경계 요소당 1값)으로 집계한다."""
    kind = str(spec.get("kind", ""))
    if kind == "boundary_in":
        return f"gate:{spec.get('origin', '')}"
    if kind == "off_ramp":
        return f"off:{spec.get('off_ramp', spec.get('origin', ''))}"
    if kind == "on_ramp":
        return f"ramp:{spec.get('ramp', '')}"
    return f"movement:{spec.get('origin', '')}->{spec.get('destination', '')}"


def grouped_boundary_densities(
    state: TrafficState,
    cfg: ExperimentConfig,
    accepted_kinds: set[str],
) -> list[float]:
    """경계 요소(게이트/off-ramp/ramp)별 집계 밀도 Σqueue/Σcapacity 목록."""
    specs = movement_specs(cfg)
    queues: Dict[str, float] = {}
    caps: Dict[str, float] = {}
    for movement, spec in specs.items():
        if str(spec.get("kind", "")) not in accepted_kinds:
            continue
        key = boundary_group_key(spec)
        queues[key] = queues.get(key, 0.0) + max(0.0, state.urban_movement_queue.get(movement, 0.0))
        caps[key] = caps.get(key, 0.0) + max(movement_storage_capacity(cfg, movement, spec), 1.0e-9)
    return [queues[key] / caps[key] for key in sorted(queues)]


def movement_balance_summary(
    state: TrafficState,
    cfg: ExperimentConfig,
    saturation_fraction: float = 0.95,
    degenerate_ratio: float = 0.5,
    eps: float = 1.0e-9,
) -> Dict[str, float]:
    """Movement-level B와 degenerate 여부를 한 번에 계산한다.

    B 자체는 §3.2의 균등성 지표이므로, 모든 큐가 비었거나 대부분 포화된 경우에는
    값이 작아도 제어 가능한 균형으로 해석하지 않도록 별도 flag를 함께 낸다.

    B_out은 통제 가능한 유출(on_ramp)만 대상으로 한다 — boundary_out은 자유 유출
    sink라 항상 최대 서비스가 정답이고 균등화가 정의되지 않는다(round-9
    "outflow 균등화 ill-posed" 결론, 그리드 라우팅 후 출구 교살로 실증).

    밀도는 경계 요소(게이트 7 / ramp 4) 단위로 집계한다 — round-9의 지표 차원을
    보존하고, β분할 movement 조각의 구조적 공큐 인공물을 제거한다.

    B_in은 외부 진입 게이트(boundary_in)만 대상으로 한다 — off-ramp 방출 큐는
    freeway 보호를 위해 의도적으로 우선 서비스되는 transfer 큐라 상시 ≈0이고
    (균형화가 아니라 우선 방출이 설계 목표), 게이트와 한 지표에 섞으면 구조적
    0들이 B_in을 지배해 게이트 간 공평성 측정이 무의미해진다.
    """
    inflow = grouped_boundary_densities(state, cfg, {"boundary_in"})
    outflow = grouped_boundary_densities(state, cfg, {"on_ramp"})
    inflow_load = float(sum(
        max(0.0, state.urban_movement_queue.get(movement, 0.0))
        for movement, spec in movement_specs(cfg).items()
        if str(spec.get("kind", "")) == "boundary_in"
    ))
    outflow_load = float(sum(
        max(0.0, state.urban_movement_queue.get(movement, 0.0))
        for movement, spec in movement_specs(cfg).items()
        if str(spec.get("kind", "")) == "on_ramp"
    ))

    def ratio(values: list[float], predicate: Callable[[float], bool]) -> float:
        if not values:
            return 1.0
        return float(sum(1 for value in values if predicate(value)) / len(values))

    in_empty = ratio(inflow, lambda value: value <= eps)
    out_empty = ratio(outflow, lambda value: value <= eps)
    in_saturated = ratio(inflow, lambda value: value >= saturation_fraction)
    out_saturated = ratio(outflow, lambda value: value >= saturation_fraction)
    empty_ratio = max(in_empty, out_empty)
    saturation_ratio = max(in_saturated, out_saturated)
    # degenerate 판정은 공큐 지배만 사용한다. saturation 기준은 큐가 cap에 클립되던
    # 시절의 가드였고, 클립 제거(점큐 무한 성장) 후에는 "큐가 크다=구조적 적체"일 뿐
    # 균형 제어가 불능인 상태가 아니다(saturation 비율은 descriptive로 유지).
    degenerate = empty_ratio >= degenerate_ratio
    return {
        "B_in": safe_balance_index(inflow),
        "B_out": safe_balance_index(outflow),
        # run-level 집계에서 부하 가중에 쓰는 load(B는 스케일 불변이라 노이즈 수준
        # 잔여 큐의 B와 실제 대기 큐의 B를 같은 무게로 평균하면 안 된다).
        "boundary_in_load_veh": inflow_load,
        "boundary_out_load_veh": outflow_load,
        "boundary_empty_ratio": float(empty_ratio),
        "boundary_saturation_ratio": float(saturation_ratio),
        "boundary_in_empty_ratio": float(in_empty),
        "boundary_out_empty_ratio": float(out_empty),
        "boundary_in_saturation_ratio": float(in_saturated),
        "boundary_out_saturation_ratio": float(out_saturated),
        "boundary_balance_degenerate": float(degenerate),
        "boundary_balance_controllable": 0.0 if degenerate else 1.0,
    }


def ensure_urban_state(state: TrafficState, cfg: ExperimentConfig) -> None:
    net = cfg.network
    _legacy_sync_index(cfg)
    n_moves, n_sources, n_links = _SYNC_COUNTS
    if (
        len(state.urban_movement_queue) >= n_moves
        and len(state.urban_arrival_buffer) >= n_sources
        and len(state.urban_link_storage) >= n_links
        and len(state.urban_storage_release_buffer) >= n_links
    ):
        # 초기화 완료(이 dict들엔 키 제거 경로가 없음) — setdefault 전수 루프 스킵.
        _sync_legacy_queues(state, cfg)
        return
    for movement, spec in movement_specs(cfg).items():
        state.urban_movement_queue.setdefault(
            movement,
            20.0 * float(spec.get("beta", 1.0)) if spec.get("kind") == "boundary_in" else 0.0,
        )
    # arrival buffer는 movement가 아니라 approach source(링크) 단위로 쌓는다 — β분할은 도착 시점.
    for source in approach_routing(cfg):
        state.urban_arrival_buffer.setdefault(source, {})
    for link, capacity in net.urban_link_storage_veh.items():
        state.urban_link_storage.setdefault(link, capacity)
        state.urban_storage_release_buffer.setdefault(link, {})
    _sync_legacy_queues(state, cfg)


def sync_onramp_queues_from_freeway(state: TrafficState, cfg: ExperimentConfig) -> None:
    """2저수지 구조에서는 freeway ramp queue를 urban 접근부 queue로 복사하지 않는다."""
    ensure_urban_state(state, cfg)
    for ramp, movements in cfg.network.on_ramp_to_movement.items():
        state.ramp_queue[ramp] = float(np.clip(
            state.ramp_queue.get(ramp, 0.0),
            0.0,
            cfg.network.ramp_queue_max_veh,
        ))
        for movement in movements:
            state.urban_movement_queue[movement] = max(
                0.0,
                state.urban_movement_queue.get(movement, 0.0),
            )
    _sync_legacy_queues(state, cfg)


def sync_onramp_queues_to_freeway(state: TrafficState, cfg: ExperimentConfig) -> None:
    """2저수지 구조에서는 urban 접근부 queue와 freeway ramp queue를 독립으로 유지한다."""
    sync_onramp_queues_from_freeway(state, cfg)


def schedule_offramp_arrivals(
    state: TrafficState,
    cfg: ExperimentConfig,
    off_ramp: str,
    vehicles: float,
    urban_step_index: int,
) -> tuple[float, float]:
    """Insert freeway-to-off-ramp vehicles into directed urban storage.

    off-ramp 차량은 storage 링크(= Wu 식17 off-ramp 큐 n_{m,d})에 적재된다. 여기서는
    점유만 생성(보존: accepted = 유입)하고, 교차로로의 방출은 매 substep에서 하류
    receiving 공간에 게이트해 드레인한다(Wu 식3, `_drain_offramp_storage`). 이전 구현은
    고정 시간지연 후 하류 정체와 무관하게 무조건 release해 storage가 transit 링크로만
    동작(점유 self-limiting)했고 capacity-drop(식22)이 발동하지 않았다.
    arrival buffer/release buffer 스케줄을 쓰지 않으므로 하류 정체 시 점유가 누적된다.
    Returns `(accepted, rejected)`.
    """
    ensure_urban_state(state, cfg)
    if vehicles <= 0.0:
        return 0.0, 0.0
    net = cfg.network
    storage_link = net.off_ramp_storage_link[off_ramp]
    available = max(0.0, state.urban_link_storage.get(storage_link, 0.0))
    accepted = min(float(vehicles), available)
    rejected = max(0.0, float(vehicles) - accepted)
    if accepted <= 0.0:
        return 0.0, rejected
    state.urban_link_storage[storage_link] = max(0.0, available - accepted)
    return accepted, rejected


def off_ramp_capacity_by_freeway_link(
    state: TrafficState,
    cfg: ExperimentConfig,
    interval_h: float | None = None,
) -> Dict[str, float]:
    ensure_urban_state(state, cfg)
    horizon_h = cfg.simulation.T_c_h if interval_h is None else interval_h
    cap: Dict[str, float] = {}
    for off_ramp in cfg.network.off_ramps:
        link = cfg.network.off_ramp_from_freeway[off_ramp]
        storage_link = cfg.network.off_ramp_storage_link[off_ramp]
        # S_eff: storage 링크에 origin을 둔 점큐가 있으면 반영(spec §3.3.2, 397행).
        available = _effective_available_space(state, cfg, storage_link)
        flow_cap = available / max(horizon_h, 1.0e-9)
        # off-ramp별 cap을 기본으로 제공하고, link aggregate는 legacy 호출 호환용으로 보존한다.
        cap[off_ramp] = flow_cap
        cap[link] = cap.get(link, 0.0) + flow_cap
    return cap


def _urban_step_index(state: TrafficState, cfg: ExperimentConfig) -> int:
    return int(round(state.time_sec / max(cfg.simulation.T_u_sec, 1.0e-9)))


def _schedule(buffer: Dict[str, Dict[int, float]], key: str, step: int, vehicles: float) -> None:
    if vehicles <= 0.0:
        return
    buffer.setdefault(key, {})
    buffer[key][step] = buffer[key].get(step, 0.0) + float(vehicles)


def _pop_buffer(buffer: Dict[str, Dict[int, float]], key: str, step: int) -> float:
    values = buffer.setdefault(key, {})
    return float(values.pop(step, 0.0))


def _drain_offramp_storage(
    state: TrafficState,
    control: ControlAction,
    cfg: ExperimentConfig,
    specs: Mapping[str, Mapping[str, object]],
    step_idx: int,
    routing: Mapping[str, list[tuple[str, float]]],
) -> Dict[str, float]:
    """off-ramp storage(=Wu 식17 off-ramp 큐)를 하류 교차로로 Wu 식3대로 방출한다.

    방출률 = green·포화유율·하류 수용공간의 min. storage 점유를 β로 off_ramp movement에
    나눠, 각 movement의 하류 receiving_link 가용공간에 게이트해 방출한다. 하류가 차면
    방출이 막혀 storage 점유가 누적(spillback) → effective_lane_profile의 λ_eff↓(식22).
    혼잡 해소 시 가용공간 복원으로 정상 방출(점유·λ_eff 복원).

    차량보존: storage occupancy(cap−available)와 movement 큐·하류 링크 점유 모두
    urban_total_vehicles에 포함되므로 storage→하류 링크 이동은 점유 중립이다. 여기서는
    off_ramp movement 큐를 경유하지 않고 storage→receiving_link로 직접 전달한다
    (off_ramp movement는 교차로 stop-line이 아니라 ramp 합류부라 별도 신호 대기 큐가 없다).
    Returns off_ramp별 방출 차량수.
    """
    net = cfg.network
    dt_h = cfg.simulation.T_u_h
    departures: Dict[str, float] = {}
    for off_ramp in net.off_ramps:
        storage_link = net.off_ramp_storage_link.get(off_ramp, "")
        if not storage_link or storage_link not in state.urban_link_storage:
            continue
        capacity = float(net.urban_link_storage_veh.get(storage_link, 0.0))
        occupancy = max(0.0, capacity - state.urban_link_storage.get(storage_link, capacity))
        if occupancy <= 0.0:
            continue
        movements = [m for m in net.off_ramp_to_movement.get(off_ramp, []) if m in specs]
        released_total = 0.0
        for movement in movements:
            spec = specs[movement]
            beta = float(spec.get("beta", 0.0))
            if beta <= 0.0:
                continue
            cap_flow = _movement_capacity_flow(control, cfg, movement, spec)
            green_fraction = _phase_green_fraction(control, cfg, spec, urban_step_index=step_idx)
            # Wu 식3: green·포화유율·(β 몫의 storage 점유)·하류 수용공간의 min.
            intended = min(beta * occupancy, dt_h * green_fraction * cap_flow)
            receiving_link = str(spec.get("receiving_link", ""))
            if receiving_link and receiving_link in state.urban_link_storage:
                # S_eff: 하류 링크 점큐 반영(spec §3.3.2, 397행) → off-ramp spillback(식22).
                receiving_space = _effective_available_space(state, cfg, receiving_link)
                actual = min(intended, receiving_space)
            else:
                actual = intended
            if actual <= 0.0:
                continue
            released_total += actual
            if receiving_link in state.urban_link_storage:
                state.urban_link_storage[receiving_link] = max(
                    0.0,
                    state.urban_link_storage.get(receiving_link, 0.0) - actual,
                )
                delay_steps = _link_delay_steps(state, cfg, receiving_link)
                arrival_step = step_idx + delay_steps
                if receiving_link in routing:
                    _schedule(state.urban_arrival_buffer, receiving_link, arrival_step, actual)
                _schedule(state.urban_storage_release_buffer, receiving_link, arrival_step, actual)
        if released_total > 0.0:
            state.urban_link_storage[storage_link] = min(
                capacity,
                state.urban_link_storage.get(storage_link, 0.0) + released_total,
            )
        departures[off_ramp] = released_total
    return departures


def _link_delay_steps(state: TrafficState, cfg: ExperimentConfig, storage_link: str) -> int:
    """내부 link travel delay(substep 수) — spec §3.3.5.

    큐 꼬리까지의 이동거리 = 가용 여유공간 S(=available). 빈 링크일수록 꼬리가 멀어 통과시간↑ →
    차량이 체류(내부 누적 형성). 큐가 차면 꼬리가 입구에 있어 즉시 도달. (이전 구현은 S 대신
    occupied=capacity−available를 써서 빈 링크 통과≈0 → 누적이 안 생기던 버그였다.)"""
    net = cfg.network
    capacity = net.urban_link_storage_veh.get(storage_link, net.boundary_queue_max_veh)
    available = max(0.0, state.urban_link_storage.get(storage_link, capacity))
    distance_km = available * net.urban_avg_vehicle_length_m / 1000.0
    travel_time_h = distance_km / max(net.urban_avg_speed_km_h, 1.0e-9)
    return max(1, int(math.ceil(travel_time_h / max(cfg.simulation.T_u_h, 1.0e-9))))


def _queue_max(cfg: ExperimentConfig, movement: str, spec: Mapping[str, object]) -> float:
    """movement 큐 클립 상한 — 사실상 비활성(점큐 모델, 보존 우선).

    큐 클립은 차량을 삭제해 보존 회계와 베이스라인 대비 공정성을 깬다(큐를 캡에
    잡아두는 쪽이 삭제 보조를 받음). 공간 제약(spillback)은 receiving-space
    allocation이 담당하므로 여기서는 수치 가드 수준의 큰 값만 둔다."""
    return 1.0e9


def _phase_green_fraction(
    control: ControlAction,
    cfg: ExperimentConfig,
    spec: Mapping[str, object],
    urban_step_index: int | None = None,
) -> float:
    """phase의 green 비율.

    urban_step_index가 주어지면 cycle 위상(offset 반영) 기반으로 해당 substep
    [t, t+T_u)와 green window의 겹침 비율(이진 + 경계 분수)을 돌려준다 — offset이
    plant 동역학(플래툰 도착–green 정렬)에 들어가는 유일한 지점. 없으면 기존처럼
    cycle 평균 비율(예측/추정 헬퍼용). 신호 cycle 구조는
    [p1 green][lost/2][p2 green][lost/2], 시작점이 offset만큼 뒤로 이동한다.
    cycle 평균 서비스량은 두 방식이 동일(g_sec×saturation)해 기존 회계와 정합."""
    phase = str(spec.get("phase", ""))
    if not phase:
        return 1.0
    net = cfg.network
    default_green = net.effective_green_total / 2.0
    green_sec = float(control.green_times.get(phase, default_green))
    cycle = max(net.cycle_length, 1.0e-9)
    if urban_step_index is None:
        return float(np.clip(green_sec / cycle, 0.0, 1.0))
    signal, _, phase_id = phase.rpartition("_")
    g1 = float(control.green_times.get(f"{signal}_p1", default_green))
    half_lost = max(0.0, net.lost_time) / 2.0
    start = 0.0 if phase_id == "p1" else g1 + half_lost
    end = min(start + green_sec, cycle)
    offset = float(control.offsets.get(signal, 0.0))
    t_u = cfg.simulation.T_u_sec
    t0 = (urban_step_index * t_u - offset) % cycle

    def seg_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    overlap = seg_overlap(t0, min(t0 + t_u, cycle), start, end)
    if t0 + t_u > cycle:
        # substep이 cycle 경계를 넘어가면 다음 cycle 머리쪽 green과의 겹침을 더한다.
        overlap += seg_overlap(0.0, t0 + t_u - cycle, start, end)
    return float(np.clip(overlap / max(t_u, 1.0e-9), 0.0, 1.0))


def _movement_capacity_flow(
    control: ControlAction,
    cfg: ExperimentConfig,
    movement: str,
    spec: Mapping[str, object],
) -> float:
    net = cfg.network
    kind = str(spec.get("kind", ""))
    # 내부(internal) movement는 allocation으로 throttle되지 않는다(불변식).
    # allocation cap은 perimeter movement(boundary_in/boundary_out/on_ramp/off_ramp)에만 적용.
    if kind not in PERIMETER_MOVEMENT_KINDS:
        return float(net.movement_capacity_veh_h)
    origin = str(spec.get("origin", ""))
    destination = str(spec.get("destination", ""))
    return float(min(
        control.inflow_outflow_allocation.get(
            movement,
            control.inflow_outflow_allocation.get(
                origin,
                control.inflow_outflow_allocation.get(destination, net.movement_capacity_veh_h),
            ),
        ),
        net.movement_capacity_veh_h,
    ))


def _allocate_receiving_counts(rule: str, intended: Dict[str, float], total_space: float) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in intended.values())
    if total <= total_space or total <= 1.0e-9:
        return {k: max(v, 0.0) for k, v in intended.items()}
    if rule == "equal_split":
        share = total_space / max(len(intended), 1)
        return {k: min(max(v, 0.0), share) for k, v in intended.items()}
    if rule == "main_priority":
        out: Dict[str, float] = {}
        remaining = total_space
        for key in sorted(intended, key=lambda item: 0 if item.startswith("in_") else 1):
            val = min(max(intended[key], 0.0), remaining)
            out[key] = val
            remaining -= val
        return out
    return {k: max(v, 0.0) * total_space / total for k, v in intended.items()}


def _sync_legacy_queues(state: TrafficState, cfg: ExperimentConfig) -> None:
    umq_get = state.urban_movement_queue.get
    for link, moves in _legacy_sync_index(cfg):
        related = 0.0
        for movement in moves:
            related += umq_get(movement, 0.0)
        state.boundary_queue[link] = max(0.0, related)
        state.urban_queue[link] = max(0.0, related)


def _storage_occupancy(state: TrafficState, cfg: ExperimentConfig) -> float:
    """링크 in-transit 점유(cap−available) 합계. movement 큐는 포함하지 않는다.

    off-ramp 램프 storage(`OR_*_storage`)는 freeway로 재귀속(design 2026-06-17)하므로
    urban 점유 합에서 제외한다(그 점유는 `_offramp_storage_occupancy`로 별도 집계).
    """
    off_ramp_storage_links = set(cfg.network.off_ramp_storage_link.values())
    total = 0.0
    for link, capacity in cfg.network.urban_link_storage_veh.items():
        if link in off_ramp_storage_links:
            continue
        total += max(0.0, capacity - state.urban_link_storage.get(link, capacity))
    return float(total)


def _offramp_storage_occupancy(state: TrafficState, cfg: ExperimentConfig) -> float:
    """off-ramp 램프 storage(`OR_*_storage`) in-transit 점유[veh] 합.

    이 점유의 TTT는 freeway_ttt로 귀속된다(design 2026-06-17). urban_ttt에서는 빠지고
    coupling이 이 값을 받아 freeway_ttt에 더한다(보존: 같은 양이 한쪽에서 빠져 다른 쪽에 더해짐).
    """
    off_ramp_storage_links = set(cfg.network.off_ramp_storage_link.values())
    total = 0.0
    for link in off_ramp_storage_links:
        capacity = cfg.network.urban_link_storage_veh.get(link)
        if capacity is None:
            continue
        total += max(0.0, capacity - state.urban_link_storage.get(link, capacity))
    return float(total)


def urban_accumulation_feedback_flow(
    state: TrafficState,
    cfg: ExperimentConfig,
    target_accumulation_veh: float,
    forecast: "list[DemandStep] | None" = None,
) -> float:
    """목표 도시 누적(N_P_star, veh)을 추적하기 위한 허용 순유입(veh/h)을 계산한다.

    forecast가 주어지면 예측 외란 유입(off-ramp 도달)을 보정한다(진단 문서 §4). off-ramp
    유입은 제어 불가 외란으로 N_P를 늘리므로, 그만큼 허용 순유입(boundary 게이팅)을 미리
    낮춰 곧 닥칠 수요 파동에 N_P가 폭증하지 않게 한다. forecast=None이면 기존 현재상태
    feedback만 사용(하위 호환)."""
    ensure_urban_state(state, cfg)
    error_veh = float(target_accumulation_veh) - state.protected_accumulation_veh(cfg.network)
    feedback_h = max(float(cfg.leader.N_P_feedback_horizon_h), 1.0e-9)
    raw_flow = error_veh / feedback_h
    if forecast:
        raw_flow -= _forecast_offramp_inflow_veh_h(state, cfg, forecast)
    limit = max(0.0, float(cfg.leader.N_P_feedback_flow_limit_veh_h))
    return float(np.clip(raw_flow, -limit, limit))


def _forecast_offramp_inflow_veh_h(
    state: TrafficState,
    cfg: ExperimentConfig,
    forecast: "list[DemandStep]",
) -> float:
    """horizon 평균 예측 off-ramp 유입[veh/h] — N_P를 늘리는 제어불가 외란.

    off-ramp 도달 유량 = freeway link 끝 유량 × split, forecast 본선 수요 비율로 스케일."""
    net = cfg.network
    steps = forecast[: max(1, cfg.mpc.horizon_steps)]
    if not steps:
        return 0.0
    total = 0.0
    for off_ramp in net.off_ramps:
        link = net.off_ramp_from_freeway.get(off_ramp, "")
        split = net.off_ramp_split_ratio.get(off_ramp, 0.0)
        flows = state.freeway_flow.get(link, [])
        base_flow = max(0.0, float(flows[-1]) if flows else 0.0)
        base_main = max(1.0e-9, float(steps[0].freeway_mainline.get(link, 0.0)))
        for s in steps:
            scale = max(0.0, float(s.freeway_mainline.get(link, 0.0))) / base_main
            total += base_flow * scale * split
    return float(total / max(len(steps), 1))


def estimate_onramp_green_release_flows(
    state: TrafficState,
    control: ControlAction,
    demand: DemandStep,
    cfg: ExperimentConfig,
    interval_h: float | None = None,
) -> Dict[str, float]:
    """on-ramp 접근부 x_on에서 ramp queue w_r로 넘어갈 수 있는 유량을 예측한다.

    Freeway follower의 경량 예측에서는 urban follower를 후보마다 다시 풀지 않고,
    현재 green/allocation을 고정한 boundary forecast만 사용한다. 이 helper는 상태를
    직접 갱신하지 않고 veh/h 단위의 예상 유량만 돌려준다.
    """
    ensure_urban_state(state, cfg)
    net = cfg.network
    specs = movement_specs(cfg)
    horizon_h = cfg.simulation.T_f_h if interval_h is None else interval_h
    release: Dict[str, float] = {}
    for ramp, movements in net.on_ramp_to_movement.items():
        requested_total = 0.0
        arrival = max(0.0, demand.ramp_arrival.get(ramp, 0.0)) * horizon_h
        arrival_share = arrival / max(len(movements), 1)
        for movement in movements:
            spec = specs.get(movement)
            if spec is None:
                continue
            available = max(0.0, state.urban_movement_queue.get(movement, 0.0)) + arrival_share
            cap_flow = _movement_capacity_flow(control, cfg, movement, spec)
            green_fraction = _phase_green_fraction(control, cfg, spec)
            requested_total += min(available, horizon_h * green_fraction * cap_flow)
        ramp_space = max(0.0, net.ramp_queue_max_veh - state.ramp_queue.get(ramp, 0.0))
        release[ramp] = min(requested_total, ramp_space) / max(horizon_h, 1.0e-9)
    return release


def estimate_onramp_reservoir_inflow(
    state: TrafficState,
    control: ControlAction,
    demand: DemandStep,
    cfg: ExperimentConfig,
    interval_h: float | None = None,
) -> Dict[str, float]:
    """on-ramp 접근부 x_on에서 green이 방출하는 유량을 ramp space 캡 없이 예측한다.

    `estimate_onramp_green_release_flows`와 동일한 requested_total(큐+도착, green 용량
    제한) 계산을 재사용하되 w_r 상한(ramp_queue_max) 포화에 따른 `min(..., ramp_space)`
    캡을 제거한다. WU-CD-F coupling의 urban→freeway 결합변수 소스로, w_r이 포화한
    peak에서도 두 green 후보 간 방출 차이를 보존해 분산 협상이 후보 제어에 반응하게 한다.
    """
    ensure_urban_state(state, cfg)
    net = cfg.network
    specs = movement_specs(cfg)
    horizon_h = cfg.simulation.T_f_h if interval_h is None else interval_h
    release: Dict[str, float] = {}
    for ramp, movements in net.on_ramp_to_movement.items():
        requested_total = 0.0
        arrival = max(0.0, demand.ramp_arrival.get(ramp, 0.0)) * horizon_h
        arrival_share = arrival / max(len(movements), 1)
        for movement in movements:
            spec = specs.get(movement)
            if spec is None:
                continue
            available = max(0.0, state.urban_movement_queue.get(movement, 0.0)) + arrival_share
            cap_flow = _movement_capacity_flow(control, cfg, movement, spec)
            green_fraction = _phase_green_fraction(control, cfg, spec)
            requested_total += min(available, horizon_h * green_fraction * cap_flow)
        release[ramp] = requested_total / max(horizon_h, 1.0e-9)
    return release


def _control_net_inflow_target_veh_h(control: ControlAction, cfg: ExperimentConfig) -> float:
    diagnostics = getattr(control, "diagnostics", {}) or {}
    for key in (
        "urban_net_inflow_target_veh_h",
        "distributed_grid_leader_net_inflow_target_rate_veh_h",
    ):
        if key in diagnostics:
            return float(diagnostics.get(key, 0.0))
    horizon_steps = max(1, int(getattr(cfg.mpc, "horizon_steps", 1)))
    horizon_h = max(float(cfg.simulation.T_c_h) * horizon_steps, 1.0e-9)
    return float(control.N_P_star) / horizon_h


def urban_substep(
    state: TrafficState,
    control: ControlAction,
    demand: DemandStep,
    cfg: ExperimentConfig,
    urban_step_index: int | None = None,
    ramp_release_veh_h: Mapping[str, float] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """movement-level horizontal queue를 `T_u` 한 스텝만 전진한다."""
    ensure_urban_state(state, cfg)
    net = cfg.network
    sim = cfg.simulation
    specs = movement_specs(cfg)
    diagnostics: Dict[str, float] = {
        "movement_queue_model_active": 1.0,
        "urban_storage_active": 1.0,
        "urban_substep_active": 1.0,
        "onramp_two_reservoir_active": 1.0,
    }
    initial_accumulation = state.protected_accumulation_veh(cfg.network)
    interval_net_inflow_target = _control_net_inflow_target_veh_h(control, cfg)
    initial_accumulation_error = 0.0
    overflow_count = 0.0
    projection_count = 0.0
    total_departures_veh = 0.0
    inbound_service_veh = 0.0
    outbound_service_veh = 0.0
    onramp_arrivals_veh = 0.0
    onramp_green_release_request_veh = 0.0
    onramp_green_releases_veh = 0.0
    onramp_green_release_shortfall_veh = 0.0
    ramp_metering_release_request_veh = 0.0
    ramp_metering_releases_veh = 0.0
    ramp_metering_release_shortfall_veh = 0.0
    ramp_metering_request_by_ramp: Dict[str, float] = {ramp: 0.0 for ramp in net.ramps}
    ramp_metering_actual_by_ramp: Dict[str, float] = {ramp: 0.0 for ramp in net.ramps}
    ramp_metering_shortfall_by_ramp: Dict[str, float] = {ramp: 0.0 for ramp in net.ramps}
    off_ramp_departures: Dict[str, float] = {r: 0.0 for r in net.off_ramps}
    step_idx = _urban_step_index(state, cfg) if urban_step_index is None else urban_step_index
    routing = approach_routing(cfg)
    sink_links = sink_storage_links(cfg)
    boundary_out_sink_veh = 0.0
    urban_gate_inflow_veh = 0.0
    urban_demand_arrivals_veh = 0.0

    for link in net.urban_link_storage_veh:
        released = _pop_buffer(state.urban_storage_release_buffer, link, step_idx)
        if released > 0.0:
            # sink(boundary_out) 링크는 release pop에서 available을 복원하지 않는다 —
            # 차량은 out 링크 점유 상태로 남아(in-transit→링크 끝 대기) 아래 유한 출구용량
            # 게이트가 cap만큼만 이탈시킨다(A″-1). available을 복원하면 점유가 사라져
            # (a) receiving 게이트에 backup이 안 잡히고 (b) sink로 기록되지 않아 차량이
            # 소멸한다(보존 위반). 내부 링크는 기존대로 복원해 다음 노드로 이동시킨다.
            if link in sink_links:
                continue
            cap = net.urban_link_storage_veh[link]
            state.urban_link_storage[link] = min(cap, state.urban_link_storage.get(link, cap) + released)

    # 유한 출구용량(A″-1): 각 boundary_out 링크에서 시스템을 떠나는 유량을
    # min(out 링크 점유, exit_capacity·dt)로 제약한다(모델 밖 하류 도로 용량). 못 나간
    # 차량은 out 링크 storage에 남아(보존) 점유가 누적되고, 그 점유가
    # `_effective_available_space`(receiving 게이트)에 반영돼 exit movement가 막히면서
    # backup이 grid로 전파된다(S_eff 일관). 자유 sink 시절엔 도착 즉시 전량 이탈해
    # off-ramp 홍수가 urban을 포화시키지 못했다(roadmap §3).
    exit_capacity_veh_h = float(net.boundary_out_capacity_veh_h)
    finite_exit = exit_capacity_veh_h > 0.0
    exit_capacity_veh = exit_capacity_veh_h * sim.T_u_h
    for link in sink_links:
        cap = net.urban_link_storage_veh.get(link, net.boundary_queue_max_veh)
        occupancy = max(0.0, cap - state.urban_link_storage.get(link, cap))
        # 유한용량이면 min(점유, exit_cap·dt), 0 이하이면 자유 sink(점유 전량 이탈, 하위호환).
        departed = min(occupancy, exit_capacity_veh) if finite_exit else occupancy
        if departed <= 0.0:
            continue
        state.urban_link_storage[link] = min(cap, state.urban_link_storage.get(link, cap) + departed)
        boundary_out_sink_veh += departed

    # arrival = "다음 노드 도착 → β분할" (spec §3.3.5). 1:1 next_movement 체인이 아님.
    for source, targets in routing.items():
        arrived = _pop_buffer(state.urban_arrival_buffer, source, step_idx)
        if arrived <= 0.0:
            continue
        for movement, beta in targets:
            state.urban_movement_queue[movement] += beta * arrived

    # 게이트 수요는 그 교차로에 "해당 방향에서 도착"으로 주입 후 동일 β분할.
    for origin in net.boundary_in_links:
        arrival = demand.urban_boundary.get(origin, 0.0) * sim.T_u_h
        targets = routing.get(origin, [])
        if arrival <= 0.0 or not targets:
            continue
        urban_demand_arrivals_veh += arrival
        for movement, beta in targets:
            state.urban_movement_queue[movement] += beta * arrival

    # 외생 on-ramp 수요는 먼저 urban 접근부 저수지 x_on(해당 ramp의 on_ramp movement들)에 쌓인다.
    for ramp, movements in net.on_ramp_to_movement.items():
        arrival = max(0.0, demand.ramp_arrival.get(ramp, 0.0)) * sim.T_u_h
        if arrival <= 0.0 or not movements:
            continue
        share = arrival / len(movements)
        for movement in movements:
            state.urban_movement_queue[movement] = state.urban_movement_queue.get(movement, 0.0) + share
        onramp_arrivals_veh += arrival

    # ramp metering은 freeway ramp 저수지 w_r에서 freeway로 빠져나가는 흐름이다.
    if ramp_release_veh_h is not None:
        for ramp, release_flow in ramp_release_veh_h.items():
            requested = max(0.0, release_flow) * sim.T_u_h
            before = max(0.0, state.ramp_queue.get(ramp, 0.0))
            actual = min(before, requested)
            shortfall = max(0.0, requested - actual)
            state.ramp_queue[ramp] = max(0.0, before - actual)
            ramp_metering_release_request_veh += requested
            ramp_metering_releases_veh += actual
            ramp_metering_release_shortfall_veh += shortfall
            ramp_metering_request_by_ramp[ramp] = ramp_metering_request_by_ramp.get(ramp, 0.0) + requested
            ramp_metering_actual_by_ramp[ramp] = ramp_metering_actual_by_ramp.get(ramp, 0.0) + actual
            ramp_metering_shortfall_by_ramp[ramp] = ramp_metering_shortfall_by_ramp.get(ramp, 0.0) + shortfall

    # urban green은 접근부 저수지(x_on, ramp행 movement들)에서 freeway ramp 저수지 w_r로
    # 보내는 흐름이다. ramp는 sink가 아니라 freeway로의 transfer(차량 보존).
    ramp_requests: Dict[str, Dict[str, float]] = {}
    for movement, spec in specs.items():
        ramp = str(spec.get("ramp", ""))
        if not ramp:
            continue
        available = max(0.0, state.urban_movement_queue.get(movement, 0.0))
        cap_flow = _movement_capacity_flow(control, cfg, movement, spec)
        green_fraction = _phase_green_fraction(control, cfg, spec, urban_step_index=step_idx)
        ramp_requests.setdefault(ramp, {})[movement] = min(
            available,
            sim.T_u_h * green_fraction * cap_flow,
        )
    for ramp, requests in ramp_requests.items():
        requested_total = sum(requests.values())
        ramp_space = max(0.0, net.ramp_queue_max_veh - state.ramp_queue.get(ramp, 0.0))
        scale = 1.0 if requested_total <= ramp_space else ramp_space / max(requested_total, 1.0e-9)
        released_total = 0.0
        for movement, requested in requests.items():
            actual = requested * scale
            before = max(0.0, state.urban_movement_queue.get(movement, 0.0))
            actual = min(before, actual)
            state.urban_movement_queue[movement] = max(0.0, before - actual)
            released_total += actual
            total_departures_veh += actual
            outbound_service_veh += actual
            if str(specs[movement].get("kind", "")) in {"boundary_in", "off_ramp"}:
                # 게이트에서 곧장 ramp로 가는 movement는 perimeter 유입이기도 하다.
                inbound_service_veh += actual
        state.ramp_queue[ramp] = min(
            net.ramp_queue_max_veh,
            max(0.0, state.ramp_queue.get(ramp, 0.0) + released_total),
        )
        onramp_green_release_request_veh += requested_total
        onramp_green_releases_veh += released_total
        onramp_green_release_shortfall_veh += max(0.0, requested_total - released_total)

    # off-ramp storage(=Wu 식17 큐)를 하류 receiving 공간에 게이트해 방출(Wu 식3). 하류
    # 정체 시 방출이 막혀 storage 점유 누적(spillback)→λ_eff↓(식22). stage 2 전에 실행해
    # 같은 substep의 하류 가용공간을 두 흐름이 공유하게 한다.
    drained = _drain_offramp_storage(state, control, cfg, specs, step_idx, routing)
    for off_ramp, released in drained.items():
        off_ramp_departures[off_ramp] = off_ramp_departures.get(off_ramp, 0.0) + released
        total_departures_veh += released
        inbound_service_veh += released  # off-ramp 합류 = perimeter 유입.

    intended_by_storage: Dict[str, Dict[str, float]] = {}
    no_storage_intended: Dict[str, float] = {}
    for movement, spec in specs.items():
        if str(spec.get("ramp", "")):
            continue  # ramp행 movement는 위 transfer 루프에서 처리됨.
        if str(spec.get("kind", "")) == "off_ramp":
            continue  # off_ramp movement는 _drain_offramp_storage가 storage에서 직접 방출.
        available = max(0.0, state.urban_movement_queue.get(movement, 0.0))
        cap_flow = _movement_capacity_flow(control, cfg, movement, spec)
        green_fraction = _phase_green_fraction(control, cfg, spec, urban_step_index=step_idx)
        intended = min(available, sim.T_u_h * green_fraction * cap_flow)
        receiving_link = str(spec.get("receiving_link", ""))
        if receiving_link and receiving_link in state.urban_link_storage:
            intended_by_storage.setdefault(receiving_link, {})[movement] = intended
        else:
            no_storage_intended[movement] = intended

    actual_departure: Dict[str, float] = dict(no_storage_intended)
    for storage_link, intended in intended_by_storage.items():
        # S_eff: 하류 링크 끝 점큐를 점유로 반영(spec §3.3.2, 397행) → backup 전파.
        available_space = _effective_available_space(state, cfg, storage_link)
        actual_departure.update(_allocate_receiving_counts(
            cfg.urban_follower.receiving_space_rule,
            intended,
            available_space,
        ))

    for movement, departed in actual_departure.items():
        if departed <= 0.0:
            continue
        spec = specs[movement]
        before = state.urban_movement_queue.get(movement, 0.0)
        actual = min(before, departed)
        total_departures_veh += actual
        state.urban_movement_queue[movement] = max(0.0, before - actual)
        receiving_link = str(spec.get("receiving_link", ""))
        if receiving_link in state.urban_link_storage:
            state.urban_link_storage[receiving_link] = max(
                0.0,
                state.urban_link_storage.get(receiving_link, 0.0) - actual,
            )
            delay_steps = _link_delay_steps(state, cfg, receiving_link)
            arrival_step = step_idx + delay_steps
            # 내부 링크면 다음 교차로 approach buffer로(도착 시 β분할), sink 링크면 release만.
            if receiving_link in routing:
                _schedule(state.urban_arrival_buffer, receiving_link, arrival_step, actual)
            _schedule(state.urban_storage_release_buffer, receiving_link, arrival_step, actual)
        if spec.get("kind") == "off_ramp":
            off_ramp = str(spec.get("off_ramp", ""))
            off_ramp_departures[off_ramp] = off_ramp_departures.get(off_ramp, 0.0) + actual
            inbound_service_veh += actual
        elif spec.get("kind") == "boundary_in":
            inbound_service_veh += actual
            urban_gate_inflow_veh += actual
        elif spec.get("kind") == "boundary_out":
            outbound_service_veh += actual

    projection_protected_veh = 0.0
    protected_queue_kinds = {"internal", "boundary_out", "off_ramp"}
    for movement, spec in specs.items():
        qmax = _queue_max(cfg, movement, spec)
        q = state.urban_movement_queue.get(movement, 0.0)
        if q > qmax:
            overflow_count += 1.0
            projection_count += q - qmax
            if str(spec.get("kind", "")) in protected_queue_kinds:
                projection_protected_veh += q - qmax
            state.urban_movement_queue[movement] = qmax

    # off-ramp 램프 storage 점유는 freeway로 재귀속(design 2026-06-17). _storage_occupancy가
    # 이미 제외하므로 urban_ttt에서 자동으로 빠진다. 그 양의 TTT는 진단으로 노출해 coupling이
    # freeway_ttt에 더한다(보존: urban에서 빠진 양 = freeway에 더해지는 양, 같은 T_u_h 단위).
    offramp_storage_occupancy = _offramp_storage_occupancy(state, cfg)
    offramp_storage_ttt = offramp_storage_occupancy * sim.T_u_h
    uncontrolled_node_movement = state.uncontrolled_node_movement_queue_veh(net)
    uncontrolled_node_storage = state.uncontrolled_node_storage_occupancy_veh(net)
    uncontrolled_node_vehicles = uncontrolled_node_movement + uncontrolled_node_storage
    urban_ttt = (
        sum(state.urban_movement_queue.values())
        + _storage_occupancy(state, cfg)
    ) * sim.T_u_h

    _sync_legacy_queues(state, cfg)
    inbound = inbound_service_veh / max(sim.T_u_h, 1.0e-9)
    outbound = outbound_service_veh / max(sim.T_u_h, 1.0e-9)
    net_inflow = inbound - outbound
    accumulation_error = 0.0
    net_inflow_error = abs(net_inflow - interval_net_inflow_target)
    diagnostics["inbound_service_veh"] = float(inbound_service_veh)
    diagnostics["outbound_service_veh"] = float(outbound_service_veh)
    diagnostics["urban_total_departures_veh"] = float(total_departures_veh)
    diagnostics["net_inflow"] = float(net_inflow)
    diagnostics["net_inflow_target"] = float(interval_net_inflow_target)
    diagnostics["urban_net_inflow_target_veh_h"] = float(interval_net_inflow_target)
    diagnostics["urban_accumulation_initial_veh"] = float(initial_accumulation)
    diagnostics["urban_accumulation_initial_error_veh"] = float(initial_accumulation_error)
    diagnostics["urban_accumulation_veh"] = float(state.protected_accumulation_veh(cfg.network))
    diagnostics["urban_accumulation_target_disabled"] = 1.0
    diagnostics["urban_accumulation_target_veh"] = 0.0
    diagnostics["urban_accumulation_error_veh"] = float(accumulation_error)
    diagnostics["urban_accumulation_abs_error_veh"] = abs(float(accumulation_error))
    diagnostics["urban_net_inflow_tracking_error_veh_h"] = float(net_inflow_error)
    diagnostics["net_inflow_tracking_error"] = float(net_inflow_error)
    diagnostics.update(movement_balance_summary(
        state,
        cfg,
        saturation_fraction=cfg.evaluation.boundary_degenerate_saturation_fraction,
        degenerate_ratio=cfg.evaluation.boundary_degenerate_ratio,
        eps=cfg.evaluation.eps,
    ))
    diagnostics.update(boundary_indices(state.boundary_queue.values(), net.boundary_queue_max_veh))
    diagnostics["queue_overflow_count"] = float(overflow_count)
    diagnostics["movement_queue_projection_veh"] = float(projection_count)
    diagnostics["movement_queue_projection_protected_veh"] = float(projection_protected_veh)
    diagnostics["urban_storage_occupancy"] = _storage_occupancy(state, cfg)
    diagnostics["urban_link_occupancy_veh"] = _storage_occupancy(state, cfg)
    diagnostics["urban_uncontrolled_node_movement_queue_veh"] = float(uncontrolled_node_movement)
    diagnostics["urban_uncontrolled_node_storage_occupancy_veh"] = float(uncontrolled_node_storage)
    diagnostics["urban_uncontrolled_node_vehicles_veh"] = float(uncontrolled_node_vehicles)
    diagnostics["urban_uncontrolled_node_ttt"] = float(uncontrolled_node_vehicles * sim.T_u_h)
    # off-ramp 램프 storage 재귀속(design 2026-06-17): freeway_ttt로 보낼 점유·TTT.
    diagnostics["offramp_storage_occupancy_veh"] = float(offramp_storage_occupancy)
    diagnostics["offramp_storage_ttt"] = float(offramp_storage_ttt)
    # 보존 회계(proposal §8): 유입(게이트 service + off_ramp 복귀 + 외생 ramp 수요)
    # = 이탈(boundary_out sink + on_ramp 전이) + Δ누적. off_ramp 복귀는 coupling에서 집계.
    diagnostics["urban_gate_inflow_veh"] = float(urban_gate_inflow_veh)
    diagnostics["urban_demand_arrivals_veh"] = float(urban_demand_arrivals_veh)
    diagnostics["boundary_out_sink_veh"] = float(boundary_out_sink_veh)
    diagnostics["urban_total_vehicles_veh"] = float(
        sum(state.urban_movement_queue.values()) + _storage_occupancy(state, cfg)
    )
    diagnostics["onramp_arrivals_veh"] = float(onramp_arrivals_veh)
    diagnostics["onramp_green_release_request_veh"] = float(onramp_green_release_request_veh)
    diagnostics["onramp_green_releases_veh"] = float(onramp_green_releases_veh)
    diagnostics["onramp_green_release_shortfall_veh"] = float(onramp_green_release_shortfall_veh)
    diagnostics["ramp_metering_release_request_veh"] = float(ramp_metering_release_request_veh)
    diagnostics["ramp_metering_releases_veh"] = float(ramp_metering_releases_veh)
    diagnostics["ramp_metering_release_shortfall_veh"] = float(ramp_metering_release_shortfall_veh)
    for ramp in net.ramps:
        diagnostics[f"ramp_metering_release_request_{ramp}_veh"] = float(
            ramp_metering_request_by_ramp.get(ramp, 0.0)
        )
        diagnostics[f"ramp_metering_release_actual_{ramp}_veh"] = float(
            ramp_metering_actual_by_ramp.get(ramp, 0.0)
        )
        diagnostics[f"ramp_metering_release_shortfall_{ramp}_veh"] = float(
            ramp_metering_shortfall_by_ramp.get(ramp, 0.0)
        )
    diagnostics["onramp_approach_queue_veh"] = float(sum(
        state.urban_movement_queue.get(movement, 0.0)
        for movements in net.on_ramp_to_movement.values()
        for movement in movements
    ))
    diagnostics["ramp_queue_veh"] = float(sum(state.ramp_queue.values()))
    diagnostics["offramp_departures_veh"] = float(sum(off_ramp_departures.values()))
    for off_ramp, value in off_ramp_departures.items():
        diagnostics[f"offramp_departures_{off_ramp}_veh"] = float(value)
    return float(urban_ttt), diagnostics


def aggregate_urban_diagnostics(
    rows: Iterable[Mapping[str, float]],
    cfg: ExperimentConfig,
    control: ControlAction,
    interval_h: float | None = None,
) -> Dict[str, float]:
    """여러 `urban_substep` diagnostics를 하나의 control/freeway interval 값으로 묶는다."""
    diagnostics_rows = [dict(row) for row in rows]
    if not diagnostics_rows:
        net_inflow_target = 0.0
        return {
            "movement_queue_model_active": 1.0,
            "urban_storage_active": 1.0,
            "urban_substep_active": 1.0,
            "onramp_two_reservoir_active": 1.0,
            "net_inflow": 0.0,
            "net_inflow_target": net_inflow_target,
            "urban_net_inflow_target_veh_h": net_inflow_target,
            "urban_net_inflow_target_veh": float(control.N_P_star),
            "urban_accumulation_target_disabled": 1.0,
            "urban_accumulation_target_veh": 0.0,
            "urban_accumulation_abs_error_veh": 0.0,
            "urban_net_inflow_tracking_error_veh_h": abs(net_inflow_target),
            "net_inflow_tracking_error": abs(net_inflow_target),
        }

    out = dict(diagnostics_rows[-1])
    sum_keys = {
        "inbound_service_veh",
        "outbound_service_veh",
        "urban_total_departures_veh",
        "queue_overflow_count",
        "movement_queue_projection_veh",
        "movement_queue_projection_protected_veh",
        "urban_gate_inflow_veh",
        "urban_demand_arrivals_veh",
        "boundary_out_sink_veh",
        "onramp_arrivals_veh",
        "onramp_green_release_request_veh",
        "onramp_green_releases_veh",
        "onramp_green_release_shortfall_veh",
        "ramp_metering_release_request_veh",
        "ramp_metering_releases_veh",
        "ramp_metering_release_shortfall_veh",
        "offramp_departures_veh",
        "urban_uncontrolled_node_ttt",
    }
    for key in set().union(*(row.keys() for row in diagnostics_rows)):
        if (
            key in sum_keys
            or (key.startswith("offramp_departures_") and key.endswith("_veh"))
            or (key.startswith("ramp_metering_release_") and key.endswith("_veh"))
        ):
            out[key] = float(sum(row.get(key, 0.0) for row in diagnostics_rows))
        elif key in {
            "movement_queue_model_active",
            "urban_storage_active",
            "urban_substep_active",
            "onramp_two_reservoir_active",
        }:
            out[key] = float(max(row.get(key, 0.0) for row in diagnostics_rows))

    # cycle 위상 plant에서 N_P는 cycle 주기로 진동하고 interval(180s)=1.5 cycle이라
    # endpoint 표본은 앨리어싱된다 — 추적 지표용으로 interval 평균 N_P를 함께 낸다.
    out["urban_accumulation_mean_veh"] = float(np.mean([
        row.get("urban_accumulation_veh", 0.0) for row in diagnostics_rows
    ]))

    horizon_h = cfg.simulation.T_c_h if interval_h is None else interval_h
    inbound = out.get("inbound_service_veh", 0.0) / max(horizon_h, 1.0e-9)
    outbound = out.get("outbound_service_veh", 0.0) / max(horizon_h, 1.0e-9)
    net_inflow = inbound - outbound
    # 제어 검증에서는 follower가 allocation을 만들 때 사용한 control-interval 목표와 비교한다.
    net_inflow_target = float(control.diagnostics.get(
        "urban_net_inflow_target_veh_h",
        out.get("urban_net_inflow_target_veh_h", out.get("net_inflow_target", 0.0)),
    ))
    net_inflow_error = abs(net_inflow - net_inflow_target)
    out["net_inflow"] = float(net_inflow)
    out["net_inflow_target"] = float(net_inflow_target)
    out["urban_net_inflow_target_veh_h"] = float(net_inflow_target)
    # Wu-faithful P-Stack은 raw leader intent와 follower-feasible projected target을 분리해
    # 기록한다. plant tracking 진단에는 실제 제어가 추적한 projected horizon target[veh]을 우선 사용한다.
    out["urban_net_inflow_target_veh"] = float(control.diagnostics.get(
        "urban_net_inflow_target_veh",
        control.N_P_star,
    ))
    out["urban_accumulation_target_disabled"] = 1.0
    out["urban_net_inflow_tracking_error_veh_h"] = float(net_inflow_error)
    out["net_inflow_tracking_error"] = float(net_inflow_error)
    return out


def urban_step(
    state: TrafficState,
    control: ControlAction,
    demand: DemandStep,
    cfg: ExperimentConfig,
) -> Tuple[float, Dict[str, float]]:
    """기존 API 호환 wrapper: 한 control interval 동안 `urban_substep`을 반복한다."""
    ensure_urban_state(state, cfg)
    total_ttt = 0.0
    diagnostics: list[Dict[str, float]] = []
    start_step = _urban_step_index(state, cfg)
    for substep in range(cfg.simulation.K_cu):
        ur_ttt, ur_diag = urban_substep(
            state,
            control,
            demand,
            cfg,
            urban_step_index=start_step + substep,
        )
        total_ttt += ur_ttt
        diagnostics.append(ur_diag)
    return float(total_ttt), aggregate_urban_diagnostics(diagnostics, cfg, control)
