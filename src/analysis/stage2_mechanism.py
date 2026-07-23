# Stage 2 control 메커니즘 검증 (plan §3~§8) — Trigger→Action→Mediator→Outcome event 분석
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState

# event 판정 상태(plan §3.1).
NOT_CHALLENGED = "NOT_CHALLENGED"
CORRECTLY_INACTIVE = "CORRECTLY_INACTIVE"
MECHANISM_REPRODUCED = "MECHANISM_REPRODUCED"
ACTIVATED_BUT_INEFFECTIVE = "ACTIVATED_BUT_INEFFECTIVE"
WRONG_DIRECTION = "WRONG_DIRECTION"
CONGESTION_SHIFT = "CONGESTION_SHIFT"

CONTROLS = ("allocation_green", "offset", "vsl", "metering")


@dataclass
class IntervalTrace:
    """interval별 의사결정·상태 스냅샷 — counterfactual replay의 분기점."""

    step: int
    time_sec: float
    state_before: TrafficState
    control: ControlAction
    diagnostics: Dict[str, float]
    interval_ttt: float
    urban_ttt: float
    freeway_ttt: float


@dataclass
class ControlEvent:
    control: str
    event_id: str
    t0_step: int
    trigger_value: float
    threshold: float
    status: str = NOT_CHALLENGED
    response_delay_intervals: Optional[int] = None
    directional_accuracy: Optional[bool] = None
    mediator_change: Dict[str, float] = field(default_factory=dict)
    outcome_actual_ttt: float = 0.0
    outcome_counterfactual_ttt: float = 0.0
    outcome_gain: float = 0.0
    shifted_subsystem_delta: float = 0.0


def run_traced_closed_loop(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
    controller_id: str = "PROPOSED-STACKELBERG",
) -> List[IntervalTrace]:
    """PROPOSED-STACKELBERG closed loop를 실행하며 interval 시작 상태를 보존한다."""
    from src.experiments.six_controller_comparison import _ControllerAdapter
    from src.simulation.simulator import MixedTrafficSimulator

    profile = DemandProfile(cfg, scenario)
    sim = MixedTrafficSimulator(cfg)
    adapter = _ControllerAdapter(cfg, controller_id)
    steps = max(1, int(round(cfg.simulation.T_total / cfg.simulation.control_interval)))
    traces: List[IntervalTrace] = []
    for step in range(steps):
        t = step * cfg.simulation.control_interval
        forecast = profile.horizon(t, cfg.mpc.horizon_steps)
        state_before = sim.state.copy()
        control, _ = adapter.decide(sim.state.copy(), forecast)
        log = sim.step(control, forecast[0], step)
        traces.append(IntervalTrace(
            step=step,
            time_sec=sim.state.time_sec,
            state_before=state_before,
            control=control,
            diagnostics={k: float(v) for k, v in log.diagnostics.items() if isinstance(v, (int, float, bool))},
            interval_ttt=log.freeway_ttt + log.urban_ttt,
            urban_ttt=log.urban_ttt,
            freeway_ttt=log.freeway_ttt,
        ))
    return traces


def _neutral_control(cfg: ExperimentConfig, control: ControlAction, target: str) -> ControlAction:
    """frozen replay용 — 대상 control만 neutral로 바꾸고 나머지는 적용값 유지(plan §3.3)."""
    net = cfg.network
    neutral = ControlAction(
        ramp_metering=dict(control.ramp_metering),
        vsl=dict(control.vsl),
        green_times=dict(control.green_times),
        offsets=dict(control.offsets),
        inflow_outflow_allocation=dict(control.inflow_outflow_allocation),
        N_P_star=control.N_P_star,
        N_UF_star=control.N_UF_star,
    )
    if target == "allocation_green":
        fixed = ControlAction.fixed(cfg)
        neutral.inflow_outflow_allocation = dict(fixed.inflow_outflow_allocation)
        neutral.green_times = dict(fixed.green_times)
    elif target == "offset":
        neutral.offsets = {signal: 0.0 for signal in net.signals}
    elif target == "vsl":
        neutral.vsl = {link: max(cfg.freeway_follower.vsl_set) for link in net.freeway_links}
    elif target == "metering":
        neutral.ramp_metering = {r: net.ramp_capacity_veh_h[r] for r in net.ramps}
    return neutral


def _replay(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
    traces: List[IntervalTrace],
    start_step: int,
    n_intervals: int,
    target: Optional[str],
) -> Dict[str, float]:
    """t0 상태에서 n_intervals 재생 — target=None이면 실제 control 재생(검증용),
    아니면 해당 control만 neutral로 고정한 counterfactual."""
    from src.simulation.coupling import run_coupled_interval

    profile = DemandProfile(cfg, scenario)
    state = traces[start_step].state_before.copy()
    total = urban = freeway = 0.0
    for offset in range(n_intervals):
        idx = start_step + offset
        if idx >= len(traces):
            break
        control = traces[idx].control
        if target is not None:
            control = _neutral_control(cfg, control, target)
        demand = profile.at(idx * cfg.simulation.control_interval)
        result = run_coupled_interval(state, control, demand, cfg)
        state.time_sec += cfg.simulation.control_interval
        total += result.freeway_ttt + result.urban_ttt
        urban += result.urban_ttt
        freeway += result.freeway_ttt
    return {"total": total, "urban": urban, "freeway": freeway}


# ---------- trigger/action/mediator 정의 (plan §4~§7) ----------

def _trigger_series(cfg: ExperimentConfig, traces: List[IntervalTrace]) -> Dict[str, List[float]]:
    """control별 trigger score 시계열 — threshold 비교로 challenged event를 정의한다."""
    net = cfg.network
    series: Dict[str, List[float]] = {c: [] for c in CONTROLS}
    for tr in traces:
        d = tr.diagnostics
        # allocation/green: 게이트 부하가 있을 때의 경계 불균형 B_in.
        load = d.get("boundary_in_load_veh", 0.0)
        series["allocation_green"].append(d.get("B_in", 0.0) if load >= 40.0 else 0.0)
        # offset: 회랑 방향성 플래툰 — 상단 회랑 양방향 점유 불균형 비율.
        state = tr.state_before
        east = sum(
            max(0.0, net.urban_link_storage_veh.get(l, 0.0) - state.urban_link_storage.get(l, 0.0))
            for l in ("A_to_B", "B_to_C")
        )
        west = sum(
            max(0.0, net.urban_link_storage_veh.get(l, 0.0) - state.urban_link_storage.get(l, 0.0))
            for l in ("B_to_A", "C_to_B")
        )
        directional = abs(east - west) / max(east + west, 1.0)
        series["offset"].append(directional if (east + west) >= 20.0 else 0.0)
        # VSL: 본선 평균 밀도비 최대값(링크 평균 — segment 초과는 exceedance로 보강).
        ratio = max(
            (sum(state.freeway_density.get(l, [0.0])) / max(len(state.freeway_density.get(l, [1])), 1)) / net.rho_crit
            for l in net.freeway_links
        )
        exceed = d.get("density_exceedance_count", 0.0)
        series["vsl"].append(max(ratio, 1.0 if exceed > 0 else 0.0))
        # metering: receiving factor 붕괴(1−factor) 또는 merge 접근 밀도비.
        receiving = d.get("mean_ramp_receiving_factor", 1.0)
        series["metering"].append(max(1.0 - receiving, ratio - 0.6))
    return series


THRESHOLDS = {
    "allocation_green": 0.03,   # eps_balance와 동일 기준의 불균형.
    "offset": 0.25,             # 방향성 25% 이상이면 progression 정렬 가치 존재.
    "vsl": 0.85,                # 밀도비 0.85 접근(활성화 임계 0.95 직전 도전 구간).
    "metering": 0.15,           # receiving 15% 붕괴 또는 밀도 근접.
}


def _action_active(cfg: ExperimentConfig, control: ControlAction, prev: Optional[ControlAction], target: str) -> bool:
    """대상 control이 neutral에서 벗어난 action을 냈는가."""
    net = cfg.network
    if target == "allocation_green":
        if not control.inflow_outflow_allocation:
            return False
        fixed = net.effective_green_total / 2.0
        return any(abs(control.green_times.get(f"{s}_p1", fixed) - fixed) > 1.0 for s in net.signals)
    if target == "offset":
        if prev is None:
            return any(abs(v) > 1.0 for v in control.offsets.values())
        return any(
            abs(control.offsets.get(s, 0.0) - prev.offsets.get(s, 0.0)) > 0.5
            for s in net.signals
        )
    if target == "vsl":
        return any(v < max(cfg.freeway_follower.vsl_set) - 0.5 for v in control.vsl.values())
    if target == "metering":
        return any(
            control.ramp_metering.get(r, net.ramp_capacity_veh_h[r]) < net.ramp_capacity_veh_h[r] - 1.0
            for r in net.ramps
        )
    return False


def _mediator_change(
    traces: List[IntervalTrace],
    t0: int,
    w_response: int,
    w_outcome: int,
    target: str,
) -> Dict[str, float]:
    """pre-window 대비 outcome-window의 mediator 변화(방향이 기대와 맞는지 판정용)."""
    pre = traces[max(0, t0 - 2): t0] or traces[t0: t0 + 1]
    out = traces[t0 + w_response: t0 + w_outcome] or traces[t0: t0 + 1]

    def mean(rows: List[IntervalTrace], key: str) -> float:
        vals = [r.diagnostics.get(key, 0.0) for r in rows]
        return sum(vals) / max(len(vals), 1)

    if target == "allocation_green":
        return {
            "delta_B_in": mean(out, "B_in") - mean(pre, "B_in"),
            "delta_departures": mean(out, "urban_total_departures_veh") - mean(pre, "urban_total_departures_veh"),
        }
    if target == "offset":
        return {"delta_urban_ttt_rate": mean(out, "urban_ttt") - mean(pre, "urban_ttt")}
    if target == "vsl":
        return {"delta_density_exceedance": mean(out, "density_exceedance_count") - mean(pre, "density_exceedance_count")}
    if target == "metering":
        return {
            "delta_receiving": mean(out, "mean_ramp_receiving_factor") - mean(pre, "mean_ramp_receiving_factor"),
            "delta_ramp_queue": mean(out, "ramp_queue_veh") - mean(pre, "ramp_queue_veh"),
        }
    return {}


EXPECTED_DIRECTION = {
    # mediator key: 기대 부호(음수=감소가 기대 방향).
    "allocation_green": ("delta_B_in", -1.0),
    "offset": ("delta_urban_ttt_rate", -1.0),
    "vsl": ("delta_density_exceedance", -1.0),
    "metering": ("delta_receiving", +1.0),
}


def detect_and_evaluate_events(
    cfg: ExperimentConfig,
    scenario: ScenarioConfig,
    traces: List[IntervalTrace],
    w_response: int = 2,
    w_outcome: int = 6,
    max_events_per_control: int = 4,
) -> List[ControlEvent]:
    """trigger 최초 초과 시점 t0마다 event를 만들고 frozen replay로 outcome을 비교한다."""
    series = _trigger_series(cfg, traces)
    events: List[ControlEvent] = []
    for control_name in CONTROLS:
        threshold = THRESHOLDS[control_name]
        values = series[control_name]
        above = False
        count = 0
        for t0, value in enumerate(values):
            if count >= max_events_per_control:
                break
            if value <= threshold:
                above = False
                continue
            if above:
                continue  # 같은 episode 내 재진입 방지 — 최초 초과만 event.
            above = True
            count += 1
            event = ControlEvent(
                control=control_name,
                event_id=f"{control_name}_{t0}",
                t0_step=t0,
                trigger_value=float(value),
                threshold=threshold,
            )
            # response delay: t0 이후 첫 active action까지의 interval 수.
            delay = None
            for k in range(0, w_response + 1):
                idx = t0 + k
                if idx >= len(traces):
                    break
                prev_ctrl = traces[idx - 1].control if idx > 0 else None
                if _action_active(cfg, traces[idx].control, prev_ctrl, control_name):
                    delay = k
                    break
            event.response_delay_intervals = delay
            if delay is None:
                event.status = ACTIVATED_BUT_INEFFECTIVE  # challenged인데 action 부재.
                events.append(event)
                continue
            # mediator 방향.
            med = _mediator_change(traces, t0, w_response, w_outcome, control_name)
            event.mediator_change = med
            key, sign = EXPECTED_DIRECTION[control_name]
            event.directional_accuracy = (med.get(key, 0.0) * sign) >= 0.0
            # counterfactual: 대상 control만 neutral로 frozen replay.
            horizon = min(w_outcome, len(traces) - t0)
            actual = _replay(cfg, scenario, traces, t0, horizon, target=None)
            counter = _replay(cfg, scenario, traces, t0, horizon, target=control_name)
            event.outcome_actual_ttt = actual["total"]
            event.outcome_counterfactual_ttt = counter["total"]
            event.outcome_gain = counter["total"] - actual["total"]  # 양수면 control이 이득.
            # congestion shift: 한 subsystem 이득이 다른 subsystem 악화로 상쇄되는지.
            urban_gain = counter["urban"] - actual["urban"]
            freeway_gain = counter["freeway"] - actual["freeway"]
            event.shifted_subsystem_delta = float(min(urban_gain, freeway_gain))
            if not event.directional_accuracy:
                event.status = WRONG_DIRECTION
            elif event.outcome_gain > 0.0 and min(urban_gain, freeway_gain) < -0.5 * event.outcome_gain:
                event.status = CONGESTION_SHIFT
            elif event.outcome_gain > 1.0e-6:
                event.status = MECHANISM_REPRODUCED
            else:
                event.status = ACTIVATED_BUT_INEFFECTIVE
            events.append(event)
        # challenged가 한 번도 없으면 정상 비활성 여부를 기록.
        if count == 0:
            inactive_ok = not any(
                _action_active(cfg, traces[i].control, traces[i - 1].control if i else None, control_name)
                for i in range(len(traces))
            )
            events.append(ControlEvent(
                control=control_name,
                event_id=f"{control_name}_none",
                t0_step=-1,
                trigger_value=max(values) if values else 0.0,
                threshold=threshold,
                status=CORRECTLY_INACTIVE if inactive_ok else NOT_CHALLENGED,
            ))
    return events


def summarize_events(events: List[ControlEvent]) -> List[Dict[str, Any]]:
    """spec 11.2 Stage 2 보고 지표 — control별 집계."""
    rows: List[Dict[str, Any]] = []
    for control_name in CONTROLS:
        evs = [e for e in events if e.control == control_name and e.t0_step >= 0]
        challenged = len(evs)
        responded = [e for e in evs if e.response_delay_intervals is not None]
        rows.append({
            "control": control_name,
            "challenged_event_count": challenged,
            "correctly_inactive_event_count": sum(
                1 for e in events if e.control == control_name and e.status == CORRECTLY_INACTIVE
            ),
            "directional_accuracy": (
                sum(1 for e in evs if e.directional_accuracy) / challenged if challenged else None
            ),
            "response_delay_mean_intervals": (
                sum(e.response_delay_intervals for e in responded) / len(responded) if responded else None
            ),
            "mechanism_success_rate": (
                sum(1 for e in evs if e.status == MECHANISM_REPRODUCED) / challenged if challenged else None
            ),
            "outcome_success_rate": (
                sum(1 for e in evs if e.outcome_gain > 0) / challenged if challenged else None
            ),
            "unnecessary_activation_rate": (
                sum(1 for e in evs if e.status == ACTIVATED_BUT_INEFFECTIVE) / challenged if challenged else None
            ),
            "congestion_shift_event_count": sum(1 for e in evs if e.status == CONGESTION_SHIFT),
            "mean_outcome_gain_veh_h": (
                sum(e.outcome_gain for e in evs) / challenged if challenged else None
            ),
        })
    return rows
