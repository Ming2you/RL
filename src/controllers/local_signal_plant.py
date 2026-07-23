# 신호 한 개의 movement 큐만 전진시키는 Wu식 국소 plant stepper (이웃 입력 동결)
"""Wu(2022) §IV-D 충실 follower의 핵심 신규 구현물 — agent i(=신호 1개)의 국소동역학 f_i.

전체망 plant(`urban_step`/`run_coupled_interval`)를 일절 호출하지 않는다. agent i의
movement 큐 q_m만 N_p×K_cu substep 전진시키며, 이웃에서 들어오는 도착(arr)·이웃 downstream
링크 가용공간(S_eff)은 결합변수로 **고정**한다. 비용은 자기 차량수 합(자기 TTS)뿐이다.

서비스/spillback 회계는 `urban_substep`(urban_queue_model.py 877~964행)을 per-signal로 복제한다.
- service_m = min(q_m, green_fraction(phase)·cap_flow_m·dt)
- 같은 receiving 링크로 가는 movement들은 그 링크의 S_eff(가용공간)를 공유(`_allocate_receiving_counts`).
- 자기 내부 링크(이 신호의 다른 movement origin)면 S_eff를 전진 중 갱신, 이웃/sink 링크면 고정.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from src.models.state import ControlAction, ExperimentConfig
from src.models.urban_queue_model import (
    _allocate_receiving_counts,
    _movement_capacity_flow,
    _phase_green_fraction,
)


@dataclass
class LocalSignalModel:
    """신호 i의 국소 rollout에 필요한, control과 무관한 정적 데이터(컨트롤러가 1회 구성).

    cap_flow는 control.inflow_outflow_allocation에 의존할 수 있으나 WU green-only 권한에서는
    allocation이 비어 있어 movement_capacity로 고정된다(여기서 미리 계산해 캐시)."""

    signal: str
    cfg: ExperimentConfig
    movements: List[str]
    specs: Dict[str, Mapping[str, object]]
    # movement -> 소속 phase id ("p1"/"p2")
    phase_of: Dict[str, str]
    # movement -> receiving 링크 키(없으면 "")
    receiving_of: Dict[str, str]
    # movement -> origin 링크 키(점큐가 점유하는 링크; 자기 내부 S_eff 갱신용)
    origin_of: Dict[str, str]
    # movement -> beta(phase arr 분배 가중)
    beta_of: Dict[str, float]
    # movement -> cap_flow[veh/h] (control 무관 캐시)
    cap_flow_of: Dict[str, float]
    receiving_space_rule: str
    # ----- ramp-aware 확장(D/F 같은 freeway-인접 신호용; ramp 없는 신호는 빈 dict) -----
    # movement -> kind ("on_ramp"/"off_ramp"/"boundary_in"/"internal"/"boundary_out")
    kind_of: Dict[str, str]
    # off_ramp 이름 -> [그 off_ramp에서 방출되는 off_ramp movement 목록] (이 신호 권역)
    offramp_movements: Dict[str, List[str]]
    # off_ramp 이름 -> storage 링크 cap[veh]
    offramp_storage_cap: Dict[str, float]
    # ramp(on-ramp) 이름 -> [그 ramp로 release하는 on_ramp movement 목록] (이 신호 권역)
    onramp_movements: Dict[str, List[str]]
    # on-ramp reservoir 큐 상한[veh]
    ramp_queue_max: float
    has_ramps: bool = False


def build_local_model(
    cfg: ExperimentConfig,
    signal: str,
    specs: Mapping[str, Mapping[str, object]],
    phase_movements: Mapping[str, Mapping[str, List[str]]],
) -> LocalSignalModel:
    movements: List[str] = []
    phase_of: Dict[str, str] = {}
    receiving_of: Dict[str, str] = {}
    origin_of: Dict[str, str] = {}
    beta_of: Dict[str, float] = {}
    cap_flow_of: Dict[str, float] = {}
    kind_of: Dict[str, str] = {}
    offramp_movements: Dict[str, List[str]] = {}
    offramp_storage_cap: Dict[str, float] = {}
    onramp_movements: Dict[str, List[str]] = {}
    net = cfg.network
    # WU green-only 권한에서 allocation이 비어 cap_flow가 control 무관이므로 빈 control로 캐시한다.
    neutral_control = ControlAction.uncontrolled(cfg)
    neutral_control.inflow_outflow_allocation = {}
    for pid in ("p1", "p2"):
        for movement in phase_movements[signal][pid]:
            spec = specs[movement]
            movements.append(movement)
            phase_of[movement] = pid
            receiving_of[movement] = str(spec.get("receiving_link", ""))
            origin_of[movement] = str(spec.get("origin", ""))
            beta_of[movement] = float(spec.get("beta", 0.0))
            cap_flow_of[movement] = float(_movement_capacity_flow(neutral_control, cfg, movement, spec))
            kind = str(spec.get("kind", ""))
            kind_of[movement] = kind
            # 이 신호 권역에 속한 off-ramp / on-ramp 인터페이스를 movement로부터 귀속한다.
            # off-ramp: kind=="off_ramp"(storage에서 방출). on-ramp(reservoir 적재): 실제 plant
            # `ramp_requests`는 spec["ramp"]가 있는 **모든** movement를 처리하므로(예: ramp행
            # boundary_in `D_W_to_on*`도 포함) kind가 아니라 ramp 태그로 귀속한다.
            ramp_tag = str(spec.get("ramp", ""))
            if kind == "off_ramp":
                off_ramp = str(spec.get("off_ramp", ""))
                if off_ramp:
                    offramp_movements.setdefault(off_ramp, []).append(movement)
                    storage = net.off_ramp_storage_link.get(off_ramp, "")
                    offramp_storage_cap[off_ramp] = float(
                        net.urban_link_storage_veh.get(storage, 0.0)
                    )
            elif ramp_tag:
                onramp_movements.setdefault(ramp_tag, []).append(movement)
    return LocalSignalModel(
        signal=signal,
        cfg=cfg,
        movements=movements,
        specs={m: dict(specs[m]) for m in movements},
        phase_of=phase_of,
        receiving_of=receiving_of,
        origin_of=origin_of,
        beta_of=beta_of,
        cap_flow_of=cap_flow_of,
        receiving_space_rule=str(cfg.urban_follower.receiving_space_rule),
        kind_of=kind_of,
        offramp_movements=offramp_movements,
        offramp_storage_cap=offramp_storage_cap,
        onramp_movements=onramp_movements,
        ramp_queue_max=float(net.ramp_queue_max_veh),
        has_ramps=bool(offramp_movements or onramp_movements),
    )


def rollout_local_tts(
    model: LocalSignalModel,
    q0: Mapping[str, float],
    arr_movement: Mapping[str, float],
    s_eff0: Mapping[str, float],
    green_p1: float,
    green_p2: float,
    substeps: int,
    dt_h: float,
    s_eff_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
) -> float:
    """신호 i의 movement 큐만 `substeps`회 전진시키고 누적 자기 차량수(자기 TTS proxy)를 반환.

    - q: 자기 movement 큐(국소 복사). arr_movement: movement별 고정 도착유량[veh/h] —
      호출처(follower)가 각 movement의 실제 소스(게이트 demand·β, on/off-ramp·β,
      상류 신호 leaving rate의 origin-link 분배)로 직접 계산해 넘긴다(phase 집계 재분배 금지).
    - s_eff: receiving 링크 가용공간. 자기 origin 링크(이 신호가 점유)는 전진 중 갱신,
      이웃/sink 링크는 s_eff0 고정값을 유지(이웃 downstream 동결).
    - 비용 = Σ_substep (Σ_m q_m) · dt  (제어변화 penalty는 호출처에서 더함).
    이게 검증 포인트 1: B·C·D·F·freeway는 절대 전진하지 않는다."""
    net = model.cfg.network
    cycle = max(net.cycle_length, 1.0e-9)
    green = {"p1": float(green_p1), "p2": float(green_p2)}
    q: Dict[str, float] = {m: max(0.0, float(q0.get(m, 0.0))) for m in model.movements}
    # 자기 origin 링크(이 신호의 movement가 점유하는 링크) 집합 — 이 링크들만 전진 중 S_eff 갱신.
    own_origin_links = {model.origin_of[m] for m in model.movements if model.origin_of[m]}
    s_eff: Dict[str, float] = {link: max(0.0, float(v)) for link, v in s_eff0.items()}

    cost = 0.0
    for sub in range(substeps):
        # 0) (오라클 probe) downstream S_eff 시변 궤적 주입 — s_eff_by_substep이 주어지면 매
        #    substep 시작(도착 주입·서비스 이전)에 각 recv 링크 S_eff를 oracle 값으로 리셋한다.
        #    직전 substep의 ego 방출로 인한 내부 감소는 여기서 무효화된다("follower가 진짜
        #    downstream을 본다"). 키가 없는 링크는 기존 내부값 유지. None이면 기존 거동과 비트 동일.
        if s_eff_by_substep is not None:
            for recv, prof in s_eff_by_substep.items():
                if recv in s_eff:
                    s_eff[recv] = max(0.0, float(prof[sub]))
        # 1) 고정 도착 주입.
        for m in model.movements:
            q[m] += arr_movement.get(m, 0.0) * dt_h
        # 2) receiving 링크별로 intended service를 모으고 S_eff로 게이트(spillback 복제).
        intended_by_link: Dict[str, Dict[str, float]] = {}
        no_link_intended: Dict[str, float] = {}
        for m in model.movements:
            pid = model.phase_of[m]
            green_fraction = green[pid] / cycle
            intended = min(q[m], dt_h * green_fraction * model.cap_flow_of[m])
            if intended < 0.0:
                intended = 0.0
            recv = model.receiving_of[m]
            if recv and recv in s_eff:
                intended_by_link.setdefault(recv, {})[m] = intended
            else:
                no_link_intended[m] = intended
        actual: Dict[str, float] = dict(no_link_intended)
        for link, intended in intended_by_link.items():
            available_space = s_eff.get(link, 0.0)
            actual.update(_allocate_receiving_counts(
                model.receiving_space_rule,
                intended,
                available_space,
            ))
        # 3) 큐 차감 + 자기 내부 링크 S_eff 점유 갱신(이웃/sink 링크는 고정 유지).
        for m in model.movements:
            departed = min(q[m], max(0.0, actual.get(m, 0.0)))
            if departed <= 0.0:
                continue
            q[m] -= departed
            recv = model.receiving_of[m]
            if recv in own_origin_links and recv in s_eff:
                # 자기 신호의 다른 movement가 origin으로 쓰는 링크면 그 링크 점유가 늘어 S_eff↓.
                s_eff[recv] = max(0.0, s_eff[recv] - departed)
        # 4) 누적 비용 = Σ 자기 차량수 · dt (자기 TTS).
        cost += sum(q.values()) * dt_h
    return float(cost)


def rollout_local_tts_phased(
    model: LocalSignalModel,
    q0: Mapping[str, float],
    arr_by_substep: Mapping[str, List[float]],
    gf_by_substep: Mapping[str, List[float]],
    s_eff0: Mapping[str, float],
    substeps: int,
    dt_h: float,
    s_eff_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
) -> float:
    """PHASE-RESOLVED + PLATOON-AWARE 국소 rollout — 자기 TTS proxy 누적 반환(non-ramp 신호용).

    `rollout_local_tts`(cycle-평균 green/cycle + 균일 도착)의 충실도 상향판. 두 가지만 다르다:
    1. 서비스 = `gf_by_substep[m][sub]`(offset-aware `_phase_green_fraction(urban_step_index=sub)`)를
       cap_flow에 곱한다. 즉 substep별 green window 겹침으로 서비스 → offset이 동역학에 들어간다.
    2. 도착 = `arr_by_substep[m][sub]`(상류 신호 green+offset discharge를 τ 지연해 재구성한
       시간분해 platoon profile, phase 예산 보존). 균일 상수 도착이 아니다.
    나머지 spillback/receiving S_eff/큐 차감 회계는 `rollout_local_tts`와 동일(깨지면 안 됨).
    호출처(follower)가 gf·arr 프로파일을 미리 계산해 넘긴다(여기는 순수 큐 전진만)."""
    q: Dict[str, float] = {m: max(0.0, float(q0.get(m, 0.0))) for m in model.movements}
    own_origin_links = {model.origin_of[m] for m in model.movements if model.origin_of[m]}
    s_eff: Dict[str, float] = {link: max(0.0, float(v)) for link, v in s_eff0.items()}

    cost = 0.0
    for sub in range(substeps):
        # 0) (오라클 probe) downstream S_eff 시변 궤적 주입 — 매 substep 시작에 oracle로 리셋.
        if s_eff_by_substep is not None:
            for recv, prof in s_eff_by_substep.items():
                if recv in s_eff:
                    s_eff[recv] = max(0.0, float(prof[sub]))
        # 1) platoon 도착 주입(시간분해, τ 지연·예산 보존).
        for m in model.movements:
            prof = arr_by_substep.get(m)
            if prof:
                q[m] += prof[sub] * dt_h
        # 2) receiving 링크별 intended service(offset-aware green fraction) → S_eff 게이트.
        intended_by_link: Dict[str, Dict[str, float]] = {}
        no_link_intended: Dict[str, float] = {}
        for m in model.movements:
            gf = gf_by_substep[m][sub]
            intended = min(q[m], dt_h * gf * model.cap_flow_of[m])
            if intended < 0.0:
                intended = 0.0
            recv = model.receiving_of[m]
            if recv and recv in s_eff:
                intended_by_link.setdefault(recv, {})[m] = intended
            else:
                no_link_intended[m] = intended
        actual: Dict[str, float] = dict(no_link_intended)
        for link, intended in intended_by_link.items():
            actual.update(_allocate_receiving_counts(
                model.receiving_space_rule, intended, s_eff.get(link, 0.0),
            ))
        # 3) 큐 차감 + 자기 내부 링크 S_eff 점유 갱신(이웃/sink 링크 고정).
        for m in model.movements:
            departed = min(q[m], max(0.0, actual.get(m, 0.0)))
            if departed <= 0.0:
                continue
            q[m] -= departed
            recv = model.receiving_of[m]
            if recv in own_origin_links and recv in s_eff:
                s_eff[recv] = max(0.0, s_eff[recv] - departed)
        cost += sum(q.values()) * dt_h
    return float(cost)


def rollout_local_tts_ramp_aware(
    model: LocalSignalModel,
    q0: Mapping[str, float],
    arr_movement: Mapping[str, float],
    s_eff0: Mapping[str, float],
    offramp_inflow: Mapping[str, float],
    offramp_occ0: Mapping[str, float],
    ramp_queue0: Mapping[str, float],
    reservoir_drain: Mapping[str, float],
    freeway_congestion: Mapping[str, float],
    ramp_metering_weight: float,
    green_p1: float,
    green_p2: float,
    substeps: int,
    dt_h: float,
    arr_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    gf_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
    s_eff_by_substep: Optional[Mapping[str, Sequence[float]]] = None,
) -> float:
    """freeway-인접 신호(D/F)의 ramp-aware 국소 rollout — 자기 own-TTS 누적 반환.

    SPEC: off-ramp storage·on-ramp reservoir를 urban agent에 귀속(Wu §IV-A). 국소
    rollout이 이 ramp 인터페이스를 **직접 전진**시키고 freeway 본선(ρ,v)만 동결한다.
    substep마다 실제 plant(`urban_substep`/`_drain_offramp_storage`) 회계 순서를 복제한다:
      (a) on_ramp 적재(ramp_requests 복제): green·cap·reservoir 여유로 movement 큐→reservoir.
      (b) off_ramp drain(`_drain_offramp_storage` 복제): green·cap·하류 S_eff로 storage→urban.
      (c) off-ramp storage 유입(동결): freeway→off-ramp 유출 = frozen offramp_inflow.
      (d) 나머지 urban movement(internal/boundary_in/boundary_out): green·cap·S_eff 게이트.
    own-TTS = Σ_substep ( Σ urban movement 큐 + Σ off-ramp storage 점유 + Σ reservoir 큐 )·dt.

    인자:
    - offramp_inflow: off_ramp 이름 -> freeway→off-ramp 유출[veh/h](frozen 결합값).
    - offramp_occ0: off_ramp 이름 -> 초기 storage 점유[veh](occupancy = cap − available).
    - ramp_queue0: ramp 이름 -> 초기 reservoir 큐[veh].
    - reservoir_drain: ramp 이름 -> freeway가 reservoir를 비우는 frozen 방출률[veh/h]
      (`compute_ramp_release_flows`). 이게 없으면 reservoir가 가득 차 green이 무력해진다.
    - freeway_congestion: ramp 이름 -> frozen 혼잡 가중 w_fw∈[0,1](1−receiving_factor).
    - ramp_metering_weight: de facto ramp metering 패널티 계수. 막힌 freeway로 보낸
      reservoir 적재(=freeway 유입)에 w_fw·계수를 곱해 비용에 더한다(SPEC line 28).
    이게 V1: D/F는 off-ramp storage·on-ramp reservoir를 전진시키되 freeway 전체 호출 없음.

    PHASE-RESOLVED 확장(P1.5, 선택 인자 — None이면 기존 cycle-평균 거동과 완전 동일):
    - arr_by_substep: queue movement별 시간분해 도착[veh/h] (platoon). 제공되면 프로파일이
      있는 movement는 균일 arr_movement 대신 arr_by_substep[m][sub]로 도착 주입.
    - gf_by_substep: movement별 offset-aware green fraction. 제공되면 green[pid]/cycle 대신
      gf_by_substep[m][sub]을 모든 서비스 지점 — (a) on-ramp request, (b) off-ramp drain
      intended, (d) 일반 movement discharge — 에서 사용한다(비-ramp `rollout_local_tts_phased`
      와 동일 철학; reservoir/storage/metering-pricing·s_eff 회계는 불변).
    """
    net = model.cfg.network
    cycle = max(net.cycle_length, 1.0e-9)
    green = {"p1": float(green_p1), "p2": float(green_p2)}

    def _gf(m: str, sub: int) -> float:
        # phase-resolved green fraction(제공 시) — 아니면 기존 cycle-평균.
        if gf_by_substep is not None:
            prof = gf_by_substep.get(m)
            if prof is not None:
                return float(prof[sub])
        return green[model.phase_of[m]] / cycle
    # ramp(reservoir)행 movement 집합 — step (a)에서 처리, step (d)에서 제외.
    ramp_movement_set = {m for mv in model.onramp_movements.values() for m in mv}
    # off_ramp movement는 큐가 아니라 storage에서 방출되므로 movement-큐 전진 대상에서 제외.
    queue_movements = [m for m in model.movements if model.kind_of[m] != "off_ramp"]
    q: Dict[str, float] = {m: max(0.0, float(q0.get(m, 0.0))) for m in queue_movements}
    # off-ramp storage 점유(veh) — 자기 권역 off_ramp만.
    occ: Dict[str, float] = {
        orr: max(0.0, float(offramp_occ0.get(orr, 0.0))) for orr in model.offramp_movements
    }
    # on-ramp reservoir 큐(veh) — 자기 권역 ramp만.
    res: Dict[str, float] = {
        ramp: max(0.0, float(ramp_queue0.get(ramp, 0.0))) for ramp in model.onramp_movements
    }
    own_origin_links = {model.origin_of[m] for m in queue_movements if model.origin_of[m]}
    s_eff: Dict[str, float] = {link: max(0.0, float(v)) for link, v in s_eff0.items()}

    cost = 0.0
    for sub in range(substeps):
        # (오라클 probe) downstream S_eff 시변 궤적 주입 — 매 substep 시작(도착·서비스 이전)에
        # oracle로 리셋. off-ramp drain (b)·일반 movement (d)가 읽는 s_eff가 시변 downstream을
        # 본다. None이면 기존 거동과 비트 동일.
        if s_eff_by_substep is not None:
            for recv, prof in s_eff_by_substep.items():
                if recv in s_eff:
                    s_eff[recv] = max(0.0, float(prof[sub]))
        # 0) 도착 주입(off_ramp 제외: storage 유입은 (c)에서 별도 처리). platoon 프로파일이
        #    제공된 movement는 시간분해 도착, 아니면 기존 균일 도착.
        for m in queue_movements:
            if arr_by_substep is not None:
                prof = arr_by_substep.get(m)
                if prof is not None:
                    q[m] += float(prof[sub]) * dt_h
                    continue
            q[m] += arr_movement.get(m, 0.0) * dt_h

        # (a0) reservoir 배출(freeway pull, frozen): w_r -= drain·dt. 이게 있어야 on_ramp
        #      green이 reservoir 여유를 만들 수 있고, 가득 찬 reservoir로의 적재가 거부돼
        #      p1(on_ramp 위주) 과잉 green이 낭비로 비용에 잡힌다.
        for ramp in model.onramp_movements:
            res[ramp] = max(0.0, res.get(ramp, 0.0) - reservoir_drain.get(ramp, 0.0) * dt_h)

        # (a) on_ramp 적재 — ramp_requests 복제: green·cap·reservoir 여유로 큐→reservoir.
        for ramp, movements in model.onramp_movements.items():
            requests: Dict[str, float] = {}
            for m in movements:
                green_fraction = _gf(m, sub)
                requests[m] = min(
                    max(0.0, q[m]), dt_h * green_fraction * model.cap_flow_of[m]
                )
            requested_total = sum(requests.values())
            ramp_space = max(0.0, model.ramp_queue_max - res.get(ramp, 0.0))
            scale = 1.0 if requested_total <= ramp_space else ramp_space / max(requested_total, 1.0e-9)
            released_total = 0.0
            for m, requested in requests.items():
                actual = min(max(0.0, q[m]), requested * scale)
                q[m] = max(0.0, q[m] - actual)
                released_total += actual
            res[ramp] = min(model.ramp_queue_max, res.get(ramp, 0.0) + released_total)
            # de facto ramp metering(SPEC line 28): 막힌 freeway로 향하는 reservoir 적재를
            # freeway 혼잡 w_fw로 가중해 비용에 더한다. drain≈released이면 이 적재분이
            # 그대로 freeway로 흘러 혼잡을 키우므로, p1(on_ramp 위주) green이 억제된다.
            cost += (
                released_total * freeway_congestion.get(ramp, 0.0) * ramp_metering_weight
            )

        # (b) off_ramp drain — `_drain_offramp_storage` 복제: green·cap·하류 S_eff로 storage→urban.
        for off_ramp, movements in model.offramp_movements.items():
            occupancy = occ.get(off_ramp, 0.0)
            if occupancy <= 0.0:
                continue
            released_total = 0.0
            for m in movements:
                beta = model.beta_of[m]
                if beta <= 0.0:
                    continue
                green_fraction = _gf(m, sub)
                intended = min(beta * occupancy, dt_h * green_fraction * model.cap_flow_of[m])
                recv = model.receiving_of[m]
                if recv and recv in s_eff:
                    actual = min(intended, s_eff[recv])
                else:
                    actual = intended
                if actual <= 0.0:
                    continue
                released_total += actual
                # 하류 링크 점유 갱신(자기 origin 링크면 점유↑, 즉 S_eff↓).
                if recv in own_origin_links and recv in s_eff:
                    s_eff[recv] = max(0.0, s_eff[recv] - actual)
            occ[off_ramp] = max(0.0, occupancy - released_total)

        # (c) off-ramp storage 유입(동결): freeway→off-ramp 유출, storage cap으로 clamp.
        for off_ramp in model.offramp_movements:
            inflow = max(0.0, float(offramp_inflow.get(off_ramp, 0.0))) * dt_h
            cap = model.offramp_storage_cap.get(off_ramp, 0.0)
            occ[off_ramp] = min(cap, occ.get(off_ramp, 0.0) + inflow)

        # (d) 나머지 urban movement(internal/boundary_in/boundary_out) — green·cap·S_eff 게이트.
        intended_by_link: Dict[str, Dict[str, float]] = {}
        no_link_intended: Dict[str, float] = {}
        for m in queue_movements:
            if m in ramp_movement_set:
                continue  # ramp행 movement는 (a)에서 처리됨.
            green_fraction = _gf(m, sub)
            intended = min(max(0.0, q[m]), dt_h * green_fraction * model.cap_flow_of[m])
            recv = model.receiving_of[m]
            if recv and recv in s_eff:
                intended_by_link.setdefault(recv, {})[m] = intended
            else:
                no_link_intended[m] = intended
        actual: Dict[str, float] = dict(no_link_intended)
        for link, intended in intended_by_link.items():
            actual.update(_allocate_receiving_counts(
                model.receiving_space_rule, intended, s_eff.get(link, 0.0),
            ))
        for m in queue_movements:
            if m in ramp_movement_set:
                continue
            departed = min(q[m], max(0.0, actual.get(m, 0.0)))
            if departed <= 0.0:
                continue
            q[m] -= departed
            recv = model.receiving_of[m]
            if recv in own_origin_links and recv in s_eff:
                s_eff[recv] = max(0.0, s_eff[recv] - departed)

        # (e) own-TTS = Σ(movement 큐 + off-ramp storage 점유 + reservoir 큐)·dt.
        cost += (sum(q.values()) + sum(occ.values()) + sum(res.values())) * dt_h
        # (f) VdB 시나리오4 재현(2026-07-19, 기본 OFF): 보호 큐 최대길이 제약의 소프트 벌점.
        # 이 신호가 보호 movement를 소유하면 초과분(q−q_max)+에 가중 w를 곱해 비용에 가산 —
        # 예측형 컨트롤러가 제약을 rollout 안에서 미리 보게 한다(반응형 집행과 대조 실험용).
        _pq_mv = str(getattr(model.cfg.mpc, "protected_queue_movement", "") or "")
        if _pq_mv and _pq_mv in q:
            _pq_w = float(getattr(model.cfg.mpc, "protected_queue_weight", 0.0))
            if _pq_w > 0.0:
                _pq_max = float(getattr(model.cfg.mpc, "protected_queue_max_veh", 50.0))
                cost += _pq_w * max(0.0, q[_pq_mv] - _pq_max) * dt_h
    return float(cost)
