# freeway link 한 개의 METANET(ρ,v)·on-ramp·off-ramp만 전진시키는 Wu식 국소 plant stepper (이웃 본선 동결)
"""Wu(2022) §IV-D 충실 follower의 freeway agent용 핵심 신규 구현물 — agent(=freeway link 1개)의 국소동역학.

전체망 freeway plant(`metanet.freeway_substep`)를 일절 호출하지 않는다. `freeway_substep`은
매 호출마다 `for link in net.freeway_links`로 **모든** freeway link을 전진시키므로, 후보 1개를
채점하려면 전체 freeway가 돌아 O(n)이 아니다. 여기서는 `freeway_substep`의 per-link METANET
갱신(밀도식 6, 속도식 7, merge/diverge, capacity-drop λ_eff)을 **단일 link**에 대해서만 복제한다.

이웃 본선 경계 동결:
- `freeway_substep`의 본선 갱신은 애초에 link 간 결합이 없다. segment 0의 upstream 속도는
  상수 `net.v_free`, 마지막 segment의 downstream 밀도는 자유출구 표준 경계
  `min(rho_for_flow[i], rho_crit)`이고 터미널 배출은 용량 cap(2026-07-18 경계 수정, plant와
  동일). 즉 plant 자체가 link을 독립적으로 전진시킨다.
  따라서 단일-link 복제는 별도의 이웃 동결 조작 없이 plant와 비트 동일한 본선 거동을 낸다
  (upstream=v_free·downstream 경계 규약을 plant와 그대로 공유한다 = de facto 동결).
- 후보 채점 동안 이 link의 segment 밀도/속도만 갱신되고 다른 link 상태는 절대 건드리지 않는다.

on-ramp/off-ramp:
- on-ramp 합류는 merge segment에 release[veh/h]를 주입(`compute_ramp_release_flows` 결과를
  follower가 후보 metering으로 계산해 넘긴다). off-ramp 유출은 off-ramp segment의 split·cap로
  본선에서 빠진다. capacity-drop은 off-ramp storage 점유 기반 λ_eff로 merge/off-ramp segment에 적용.

own-TTS = Σ_substep (자기 link segment 차량 + 자기 on-ramp queue + 자기 off-ramp storage 점유)·dt.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

from src.models.metanet import (
    _clip,
    _configured_segment_index,
    _ramp_merge_index,
    effective_desired_speed_kmh,
    effective_rho_crit,
    metanet_speed_update_kmh,
    offramp_spillback_lambda_eff,
    segment_flow_veh_h,
    select_anticipation_nu,
)
from src.models.state import ControlAction, ExperimentConfig, TrafficState, segment_vsl


@dataclass
class LocalFreewayModel:
    """freeway link 1개의 국소 rollout에 필요한, control과 무관한 정적 데이터(컨트롤러가 1회 구성)."""

    link: str
    cfg: ExperimentConfig
    n_seg: int
    # 이 link이 소유하는 on-ramp 목록(ramp_to_freeway[ramp]==link)과 merge segment index.
    owned_ramps: List[str]
    ramp_merge_idx: Dict[str, int]
    # 이 link에서 갈라지는 off-ramp 목록과 off-ramp segment index.
    owned_offramps: List[str]
    offramp_seg_idx: Dict[str, int]
    # off_ramp -> split ratio
    offramp_split: Dict[str, float]
    # off_ramp -> storage 링크 cap[veh]
    offramp_storage_cap: Dict[str, float]
    # segment index -> 그 segment에서 빠지는 off-ramp 목록
    offramps_by_segment: Dict[int, List[str]]


def build_local_freeway_model(cfg: ExperimentConfig, link: str) -> LocalFreewayModel:
    net = cfg.network
    n_seg = net.freeway_segments_per_link
    owned_ramps = [r for r in net.ramps if net.ramp_to_freeway.get(r) == link]
    ramp_merge_idx = {r: _ramp_merge_index(cfg, r, n_seg) for r in owned_ramps}
    owned_offramps = [o for o in net.off_ramps if net.off_ramp_from_freeway.get(o) == link]
    offramp_seg_idx: Dict[str, int] = {}
    offramps_by_segment: Dict[int, List[str]] = {}
    offramp_storage_cap: Dict[str, float] = {}
    offramp_split: Dict[str, float] = {}
    for off_ramp in owned_offramps:
        seg = _configured_segment_index(
            getattr(net, "off_ramp_segment_index", {}),
            off_ramp,
            n_seg - 1,
            n_seg,
        )
        offramp_seg_idx[off_ramp] = seg
        offramps_by_segment.setdefault(seg, []).append(off_ramp)
        offramp_split[off_ramp] = _clip(float(net.off_ramp_split_ratio.get(off_ramp, 0.0)), 0.0, 1.0)
        storage = net.off_ramp_storage_link.get(off_ramp, "")
        offramp_storage_cap[off_ramp] = float(net.urban_link_storage_veh.get(storage, 0.0))
    return LocalFreewayModel(
        link=link,
        cfg=cfg,
        n_seg=n_seg,
        owned_ramps=owned_ramps,
        ramp_merge_idx=ramp_merge_idx,
        owned_offramps=owned_offramps,
        offramp_seg_idx=offramp_seg_idx,
        offramp_split=offramp_split,
        offramp_storage_cap=offramp_storage_cap,
        offramps_by_segment=offramps_by_segment,
    )


def _local_lane_profile(
    model: LocalFreewayModel,
    occupancy: Mapping[str, float],
    demand,
) -> List[float]:
    """이 link segment별 effective lane profile — `metanet.effective_lane_profile`의 단일-link 복제.

    off-ramp storage 점유(occupancy[veh])로 off-ramp segment의 λ_eff를 줄이고(capacity drop),
    incident lane loss(demand.freeway_lane_loss[link])가 있으면 더 작은 값을 적용한다.
    """
    cfg = model.cfg
    net = cfg.network
    drop = cfg.freeway_offramp_capacity_drop
    lanes = [float(net.freeway_lanes) for _ in range(model.n_seg)]
    if drop.enabled:
        for off_ramp in model.owned_offramps:
            cap = model.offramp_storage_cap.get(off_ramp, 0.0)
            occ = _clip(float(occupancy.get(off_ramp, 0.0)), 0.0, cap)
            lambda_eff = offramp_spillback_lambda_eff(
                occ, cap, float(net.freeway_lanes),
                float(drop.lane_reduction), float(drop.gamma), float(drop.b),
            )
            seg = model.offramp_seg_idx[off_ramp]
            lanes[seg] = min(lanes[seg], lambda_eff)
    lane_losses = getattr(demand, "freeway_lane_loss", {}) if demand is not None else {}
    seg_losses = lane_losses.get(model.link, {}) if isinstance(lane_losses, dict) else {}
    for segment, loss in seg_losses.items():
        seg = int(segment)
        if not 0 <= seg < model.n_seg:
            continue
        incident_lanes = max(1.0e-9, float(net.freeway_lanes) - max(0.0, float(loss)))
        lanes[seg] = min(lanes[seg], incident_lanes)
    return lanes


def freeway_substep_local(
    model: LocalFreewayModel,
    rhos: List[float],
    speeds: List[float],
    prev_lanes: List[float],
    occupancy: Mapping[str, float],
    mainline_origin_queue: float,
    ramp_release: Mapping[str, float],
    offramp_capacity: Mapping[str, float],
    control: ControlAction,
    demand,
    buffer_bc=None,
) -> Tuple[List[float], List[float], List[float], float, Dict[str, float], List[float]]:
    """단일 link METANET 한 substep 전진 — `metanet.freeway_substep`의 이 link 루프 바디를 복제.

    반환 (next_rhos, next_speeds, next_lanes, mainline_origin_queue, offramp_flow_by_offramp, vehicle_count).
    off-ramp storage 점유 갱신은 호출처(stepper)가 offramp_flow와 drain으로 처리한다.
    """
    cfg = model.cfg
    net = cfg.network
    sim = cfg.simulation
    dt_h = sim.T_f_h
    link = model.link
    cap_factor = getattr(demand, "incident_capacity_factor", 1.0)
    q_cap = net.freeway_capacity_veh_h * cap_factor

    lanes_now = _local_lane_profile(model, occupancy, demand)

    # ramp 유입을 merge segment에 모은다.
    ramp_in_by_segment = [0.0 for _ in range(model.n_seg)]
    for ramp in model.owned_ramps:
        ramp_in_by_segment[model.ramp_merge_idx[ramp]] += max(0.0, float(ramp_release.get(ramp, 0.0)))

    # 차량수는 previous lanes로, 밀도는 lanes_now로(metanet.py:326-333 복제).
    vehicles = [
        max(0.0, rho) * net.freeway_segment_length_km * max(lane, 1.0e-9)
        for rho, lane in zip(rhos, prev_lanes)
    ]
    rho_for_flow = [
        n / max(net.freeway_segment_length_km * max(lane, 1.0e-9), 1.0e-9)
        for n, lane in zip(vehicles, lanes_now)
    ]
    q_values = [
        segment_flow_veh_h(rho, speed, lane)
        for rho, speed, lane in zip(rho_for_flow, speeds, lanes_now)
    ]
    # plant(metanet.py)의 queue-discharge capacity drop 복제 — 예측-플랜트 정합 필수.
    phi_cd = float(getattr(net, "capacity_drop_discharge_phi", 1.0) or 1.0)
    if phi_cd < 1.0:
        for _i in range(model.n_seg):
            if rho_for_flow[_i] > effective_rho_crit(net, segment_vsl(control, link, _i, cfg)):
                q_values[_i] = min(
                    q_values[_i],
                    phi_cd * q_cap * max(lanes_now[_i], 1.0e-9) / max(float(net.freeway_lanes), 1.0e-9),
                )
    receiving = [
        max(
            0.0,
            (net.rho_max - rho_for_flow[i]) * net.freeway_segment_length_km
            * max(lanes_now[i], 1.0e-9) / max(dt_h, 1.0e-9),
        )
        for i in range(model.n_seg)
    ]
    receiving_for_mainline = [
        max(0.0, receiving[i] - max(0.0, ramp_in_by_segment[i]))
        for i in range(model.n_seg)
    ]
    off_ratio_by_segment = [
        _clip(
            sum(model.offramp_split.get(o, 0.0) for o in model.offramps_by_segment.get(i, [])),
            0.0, 1.0,
        )
        for i in range(model.n_seg)
    ]
    mainline_sending = [
        (1.0 - off_ratio_by_segment[i]) * q_values[i] for i in range(model.n_seg)
    ]
    q_inter = [
        min(mainline_sending[i], receiving_for_mainline[i + 1]) for i in range(model.n_seg - 1)
    ]
    # 진입 경계.
    mainline_demand = max(0.0, demand.freeway_mainline.get(link, 0.0)) if demand is not None else 0.0
    queued_flow = mainline_origin_queue / max(dt_h, 1.0e-9)
    entry_request = mainline_demand + queued_flow
    if buffer_bc is not None:
        # 완충 plant 동결 결합(Phase B): 코어 유입 = 상류 완충 끝 sending(동결) ∧ 코어 수용.
        _bu_send_bc, _bu_speed_bc, _bd_rho0_bc = buffer_bc
        entry_realized = min(_bu_send_bc, receiving_for_mainline[0])
    else:
        entry_realized = min(entry_request, q_cap, receiving_for_mainline[0])
    next_origin_queue = max(0.0, mainline_origin_queue + dt_h * (mainline_demand - entry_realized))

    offramp_flow: Dict[str, float] = {o: 0.0 for o in model.owned_offramps}
    next_rhos: List[float] = []
    next_speeds: List[float] = []
    next_lanes: List[float] = []
    next_vehicle_count: List[float] = []
    for i, rho in enumerate(rho_for_flow):
        q_in = entry_realized if i == 0 else q_inter[i - 1]
        q_in += ramp_in_by_segment[i]
        terminal_out = None
        if i == model.n_seg - 1:
            if buffer_bc is not None:
                # 완충 동결 결합: 하류 완충 머리 수용량(동결 밀도 기반)으로 배출 제한.
                terminal_out = min(
                    mainline_sending[i],
                    max(0.0, (net.rho_max - _bd_rho0_bc) * net.freeway_segment_length_km
                        * max(lanes_now[i], 1.0e-9) / max(dt_h, 1.0e-9)),
                )
            elif getattr(net, "terminal_zero_gradient", False):
                # 구경계 복원 모드(plant와 동일): sending 그대로 배출(cap 없음).
                terminal_out = mainline_sending[i]
            else:
                # plant(metanet.py freeway_substep)와 동일한 자유출구 CTM supply cap.
                terminal_cap = q_cap * max(lanes_now[i], 1.0e-9) / max(float(net.freeway_lanes), 1.0e-9)
                terminal_out = min(mainline_sending[i], terminal_cap)
            q_out = terminal_out
        else:
            q_out = q_inter[i]
        boundary_speed_cap = None
        effective_off_total = 0.0
        normal_off_total = 0.0
        for off_ramp in model.offramps_by_segment.get(i, []):
            ratio = model.offramp_split.get(off_ramp, 0.0)
            normal_off = ratio * q_values[i]
            cap = offramp_capacity.get(off_ramp, offramp_capacity.get(link)) if offramp_capacity else None
            effective_off = normal_off if cap is None else min(normal_off, max(0.0, cap))
            effective_off_total += effective_off
            normal_off_total += normal_off
            offramp_flow[off_ramp] = offramp_flow.get(off_ramp, 0.0) + effective_off
        if normal_off_total > 0.0:
            q_out += effective_off_total
            if normal_off_total > effective_off_total + 1.0e-9:
                boundary_speed_cap = q_out / max(rho * lanes_now[i], 1.0e-9)
        if terminal_out is not None and terminal_out < mainline_sending[i] - 1.0e-9:
            # plant와 동일: 터미널 배출이 용량에 걸리면 속도도 flow-정합으로 cap.
            exit_speed_cap = q_out / max(rho * lanes_now[i], 1.0e-9)
            boundary_speed_cap = (
                exit_speed_cap if boundary_speed_cap is None
                else min(boundary_speed_cap, exit_speed_cap)
            )
        vehicle_raw = vehicles[i] + dt_h * (q_in - q_out)
        vehicle_new = max(0.0, vehicle_raw)
        rho_new = vehicle_new / max(net.freeway_segment_length_km * max(lanes_now[i], 1.0e-9), 1.0e-9)

        upstream_speed = (_bu_speed_bc if buffer_bc is not None else net.v_free) if i == 0 else speeds[i - 1]
        if i + 1 < model.n_seg:
            downstream_rho = rho_for_flow[i + 1]
        elif buffer_bc is not None:
            # 완충 동결 결합: 코어 터미널 하류밀도 = 하류 완충 첫 셀(결정시점 동결).
            downstream_rho = _bd_rho0_bc
        elif getattr(net, "terminal_zero_gradient", False):
            # 구경계 복원 모드(plant와 동일): sink 밀도 = 자기 밀도(zero-gradient).
            downstream_rho = rho_for_flow[i]
        else:
            # plant와 동일한 자유출구 표준 경계: 가상 하류밀도 = min(자기, 임계).
            downstream_rho = min(rho_for_flow[i], float(net.rho_crit))
        vsl_i = segment_vsl(control, link, i, cfg)
        vsl_max = max(cfg.freeway_follower.vsl_set)
        vsl_active_i = vsl_i < vsl_max - 0.5
        v_eff = effective_desired_speed_kmh(
            rho, net.v_free, net.rho_crit, vsl_i, net.alpha_vsl, vsl_active_i, net.metanet_a_m,
            getattr(net, "vsl_fd_two_branch", False), net.rho_max,
            float(getattr(net, "rho_crit_two_branch", 0.0) or 0.0),
        )
        v_new = metanet_speed_update_kmh(
            speeds[i], upstream_speed, rho, downstream_rho, v_eff, dt_h,
            net.freeway_segment_length_km, net.metanet_tau_h,
            select_anticipation_nu(rho, net), net.metanet_kappa_veh_km_lane, net.v_min,
        )
        _delta_m = float(getattr(net, "metanet_delta_merge", 0.0) or 0.0)
        if _delta_m > 0.0 and ramp_in_by_segment[i] > 0.0:
            # plant merge 항 복제(예측-플랜트 정합): Δv=−δ·T·q_ramp·v/(L·λ·(ρ+κ)).
            v_new = max(
                net.v_min,
                v_new
                - _delta_m * dt_h * ramp_in_by_segment[i] * speeds[i]
                / (net.freeway_segment_length_km * max(lanes_now[i], 1.0e-9)
                   * (rho + net.metanet_kappa_veh_km_lane)),
            )
        if boundary_speed_cap is not None and v_new > boundary_speed_cap:
            v_new = max(net.v_min, boundary_speed_cap)
        next_rhos.append(rho_new)
        next_speeds.append(v_new)
        next_lanes.append(float(lanes_now[i]))
        next_vehicle_count.append(float(vehicle_new))

    return next_rhos, next_speeds, next_lanes, next_origin_queue, offramp_flow, next_vehicle_count
