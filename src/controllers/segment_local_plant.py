# freeway segment 1개(=13-player agent)만 전진시키는 국소 plant — 이웃 seg는 동결 궤적(y)으로 입력
"""13-player 재구축(plan-13player-rebuild.md)의 1단계 — segment agent의 국소동역학.

설계 원칙: METANET 수식을 복제하지 않는다. 기존 link-국소 stepper(`freeway_substep_local`)에
**이웃 segment 값은 동결 궤적(y)에서, 자기 segment 값은 자기 국소 상태에서** 채운 배열을
넘겨 호출하고, 자기 segment 결과만 취한다. 이러면

- link 전진과의 비트 일치가 구조적으로 보장되고(정합성 테스트가 이를 고정),
- 이웃은 substep당 상수 입력이므로 Wu §IV-D의 "결합변수 고정 + 자기 서브망만 전진"
  의미론이 정확히 성립한다(4-seg 산술을 전부 계산하지만 이웃 상태는 절대 갱신하지 않음 —
  비용 지배항은 후보 열거 × rollout이지 seg당 산술이 아니다).

y 스키마(FrozenLinkTrajectory) — agent 간 교환 대상:
- rhos/speeds/prev_lanes[t][j]: 이웃 segment 본선 상태 (f↔f 경계: METANET 상류 유입·속도,
  하류 receiving·anticipation 밀도가 전부 이 셋에서 유도됨)
- ramp_release[t][ramp]: 이웃 agent 소유 on-ramp의 방류(F_L2↔F_L3 예산 합의 결합변수 겸용)
- occupancy[t][off_ramp]: off-ramp storage 점유 — **urban D/F agent 소유 상태**(f→u 유출과
  u 신호 drain으로 urban이 갱신), freeway는 λ_eff(capacity drop) 입력으로만 읽음
- offramp_capacity[t][off_ramp]: urban receiving link 가용공간이 주는 유출 상한(u→f supply)
- origin_queue[t]: seg0 소유 상태의 동결값(비-seg0 agent의 배열 조립용; 자기 출력엔 무영향)

horizon을 넘어가는 t는 마지막 값 유지(hold-last).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from src.controllers.local_freeway_plant import (
    LocalFreewayModel,
    build_local_freeway_model,
    freeway_substep_local,
)
from src.models.state import ControlAction, ExperimentConfig


@dataclass
class SegmentAgentModel:
    """segment agent 1개의 정적 데이터 — 소유 lever와 link-모델 참조(컨트롤러가 1회 구성)."""

    link: str
    seg: int
    link_model: LocalFreewayModel
    # 이 segment에 merge하는 on-ramp(소유 lever). 13-player 매핑: seg2=R_D, seg3=R_F.
    owned_ramps: List[str]
    # 이 segment에서 갈라지는 off-ramp — 유출(q_out) 계산 담당은 이 agent, storage 상태는 urban 소유.
    boundary_offramps: List[str]
    # seg0만 mainline origin queue를 소유 상태로 가진다.
    owns_origin_queue: bool = False


def build_segment_agent_models(cfg: ExperimentConfig, link: str) -> List[SegmentAgentModel]:
    """link 하나를 segment agent들로 분할 — 13-player 매핑(승인 2026-07-10)."""
    link_model = build_local_freeway_model(cfg, link)
    agents: List[SegmentAgentModel] = []
    for seg in range(link_model.n_seg):
        owned = [r for r in link_model.owned_ramps if link_model.ramp_merge_idx[r] == seg]
        agents.append(
            SegmentAgentModel(
                link=link,
                seg=seg,
                link_model=link_model,
                owned_ramps=owned,
                boundary_offramps=list(link_model.offramps_by_segment.get(seg, [])),
                owns_origin_queue=(seg == 0),
            )
        )
    return agents


@dataclass
class FrozenLinkTrajectory:
    # Phase B: 완충 plant 동결 경계 (bu_send, bu_speed, bd_rho0) — None이면 무완충.
    buffer_bc = None
    """이웃 agent들이 내놓은 link 전체 궤적의 동결 스냅샷 — substep t 인덱스, hold-last."""

    rhos: List[List[float]]
    speeds: List[List[float]]
    prev_lanes: List[List[float]]
    origin_queue: List[float]
    ramp_release: List[Dict[str, float]]
    occupancy: List[Dict[str, float]]
    offramp_capacity: List[Dict[str, float]] = field(default_factory=list)

    def _idx(self, t: int) -> int:
        n = len(self.rhos)
        return min(max(t, 0), n - 1)

    def at(self, t: int) -> Tuple[
        List[float], List[float], List[float], float,
        Dict[str, float], Dict[str, float], Dict[str, float],
    ]:
        k = self._idx(t)
        cap = self.offramp_capacity[k] if self.offramp_capacity else {}
        return (
            list(self.rhos[k]),
            list(self.speeds[k]),
            list(self.prev_lanes[k]),
            float(self.origin_queue[k]),
            dict(self.ramp_release[k]),
            dict(self.occupancy[k]),
            dict(cap),
        )


@dataclass
class SegmentLocalState:
    """segment agent가 rollout 동안 스스로 갱신하는 자기 상태."""

    rho: float
    speed: float
    prev_lane: float
    origin_queue: float = 0.0  # seg0만 의미


def segment_substep_local(
    agent: SegmentAgentModel,
    frozen: FrozenLinkTrajectory,
    t: int,
    own: SegmentLocalState,
    own_ramp_release: Mapping[str, float],
    control: ControlAction,
    demand,
    extra_overrides: Optional[Mapping[int, SegmentLocalState]] = None,
) -> Tuple[SegmentLocalState, Dict[str, float], float]:
    """segment agent 1 substep 전진 — 반환 (다음 자기 상태, 자기 seg off-ramp 유출, 자기 seg 차량수).

    이웃 배열은 frozen.at(t)로 채우고 자기 seg 항목만 own/own_ramp_release로 덮어쓴 뒤
    `freeway_substep_local`을 호출, index=agent.seg 결과만 취한다.
    extra_overrides: radius-국소 rollout용 — frozen 대신 함께 전진 중인 이웃 seg의
    현재 상태를 주입(PFO 강화 변형; 기본 None이면 순수 동결-이웃).
    """
    rhos, speeds, prev_lanes, origin_q, releases, occupancy, cap = frozen.at(t)
    seg = agent.seg
    rhos[seg] = float(own.rho)
    speeds[seg] = float(own.speed)
    prev_lanes[seg] = float(own.prev_lane)
    if extra_overrides:
        for j, st in extra_overrides.items():
            if 0 <= int(j) < len(rhos) and int(j) != seg:
                rhos[int(j)] = float(st.rho)
                speeds[int(j)] = float(st.speed)
                prev_lanes[int(j)] = float(st.prev_lane)
    if agent.owns_origin_queue:
        origin_q = float(own.origin_queue)
    for ramp in agent.owned_ramps:
        releases[ramp] = max(0.0, float(own_ramp_release.get(ramp, 0.0)))

    next_rhos, next_speeds, next_lanes, next_origin_q, offramp_flow, vehicle_count = (
        freeway_substep_local(
            agent.link_model,
            rhos,
            speeds,
            prev_lanes,
            occupancy,
            origin_q,
            releases,
            cap,
            control,
            demand,
            buffer_bc=getattr(frozen, "buffer_bc", None),
        )
    )
    next_state = SegmentLocalState(
        rho=float(next_rhos[seg]),
        speed=float(next_speeds[seg]),
        prev_lane=float(next_lanes[seg]),
        origin_queue=float(next_origin_q) if agent.owns_origin_queue else 0.0,
    )
    own_offramp_flow = {
        o: float(offramp_flow.get(o, 0.0)) for o in agent.boundary_offramps
    }
    return next_state, own_offramp_flow, float(vehicle_count[seg])


def frozen_trajectory_from_state(
    cfg: ExperimentConfig,
    link: str,
    state,
    control: ControlAction,
    horizon_substeps: int,
) -> FrozenLinkTrajectory:
    """합의 첫 iteration용 warm-start — 현재 실측 상태를 hold-constant로 편 궤적.

    이후 iteration은 각 agent가 직전 rollout에서 내놓은 궤적으로 교체한다(합의 루프 담당).
    ramp_release는 직전 control의 metering을 유지, occupancy는 현 storage 점유를 유지.
    """
    net = cfg.network
    n_seg = net.freeway_segments_per_link
    rhos0 = [float(v) for v in state.freeway_density.get(link, [0.0] * n_seg)]
    speeds0 = [float(v) for v in state.freeway_speed.get(link, [net.v_free] * n_seg)]
    lanes0 = [float(v) for v in state.freeway_effective_lanes.get(link, [])] or [
        float(net.freeway_lanes) for _ in range(n_seg)
    ]
    if len(lanes0) != n_seg:
        lanes0 = [float(net.freeway_lanes) for _ in range(n_seg)]
    origin0 = max(0.0, float(state.mainline_origin_queue.get(link, 0.0)))
    release0: Dict[str, float] = {}
    for ramp in net.ramps:
        if net.ramp_to_freeway.get(ramp) == link:
            release0[ramp] = max(0.0, float(control.ramp_metering.get(ramp, 0.0)))
    occ0: Dict[str, float] = {}
    for off_ramp in net.off_ramps:
        if net.off_ramp_from_freeway.get(off_ramp) != link:
            continue
        storage = net.off_ramp_storage_link.get(off_ramp, "")
        cap = float(net.urban_link_storage_veh.get(storage, 0.0))
        avail = float(state.urban_link_storage.get(storage, cap))
        occ0[off_ramp] = max(0.0, cap - avail)
    steps = max(1, int(horizon_substeps))
    traj = FrozenLinkTrajectory(
        rhos=[list(rhos0) for _ in range(steps)],
        speeds=[list(speeds0) for _ in range(steps)],
        prev_lanes=[list(lanes0) for _ in range(steps)],
        origin_queue=[origin0 for _ in range(steps)],
        ramp_release=[dict(release0) for _ in range(steps)],
        occupancy=[dict(occ0) for _ in range(steps)],
        offramp_capacity=[{} for _ in range(steps)],
    )
    # Phase B(완충 동결 결합): 완충 plant면 결정시점 완충 경계를 동결해 첨부.
    if int(getattr(net, "freeway_buffer_segments", 0)) > 0:
        _bu_r = state.freeway_buffer_up_density.get(link) or []
        _bu_v = state.freeway_buffer_up_speed.get(link) or []
        _bd_r = state.freeway_buffer_down_density.get(link) or []
        if _bu_r and _bd_r:
            from src.models.metanet import segment_flow_veh_h as _sfvh
            _bu_send = _sfvh(_bu_r[-1], _bu_v[-1], float(net.freeway_lanes))
            _phi_bc = float(getattr(net, "capacity_drop_discharge_phi", 1.0) or 1.0)
            if _phi_bc < 1.0 and _bu_r[-1] > float(net.rho_crit):
                # plant capacity drop과 동일 cap(동결 BC 정합).
                _bu_send = min(_bu_send, _phi_bc * float(net.freeway_capacity_veh_h))
            traj.buffer_bc = (
                _bu_send,
                float(_bu_v[-1]),
                float(_bd_r[0]),
            )
    return traj
