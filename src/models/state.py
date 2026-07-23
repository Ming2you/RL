from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _is_integer_ratio(numerator: float, denominator: float, eps: float = 1.0e-9) -> bool:
    ratio = numerator / denominator
    return abs(ratio - round(ratio)) <= eps


def load_jsonish(path: str | Path) -> Dict[str, Any]:
    """Load project config from JSON-compatible YAML or a small YAML subset.

    The runtime intentionally has no PyYAML dependency. JSON remains the
    preferred fully structured format, and the fallback supports the simple
    mapping/list/scalar YAML used by the repository config files.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        parsed = _load_simple_yaml(text)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path} must contain a mapping at the top level.")
        return parsed


def _load_simple_yaml(text: str) -> Any:
    def parse_scalar(value: str) -> Any:
        value = value.strip()
        if value == "":
            return ""
        if value in ("true", "True"):
            return True
        if value in ("false", "False"):
            return False
        if value in ("null", "None", "~"):
            return None
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        try:
            if any(c in value for c in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value

    rows: List[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        rows.append((indent, stripped))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(rows):
            return {}, index
        if rows[index][0] < indent:
            return {}, index
        is_list = rows[index][1].startswith("- ")
        if is_list:
            values: List[Any] = []
            while index < len(rows) and rows[index][0] == indent and rows[index][1].startswith("- "):
                item = rows[index][1][2:].strip()
                if item:
                    values.append(parse_scalar(item))
                    index += 1
                else:
                    nested_indent = rows[index + 1][0] if index + 1 < len(rows) else indent + 2
                    value, index = parse_block(index + 1, nested_indent)
                    values.append(value)
            return values, index

        values: Dict[str, Any] = {}
        while index < len(rows) and rows[index][0] == indent and not rows[index][1].startswith("- "):
            key, sep, rest = rows[index][1].partition(":")
            if not sep:
                raise ValueError(f"Unsupported YAML line: {rows[index][1]}")
            key = key.strip()
            rest = rest.strip()
            if rest:
                values[key] = parse_scalar(rest)
                index += 1
            else:
                nested_indent = rows[index + 1][0] if index + 1 < len(rows) else indent + 2
                value, index = parse_block(index + 1, nested_indent)
                values[key] = value
        return values, index

    parsed, end = parse_block(0, rows[0][0] if rows else 0)
    if end != len(rows):
        raise ValueError("Unsupported YAML indentation or mixed collection structure.")
    return parsed


@dataclass
class SimulationConfig:
    T_total: float = 7200.0
    T_f: float = 10.0
    T_u: float = 5.0
    control_interval: float = 180.0
    random_seed: int = 42
    unit_time: str = "seconds"
    unit_flow: str = "veh/h"
    unit_speed: str = "km/h"
    unit_density: str = "veh/km/lane"
    derived_time_ratios: Dict[str, float] = field(default_factory=dict)

    @property
    def T_f_sec(self) -> float:
        return self.T_f

    @property
    def T_u_sec(self) -> float:
        return self.T_u

    @property
    def T_c_sec(self) -> float:
        return self.control_interval

    @property
    def T_f_h(self) -> float:
        return self.T_f_sec / 3600.0

    @property
    def T_u_h(self) -> float:
        return self.T_u_sec / 3600.0

    @property
    def T_c_h(self) -> float:
        return self.T_c_sec / 3600.0

    @property
    def control_interval_h(self) -> float:
        return self.T_c_h

    @property
    def n_control_steps(self) -> int:
        return max(1, int(math.ceil(self.T_total / self.T_c_sec)))

    @property
    def K_fu(self) -> int:
        return int(round(self.T_f_sec / self.T_u_sec))

    @property
    def K_cf(self) -> int:
        return int(round(self.T_c_sec / self.T_f_sec))

    @property
    def K_cu(self) -> int:
        return int(round(self.T_c_sec / self.T_u_sec))

    def validate(self) -> None:
        if min(self.T_f_sec, self.T_u_sec, self.T_c_sec) <= 0.0:
            raise ValueError("T_f, T_u, and control_interval must be positive.")
        if not _is_integer_ratio(self.T_f_sec, self.T_u_sec):
            raise ValueError("T_f must be an integer multiple of T_u.")
        if not _is_integer_ratio(self.T_c_sec, self.T_f_sec):
            raise ValueError("control_interval must be an integer multiple of T_f.")


@dataclass
class NetworkConfig:
    freeway_links: List[str] = field(default_factory=lambda: ["FW_W", "FW_E"])
    freeway_segments_per_link: int = 4
    # 완충 세그먼트 수(양쪽 각각, 2026-07-19): 0=완충 없음(기존 거동 비트동일). 코어 배열
    # 길이는 불변 — controllers/SEG13/VSL 키 전부 무접촉, plant만 상·하류 체인 연장.
    freeway_buffer_segments: int = 0
    freeway_segment_length_km: float = 0.5
    freeway_lanes: int = 2
    v_free: float = 100.0
    rho_crit: float = 33.5
    rho_max: float = 95.01964207118104
    freeway_capacity_veh_h: float = 3600.0
    # queue-discharge capacity drop(2026-07-20): 혼잡(ρ>ρ_crit_eff) 세그먼트의 송출을
    # 용량의 φ배로 제한. 1.0(기본)=완전 비활성(비트동일). 실측 근거: breakdown 후 배출률
    # 5~18% 감소(Hall & Agyemang-Duah 1991; Cassidy & Bertini 1999). ρ95식 FD 변형과 달리
    # 정적 FD·저장용량 불변 — 혼잡 진입의 동적 비용만 추가(민감도 arm φ=0.85).
    capacity_drop_discharge_phi: float = 1.0
    # METANET 표준 on-ramp merge 항 δ(2026-07-20, Messmer & Papageorgiou): 합류 유입이
    # 본선 속도를 교란 Δv=−δ·T·q_ramp·v/(L·λ·(ρ+κ)). 0(기본)=비활성(비트동일).
    # 문헌 표준값 δ≈0.0122 — merge 지점에서 breakdown이 촉발되는 현실 재현 + metering의
    # 교과서적 payoff(합류 교란 저감) 신설.
    metanet_delta_merge: float = 0.0
    ramps: List[str] = field(default_factory=lambda: ["R_D_W", "R_F_W", "R_D_E", "R_F_E"])
    ramp_to_freeway: Dict[str, str] = field(default_factory=lambda: {
        "R_D_W": "FW_W", "R_F_W": "FW_W", "R_D_E": "FW_E", "R_F_E": "FW_E"
    })
    ramp_capacity_veh_h: Dict[str, float] = field(default_factory=lambda: {
        "R_D_W": 1500.0, "R_F_W": 1500.0, "R_D_E": 1500.0, "R_F_E": 1500.0
    })
    ramp_merge_segment_index: Dict[str, int] = field(default_factory=lambda: {
        "R_D_W": 2, "R_F_W": 2, "R_D_E": 2, "R_F_E": 2
    })
    ramp_queue_max_veh: float = 180.0
    signals: List[str] = field(default_factory=lambda: ["A", "B", "C", "D", "F"])
    uncontrolled_nodes: List[str] = field(default_factory=lambda: ["E"])
    urban_links: List[str] = field(default_factory=lambda: [
        "A_B", "B_C", "A_D", "B_E", "C_F", "D_E", "E_F"
    ])
    cycle_length: float = 120.0
    lost_time: float = 8.0
    green_min: float = 20.0
    green_max: float = 92.0
    boundary_in_links: List[str] = field(default_factory=lambda: [
        "in_A_top", "in_A_left", "in_B_top", "in_C_top", "in_C_right", "in_D_left", "in_F_right"
    ])
    boundary_out_links: List[str] = field(default_factory=lambda: [
        "out_A_top", "out_A_left", "out_B_top", "out_C_top", "out_C_right", "out_D_left", "out_F_right"
    ])
    boundary_queue_max_veh: float = 240.0
    # boundary_out(출구) 링크당 시스템 이탈 유량 상한[veh/h] — 모델 밖 하류 도로 용량(A″-1).
    # 1600은 5개 표준 시나리오를 비왜곡(TTT 차이 ≤0.2%)하도록 calibration된 값이다. 표준
    # 운영엔 binding 안 하고, off-ramp split을 올린 heavy-transfer 시나리오에서만 출구가
    # binding해 off-ramp spillback을 형성한다(유입을 끊으면 urban 회복). 0 이하이면 자유 sink.
    boundary_out_capacity_veh_h: float = 1600.0
    movement_capacity_veh_h: float = 1400.0
    # urban_movements/turning_ratios/on·off_ramp_to_movement는 비워두면
    # grid_node_legs 토폴로지에서 자동 유도된다(__post_init__, grid_routing_proposal §3).
    grid_node_legs: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    turning_ratios: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    urban_movements: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    grid_link_storage_veh: float = 220.0
    urban_link_storage_veh: Dict[str, float] = field(default_factory=lambda: {
        "A_top_out": 220.0, "A_left_out": 220.0, "B_top_out": 220.0,
        "C_top_out": 220.0, "C_right_out": 220.0, "D_left_out": 220.0, "F_right_out": 220.0,
        "D_R_W": 180.0, "F_R_W": 180.0, "D_R_E": 180.0, "F_R_E": 180.0,
        "OR_D_W_storage": 120.0, "OR_F_W_storage": 120.0, "OR_D_E_storage": 120.0, "OR_F_E_storage": 120.0,
    })
    on_ramp_to_movement: Dict[str, List[str]] = field(default_factory=dict)
    off_ramps: List[str] = field(default_factory=lambda: ["OR_D_W", "OR_F_W", "OR_D_E", "OR_F_E"])
    off_ramp_from_freeway: Dict[str, str] = field(default_factory=lambda: {
        "OR_D_W": "FW_W",
        "OR_F_W": "FW_W",
        "OR_D_E": "FW_E",
        "OR_F_E": "FW_E",
    })
    off_ramp_segment_index: Dict[str, int] = field(default_factory=lambda: {
        "OR_D_W": 1, "OR_F_W": 2, "OR_D_E": 1, "OR_F_E": 2
    })
    off_ramp_storage_link: Dict[str, str] = field(default_factory=lambda: {
        "OR_D_W": "OR_D_W_storage",
        "OR_F_W": "OR_F_W_storage",
        "OR_D_E": "OR_D_E_storage",
        "OR_F_E": "OR_F_E_storage",
    })
    off_ramp_to_movement: Dict[str, List[str]] = field(default_factory=dict)
    off_ramp_split_ratio: Dict[str, float] = field(default_factory=lambda: {
        "OR_D_W": 0.06,
        "OR_F_W": 0.06,
        "OR_D_E": 0.06,
        "OR_F_E": 0.06,
    })
    v_min: float = 5.0
    alpha_vsl: float = 0.0
    metanet_tau_h: float = 0.005
    metanet_tau_sec: float = 18.0
    metanet_nu_km2_h: float = 65.0
    metanet_kappa_veh_km_lane: float = 40.0
    metanet_a_m: float = 1.867
    # Capacity drop (Arora & Kattan modified METANET, eq 9): 혼잡 regime(ρ>ρ_crit)에서
    # anticipation ν를 ν_cong로 전환해 속도·flow를 떨어뜨려 capacity drop을 표현한다.
    # toggle off면 ν_free(=metanet_nu_km2_h) 단일값으로 기존 거동 그대로.
    metanet_nu_cong_km2_h: float = 65.0
    capacity_drop_anticipation: bool = False
    metanet_rho_eps: float = 0.001
    urban_Q_sat_veh_h: float = 1000.0
    urban_avg_vehicle_length_m: float = 6.0
    urban_avg_speed_km_h: float = 50.0
    green_min_fraction: float = 0.2
    green_max_fraction: float = 0.8

    def __post_init__(self) -> None:
        # 토폴로지·movement·β를 leg 인접에서 자동 유도 — hand-list 금지(grid_routing_proposal §3).
        from .grid_topology import (
            build_urban_movements,
            default_grid_node_legs,
            derive_turning_ratios,
            internal_links,
        )

        if not self.grid_node_legs:
            self.grid_node_legs = default_grid_node_legs()
        if not self.turning_ratios:
            self.turning_ratios = derive_turning_ratios(self.grid_node_legs)
        for node, by_approach in self.turning_ratios.items():
            for token, beta in by_approach.items():
                total = sum(float(v) for v in beta.values())
                if abs(total - 1.0) > 1.0e-6:
                    raise ValueError(f"turning_ratios[{node}][{token}] must sum to 1, got {total}.")
        if not self.urban_movements:
            self.urban_movements = build_urban_movements(
                self.grid_node_legs,
                self.turning_ratios,
                self.signals,
                self.ramp_to_freeway,
            )
        # 내부 그리드 directed link(14개) storage를 자동 추가 — 용량 220 통일.
        for link in internal_links(self.grid_node_legs):
            self.urban_link_storage_veh.setdefault(link, float(self.grid_link_storage_veh))
        if not self.on_ramp_to_movement:
            self.on_ramp_to_movement = {ramp: [] for ramp in self.ramps}
            for movement, spec in self.urban_movements.items():
                ramp = str(spec.get("ramp", ""))
                if ramp and spec.get("kind") == "on_ramp":
                    self.on_ramp_to_movement.setdefault(ramp, []).append(movement)
        if not self.off_ramp_to_movement:
            self.off_ramp_to_movement = {off_ramp: [] for off_ramp in self.off_ramps}
            for movement, spec in self.urban_movements.items():
                off_ramp = str(spec.get("off_ramp", ""))
                if off_ramp:
                    self.off_ramp_to_movement.setdefault(off_ramp, []).append(movement)

    @property
    def movement_links(self) -> List[str]:
        return list(self.boundary_in_links) + list(self.boundary_out_links)

    @property
    def total_ramp_capacity(self) -> float:
        return float(sum(self.ramp_capacity_veh_h[r] for r in self.ramps))

    @property
    def effective_green_total(self) -> float:
        return max(0.0, self.cycle_length - self.lost_time)


@dataclass
class CapacityDropConfig:
    enabled: bool = True
    lane_reduction: float = 0.35
    gamma: float = 0.5
    b: float = 2.0


@dataclass
class MPCConfig:
    horizon_steps: int = 3
    # leader value-function 깊이(2026-07-08): leader의 full rollout(_predict)을 horizon_steps 너머
    # 이만큼 더 굴려 terminal cost V를 계산한다. **_predict은 leader 전용**(follower는 자기 local
    # rollout) → follower는 myopic-3 그대로, leader만 (3+d) full coupled rollout으로 후보 랭킹·price를
    # 매긴다 = ∂(TTT+V)/∂lever, V=leader full rollout tail. 기본 0=비트동일.
    leader_value_depth: int = 0
    # far(MFD tail) terminal cost(2026-07-08 구현, 2026-07-09 기본 ON — 사용자 지시):
    # leader 후보 채점(V=near+far)과 가격 rollout(price_far) 양쪽이 참조. urban(N²/2G,
    # boundary 큐 포함) + freeway(본선 N²/2G_fw + ramp 큐 bilinear). weight=1이 물리 정확값.
    leader_mfd_far_enabled: bool = True
    # far 항의 주행시간·배수율을 state(rho·유효차선)에서 유도(2026-07-16). 기본 False=상수식.
    leader_mfd_far_state_aware: bool = True
    # proxy-gradient 유도 리더 시드(2026-07-21): coarse 시드에 -∇f_proxy 하강 광선 추가.
    # False(기본)=원본 균등 Halton만(비트동일). enable_gradseed()가 True로 세팅.
    leader_gradseed_enabled: bool = False
    # box 상단 편향 Halton warp 지수(2026-07-21): u->u^p, p<1이면 상단 집중. 1.0=균등(비트동일).
    leader_bias_sample_pow: float = 1.0
    # 실제-v far(2026-07-20, 사용자 설계): state-aware 유도의 속도원(源)을 정적 FD V(ρ) 대신
    # rollout 말단 state의 **실제 속도**로 — Wang 물리의 동적 capacity drop(정적 FD에 없음)이
    # 말단 v에 이미 반영돼 있으므로 추가 적분 없이 배수율·주행시간에 전파. 혼잡(ρ>ρ_crit)
    # 병목의 배수율도 실제 유량(ρ·v·λ)으로 cap. 기본 False=기존 FD 유도(비트동일).
    leader_mfd_far_real_speed: bool = False
    leader_mfd_far_weight: float = 1.0
    # FAR-D0(2026-07-09): depth=0에서도 rollout TTT + far로 후보 채점(역사적 d0는
    # follower 응답 proxy 랭킹 — 채점 형태 고정한 "얕은 leader" 검정용). 기본 False.
    leader_mfd_far_at_d0: bool = False
    leader_search_mode: str = "continuous"
    leader_candidate_count: int = 49  # ★1단 채택 철회(2026-07-18): 60스텝선 파국(26스텝 A/B가 은폐). CAND=25 env로만.
    leader_refinement_candidate_count: int = 25
    leader_global_refresh_sec: float = 1800.0
    leader_local_np_radius_veh: float = 40.0
    leader_local_nuf_radius_veh_h: float = 1500.0
    # refined_candidates의 반경을 **실제로 구속**시킬지(2026-07-16 A/B). False면 앵커가
    # bounds로만 clip되어 반경을 우회 → '국소 재탐색'이 전역이 된다(실측 [1200,6000]).
    # 주의: 반경 1500은 6/20 도입값인데, trust_frac이 7/15에 0.25→0.20으로 바뀌며
    # 팔로워 합계 이동폭이 4×0.25×1500=1500 → 4×0.20×1500=1200이 되어 짝이 깨졌다.
    leader_local_radius_strict: bool = False
    # 후보 격자를 np-major로 앞에서 자르면 N_P 축이 굶는다(실측: 고유 N_P 4/N_UF 19,
    # 하한 앵커 2개가 38/49 독식, center 1개). True면 stride 표본으로 두 축 균등 표본.
    leader_budget_fair: bool = False
    # N_UF hard budget 제거 — leader는 가격만 넘기고 follower는 PFO autonomous metering
    # (2026-07-16 A/B). '+4.78%가 예산 몫인가 가격 몫인가' 분해용. 기본 False=비트동일.
    leader_budget_off: bool = False
    # METER-BOX(2026-07-17, 사용자 설계): SEG13 metering 후보를 고정 격자 {cap·f} 대신
    # 직전 commit m_prev ± R 박스 안 등간격 5점으로. 실측 근거: 선형 가격 × 이산 격자
    # → 내부 rung 선택 0/160(끝점 60~62%, 부호→끝점 적중 80~85%), 풀스팬 |Δ|=1125 점프.
    # R=300이면 박스=가격 FD 측정폭(0.20×1500) → 외삽 소멸. 예산 사영(_scale_to)도
    # 같은 박스로 clamp(박스=하드, 예산이 양보). None=기존 격자 = 비트동일(기본).
    # SEG13 경로 전용 — 비-SEG13은 이미 metering_marginal_price_trust_frac이 묶는다.
    seg13_meter_box_veh_h: Optional[float] = None
    # 비대칭 박스 올림폭(2026-07-17 2차): 대칭 박스 파국 2셀(170_w −140.9%/200_w −36.3%)이
    # 전부 '낮은 곳에 갇혀 못 올라옴'(리더 intent 후반 3651/4054 vs PD4 5895/5780)이라
    # 올림(방류·회복 방향)만 넓힌다. None=seg13_meter_box_veh_h와 동일(대칭, 기존 거동).
    seg13_meter_box_up_veh_h: Optional[float] = None
    # VSL-BOX(2026-07-17): SEG13 VSL 후보 앵커를 Jacobi 내부 snapshot → 직전 step
    # commit(previous)으로 + 반폭 R[km/h]. 기존은 sweep마다 재앵커돼 스텝당 실측 50
    # (명목 max_vsl_step 20의 2.5배, ③ 10셀 위반 112/7020). None=기존(비트동일).
    seg13_vsl_box_kmh: Optional[float] = None
    # BOX-WALK(2026-07-17 3차): 리더 rollout(_predict)에서 2번째 interval부터 metering을
    # 후보 intent(N_UF*) 방향으로 스텝당 램프별 ±R 전진 — 박스의 다중스텝 도달을 채점에
    # 반영("박스 끝 너머가 안 보임" 200_w −29.78% 수선 겸 가설 검증). 기본 False=비트동일.
    leader_rollout_box_walk: bool = False
    # BOX-WALK-VG(사용자 지적 "vsl도 green도 점진 탐색"): VSL·green도 rollout에 다중스텝
    # 이동 모델링. 후보가 목표를 안 주므로 끝 지속(edge persistence) — 이번 solve가 이동
    # 한계 끝까지 밀었으면 rollout에서 같은 방향·속도로 전진(전역 한계 정지). 기본 False.
    leader_rollout_box_walk_vg: bool = False
    # BASELINE-BOX(2026-07-17 밤, 사용자 지시 "PFO도 이동 반경"): 비-SEG13(PFO) 경로에
    # walk-MVG와 동일한 per-step 이동 한계 — metering |Δ|≤300(prev 앵커), green ±6s.
    # 실측: 무제한 PFO는 metering 최대 1125/green 최대 57s(공정비교 §2.4 위반).
    # VSL은 PFO 실측 이동 0이라 미적용. 기본 False=비트동일.
    baseline_move_box: bool = False
    leader_continuous_max_evals: int = 25
    leader_continuous_seed_count: int = 7
    leader_continuous_prefilter_samples: int = 31
    leader_continuous_prefilter_top_k: int = 7
    # 비-global(local) 스텝 전용 축소 예산. full 탐색은 leader_global_refresh_sec마다만 하고,
    # 그 외 스텝은 previous 인근에서 sensitivity 상위 몇 점만 평가해 비용을 낮춘다.
    leader_continuous_local_max_evals: int = 6
    leader_continuous_local_seed_count: int = 3
    leader_continuous_local_prefilter_samples: int = 12
    leader_continuous_local_prefilter_top_k: int = 3
    leader_continuous_hard_precheck: bool = True
    leader_continuous_precheck_spillback_tolerance_veh: float = 0.0
    leader_continuous_parallel_multistart: bool = True
    leader_continuous_use_sensitivity_directions: bool = True
    leader_continuous_local_iterations: int = 4
    leader_continuous_initial_step_fraction: float = 0.35
    leader_continuous_shrink_factor: float = 0.5
    leader_continuous_min_np_step_veh: float = 40.0
    leader_continuous_min_nuf_step_veh_h: float = 125.0
    follower_solver_mode: str = "two_block"
    max_nash_iter: int = 10
    nash_obj_tol: float = 1.0e-3
    nash_control_tol: float = 1.0e-3
    nash_relaxation_alpha: float = 0.8
    distributed_coupling_tol: float = 1.0e-3
    control_horizon_steps: int = 3
    urban_freeway_tts_weight_alpha: float = 1.0
    optimizer_maxiter: int = 40
    optimizer_n_starts: int = 2
    centralized_solver_mode: str = "slsqp"
    centralized_slsqp_ftol: float = 1.0e-3
    # SLSQP 하드 평가예산(스텝당 full-network rollout 상한). 0=무제한(수렴까지, 비평활서 폭주).
    # >0이면 예산 소진 시 best-so-far 반환 후 종료 — grid와 동일 예산 공정비교용(2026-07-21).
    centralized_slsqp_max_eval: int = 0
    # P-CENT 구조적 그리드 tightness 실험 노브(기본=현행). refresh_sec 낮추면 전역 재탐색을
    # 매 스텝, dense=True면 레버별 격자 레벨을 조밀하게(상한 빡빡함 측정용).
    centralized_grid_refresh_sec: float = 1800.0
    centralized_grid_dense: bool = False
    relaxed_quantized_controls: bool = False
    relaxed_green_quantum_sec: float = 1.0
    relaxed_vsl_quantum_km_h: float = 10.0
    relaxed_rounding_mode: str = "floor"
    relaxed_wu_vsl_include_neutral: bool = True
    grid_global_refresh_sec: float = 1800.0
    grid_parallel_backend: str = "thread"
    grid_parallel_max_workers: int = 8
    grid_parallel_min_items: int = 2
    grid_parallel_chunk_size: int = 8
    grid_reuse_process_pool: bool = True
    stackelberg_prefilter_top_k: int = 4
    stackelberg_prefilter_local_top_k: int = 4
    stackelberg_fallback_full_refresh_sec: float = 1800.0
    stackelberg_fallback_use_cached_pfo: bool = True
    stackelberg_enable_fallback: bool = True
    stackelberg_enable_pfo_incumbent: bool = True
    # fallback guard의 leader vs PFO 비교 척도를 penalized objective 대신 realized rollout-TTT로.
    # (penalized obj는 TTT와 어긋나 sweet_128 등에서 TTT 좋은 leader를 잘못 기각했다, 2026-06-25.)
    stackelberg_fallback_guard_use_rollout_ttt: bool = True
    # ---- 층2(2026-07-14): 낙관편향 β̂ 추정기 + β̂ 보정 guard + trailing-regret 스위치 ----
    # 배경: leader 내부 rollout은 체계적으로 낙관(~30%: 제약 누락·capacity-drop 절벽 평활·
    # horizon 절단·동결 결합)이고 argmax 선택이 이를 증폭(optimizer's curse) — 그 결과
    # 예측 vs 예측 비교인 fallback guard는 실현 +49% 파국에서도 발화하지 못했다.
    # 기본 전부 OFF = 비트동일.
    # β̂ = EWMA(실현 interval TTT / 커밋 계획의 예측 첫-interval TTT). 추정 전용(행동 불변),
    # 진단 leader_beta_hat/leader_pred_interval_ttt/leader_realized_interval_ttt export.
    leader_bias_estimator: bool = True  # 2026-07-14: β̂ 계기 기본 ON(진단 전용, 결정 무변경 — 논문 그림 재료)
    # guard의 leader측 예측 rollout TTT를 β̂ 배율로 보정(β̂·leader_pred > incumbent_pred면
    # 기각). incumbent측은 무보정 — argmax 선택편향이 없다. leader_bias_estimator 필요.
    fallback_guard_beta: bool = False
    # k>0: 최근 k스텝 실현 interval TTT 합 > 같은 스텝 incumbent 예측 합×1.10이면
    # 다음 k스텝 강제 incumbent 커밋(hysteresis — 강제 중에도 창 계속 갱신). 0=OFF.
    regret_guard_steps: int = 3  # 2026-07-14 동결(A4 승인): 실현-regret 안전망 기본 ON — 파국 차단(dhigh2_w +49%→+7.9%)·승리셀 무해/개선(3/3)
    stackelberg_leader_parallel_backend: str = "thread"
    stackelberg_leader_parallel_max_workers: int = 4
    stackelberg_inner_backend_when_outer_process: str = "thread"
    stackelberg_reuse_process_pool: bool = True
    stackelberg_allocation_mode: str = "direct"
    wu_np_storage_guard: bool = False
    wu_np_arrival_mode: str = "horizon"
    wu_np_phase_substep: bool = False
    wu_faithful_np_predictor_mode: str = "legacy"
    # λ_P windup 수선(2026-07-11, 규칙 2종 — 구조 불변): NP_FIX=0으로 구거동 재현.
    # ① 내부 투영: target을 feasibility 모서리(feas_min) 대신 내부점으로 클립 —
    #    모서리는 균형이 아니라 오차가 구조적 양수(+109~301 실측)여서 λ 단방향 적분.
    np_target_interior_frac: float = 0.25
    # ② 경부하 deadband: 보호 accumulation < frac·N_P_crit면 적분 대신 감쇠(0.5×).
    np_dual_deadband_frac: float = 0.9
    # NP-CAND-λ̂(2026-07-12, 리뷰 4안 → 원고 (47)~(51) 정식화): 후보 평가마다 λ를 후보
    # target으로 1회 선반영(predictor) + 차기 스텝 실현 유입으로 교정(corrector).
    # 기본 ON(2026-07-12 확정 — A/B: 8-seg 155_skew 7490/준거 7509, 190 15535/15652).
    # NP_CAND_LAMBDA=0으로 구거동(스텝 내 λ 동결) 재현.
    np_candidate_lambda: bool = True
    # r̂ 편향 보정(2026-07-13): λ̂ 오차 비교의 target을 실현 공간으로 환산(r̂·Ñ).
    # 계획(예측 Σnin) vs 실현(ΔN_P×H)의 낙관 편향이 채널 휴면의 원인 — 보정 시
    # 실현 유입이 환산 상한에 닿으면 λ̂가 실제로 발동한다. False=비트동일.
    np_bias_correction: bool = False
    # deadband v2(2026-07-13): 위반 신호(유입 > 환산 target)는 저stock 게이트를 우회해
    # 적분한다 — 펄스 loading edge에서 stock 지연이 진짜 위반을 삼키는 병리 해소.
    # 경부하 windup 수선(위반 없는 저stock 감쇠)은 보존. False=비트동일.
    np_deadband_violation_override: bool = False
    # 방법 A — candidate 내부 primal-dual 반복 횟수 K. 0=OFF(현행 1회 선반영).
    # K>0이면 후보별로 λ^(κ+1)=Π[λ^(κ)+γ_P(Σν^(κ)−Ñ^(c))]를 K회(조기수렴 허용) 반복하고
    # 각 κ마다 Jacobi를 재수렴시켜 (λ*, green*) 안장점을 커밋한다.
    np_primal_dual_iters: int = 0
    # 방법 A 내부 반복용 γ_P 배율 — 1회당 0.01은 K≤5 안에 수렴 불가, 25배≈0.25로
    # 5회 내 안장점 도달.
    np_pd_gain_mult: float = 25.0
    # ε-best-response gap probe(2026-07-12, 리뷰 2.2/2.8 대응): 수렴 고정점에서 각
    # follower를 최종 결합변수 하에 단독 재최적화해 개선 여지를 측정(진단 전용, 행동 불변).
    eps_gap_probe: bool = False
    # Leader hinge 복원(2026-07-12, 사용자 지시): 구 F1 hinge 2종(freeway ρ_crit 초과,
    # urban 0.5cap spill)을 leader candidate 채점에만 가산. 기본 ON(A/B 양면 중립:
    # 190 15722/준거 15652, 155_skew 7514/7509 — 무해 확인). LEADER_HINGE=0으로 해제.
    leader_hinge_enabled: bool = False  # 2026-07-14 동결: 다이아몬드 기하에서 만료(155 -572·155_skew -349·190 중립, 매트릭스 3셀 일관). LEADER_HINGE=1로 복원 A/B
    leader_hinge_weight: float = 1.0
    leader_hinge_spill_frac: float = 0.5
    wu_faithful_np_coordination_mode: str = "cap"
    # N_UF 조정 모드: "equality"=leader link budget으로 metering 합을 hard 고정,
    # "cap"=budget을 상한으로만 쓰고 자율 metering 좌표하강을 존중(합 ≤ budget 투영),
    # "dual"=λ_UF(signed, step 간 적분)로 Σmeter가 N_UF*를 추적.
    # 기본 equality(2026-07-09 복귀 — 사전등록 기준). dual은 3세대 검증 끝에 부정결과:
    # v1 bootstrap deadlock(λ 영영 0) → v2 windup(incumbent λ-면제) → v3 공정 테스트
    # (λ 정상 작동, 루프 닫힘)에서도 equality에 +583 패배. 원인: leader target(N_UF*)이
    # 매 스텝 움직이는데 적분 dual은 저역통과 전송이라 항상 지연 — equality는 무지연.
    # (Weitzman: 절벽 레버는 수량 지배 + 비정상 target은 수량이 전송 우위.)
    wu_faithful_nuf_coordination_mode: str = "equality"


@dataclass
class LeaderConfig:
    objective_mode: str = "follower_ttt"
    w_P: float = 1.0
    w_F: float = 1.0
    w_L: float = 0.0
    # Step D: boundary_in 큐 비용은 진단용으로 계산하되 leader_total_objective에는 더하지 않는다.
    w_boundary_in: float = 0.0
    # ramp-queue terminal cost (2026-07-07, "ramp = hidden space" 처방): on-ramp 큐에 stuck된
    # 차 = exit에서 가장 먼 차 = 최대 deferred 비용. leader objective에 선형 ramp-큐 항을 더하면
    # 방류(N_UF↑)가 ramp를 배수해 penalty↓ → realized-TTT의 위치 불변성(방류=차 이동, net-0)이
    # 만든 flat을 깨고 방류 신호를 준다. terminal cost = 무한지평 cost-to-go 근사(짧은 horizon을
    # 무한지평처럼 — Mayne et al. 2000; store-and-forward 큐목적과 정합). 기본 0=비트동일.
    w_ramp_queue: float = 0.0
    mfd_penalty_mode: str = "all_urban_halfcap"
    mfd_storage_threshold_ratio: float = 0.5
    mfd_storage_weight: float = 1.0
    mfd_boundary_queue_capacity_veh: float = 220.0
    N_P_star_range: List[float] = field(default_factory=lambda: [-3500.0, 3500.0])
    N_UF_star_range: List[float] = field(default_factory=lambda: [0.0, 6000.0])
    N_P_crit_veh: float = 509.448830418254
    N_P_candidate_lower_factor: float = 0.40
    N_P_candidate_upper_factor: float = 1.35
    N_P_star_unit: str = "veh"
    N_UF_star_unit: str = "veh_per_hour"
    N_P_feedback_horizon_h: float = 0.5
    N_P_feedback_flow_limit_veh_h: float = 800.0
    N_UF_feasible_margin: float = 0.95
    # Step D: non-convergence penalty도 진단용으로만 계산한다.
    non_convergence_penalty: float = 500.0
    non_convergence_objective_residual_scale: float = 1.0
    non_convergence_control_residual_scale: float = 1.0
    state_accumulation_exclude_boundary_legs: bool = True
    use_effective_lanes_for_density_penalty: bool = True
    metering_congestion_weight: float = 0.45
    metering_queue_weight: float = 4.0
    vsl_activation_density_ratio: float = 0.95
    metering_activation_density_ratio: float = 0.95
    # low_demand 회귀 수정(fix 1): 저혼잡(density<=metering_activation)에서 N_UF* 하한을
    # heuristic의 이 비율로 강제하던 clamp. 0.0이면 강제 없음 → leader가 낮은 자연 방출(=PFO 동등
    # 운전점)도 탐색 가능. (구버전 동작 재현: 0.75)
    uncongested_nuf_floor_frac: float = 0.0


@dataclass
class FreewayFollowerConfig:
    eps_F: float = 100.0
    vsl_set: List[float] = field(default_factory=lambda: [50, 60, 70, 80, 90, 100])
    max_vsl_step: float = 20.0
    ramp_queue_penalty: float = 10.0
    density_penalty: float = 10.0
    metering_smoothness_weight: float = 0.1
    vsl_smoothness_weight: float = 0.1
    # WU-CD-F freeway agent의 VSL multi-step 예측 horizon(Wu Np=10). storage-aware probe가
    # "VSL↓→off-ramp 유입↓→storage 회복→λ_eff 회복→본선 차량수↓"의 multi-step 이득을
    # 보려면 한두 step으로는 부족하다(off-ramp 동역학이 여러 step에 걸쳐 회복). 0 이하면
    # mpc.horizon_steps로 fallback. 비용 폭증을 막기 위해 segment 후보 가지치기는 유지.
    freeway_prediction_horizon_steps: int = 3  # 2026-07-14 동결: 전 시스템 H=3 통일(FH 스윕으로 10 잉여 확인, 원고 서술 일치. FH env로 A/B 가능)
    horizon_beam_width: int = 2
    horizon_ramp_candidate_limit: int = 3
    horizon_vsl_candidate_limit_per_link: int = 3
    vsl_sequence_search: bool = True
    vsl_sequence_horizon_steps: int = 4
    vsl_sequence_candidate_limit: int = 12
    ramp_metering_rate_min: float = 0.2
    ramp_metering_rate_max: float = 1.0
    vsl_min_km_h: float = 60.0
    vsl_max_km_h: float = 106.0


@dataclass
class UrbanFollowerConfig:
    eps_U: float = 100.0
    eps_g: float = 5.0
    max_offset_step: float = 15.0
    boundary_balance_weight: float = 10.0
    offset_smoothness_weight: float = 0.1
    green_smoothness_weight: float = 0.1
    receiving_space_rule: str = "proportional"
    allocation_pso_particles: int = 18
    allocation_pso_iterations: int = 24
    allocation_green_band_sec: float = 5.0


@dataclass
class EvaluationConfig:
    main_metric: str = "total_ttt"
    main_metric_direction: str = "lower_is_better"
    min_improvement_pct: float = 8.0
    eps: float = 1.0e-9
    eps_balance: float = 0.03
    boundary_degenerate_saturation_fraction: float = 0.95
    boundary_degenerate_ratio: float = 0.5
    # run 전체에서 제어가능(비-degenerate) interval 비율이 이 값 미만일 때만
    # boundary balance를 degenerate로 판정한다(부하 구간 시간 집계 기준).
    boundary_controllable_min_fraction: float = 0.25


@dataclass
class AutoTuningConfig:
    enabled: bool = True
    max_iterations: int = 5
    preserve_all_runs: bool = True


@dataclass
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    freeway_offramp_capacity_drop: CapacityDropConfig = field(default_factory=CapacityDropConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    freeway_follower: FreewayFollowerConfig = field(default_factory=FreewayFollowerConfig)
    urban_follower: UrbanFollowerConfig = field(default_factory=UrbanFollowerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    auto_tuning: AutoTuningConfig = field(default_factory=AutoTuningConfig)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        cfg = cls(
            simulation=SimulationConfig(**raw.get("simulation", {})),
            network=NetworkConfig(**raw.get("network", {})),
            freeway_offramp_capacity_drop=CapacityDropConfig(**raw.get("freeway_offramp_capacity_drop", {})),
            mpc=MPCConfig(**raw.get("mpc", {})),
            leader=LeaderConfig(**raw.get("leader", {})),
            freeway_follower=FreewayFollowerConfig(**raw.get("freeway_follower", {})),
            urban_follower=UrbanFollowerConfig(**raw.get("urban_follower", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            auto_tuning=AutoTuningConfig(**raw.get("auto_tuning", {})),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        self.simulation.validate()
        if self.mpc.follower_solver_mode not in {"two_block", "distributed"}:
            raise ValueError("mpc.follower_solver_mode must be two_block or distributed.")
        if self.mpc.leader_search_mode not in {"grid", "continuous"}:
            raise ValueError("mpc.leader_search_mode must be grid or continuous.")
        if self.mpc.centralized_solver_mode not in {"structured_grid", "slsqp"}:
            raise ValueError("mpc.centralized_solver_mode must be structured_grid or slsqp.")
        if self.mpc.centralized_slsqp_ftol <= 0.0:
            raise ValueError("mpc.centralized_slsqp_ftol must be positive.")
        if self.mpc.leader_candidate_count <= 0:
            raise ValueError("mpc.leader_candidate_count must be positive.")
        if self.mpc.leader_refinement_candidate_count <= 0:
            raise ValueError("mpc.leader_refinement_candidate_count must be positive.")
        if self.mpc.leader_global_refresh_sec <= 0.0:
            raise ValueError("mpc.leader_global_refresh_sec must be positive.")
        if self.mpc.leader_local_np_radius_veh <= 0.0:
            raise ValueError("mpc.leader_local_np_radius_veh must be positive.")
        if self.mpc.leader_local_nuf_radius_veh_h <= 0.0:
            raise ValueError("mpc.leader_local_nuf_radius_veh_h must be positive.")
        if self.mpc.leader_continuous_max_evals <= 0:
            raise ValueError("mpc.leader_continuous_max_evals must be positive.")
        if self.mpc.leader_continuous_seed_count <= 0:
            raise ValueError("mpc.leader_continuous_seed_count must be positive.")
        if self.mpc.leader_continuous_prefilter_samples < 0:
            raise ValueError("mpc.leader_continuous_prefilter_samples must be non-negative.")
        if self.mpc.leader_continuous_prefilter_top_k <= 0:
            raise ValueError("mpc.leader_continuous_prefilter_top_k must be positive.")
        if self.mpc.leader_continuous_local_max_evals <= 0:
            raise ValueError("mpc.leader_continuous_local_max_evals must be positive.")
        if self.mpc.leader_continuous_local_seed_count <= 0:
            raise ValueError("mpc.leader_continuous_local_seed_count must be positive.")
        if self.mpc.leader_continuous_local_prefilter_samples < 0:
            raise ValueError("mpc.leader_continuous_local_prefilter_samples must be non-negative.")
        if self.mpc.leader_continuous_local_prefilter_top_k <= 0:
            raise ValueError("mpc.leader_continuous_local_prefilter_top_k must be positive.")
        if self.mpc.leader_continuous_precheck_spillback_tolerance_veh < 0.0:
            raise ValueError("mpc.leader_continuous_precheck_spillback_tolerance_veh must be non-negative.")
        if self.mpc.leader_continuous_local_iterations < 0:
            raise ValueError("mpc.leader_continuous_local_iterations must be non-negative.")
        if self.mpc.leader_continuous_initial_step_fraction <= 0.0:
            raise ValueError("mpc.leader_continuous_initial_step_fraction must be positive.")
        if not 0.0 < self.mpc.leader_continuous_shrink_factor < 1.0:
            raise ValueError("mpc.leader_continuous_shrink_factor must be in (0, 1).")
        if self.mpc.leader_continuous_min_np_step_veh <= 0.0:
            raise ValueError("mpc.leader_continuous_min_np_step_veh must be positive.")
        if self.mpc.leader_continuous_min_nuf_step_veh_h <= 0.0:
            raise ValueError("mpc.leader_continuous_min_nuf_step_veh_h must be positive.")
        if self.mpc.distributed_coupling_tol <= 0.0:
            raise ValueError("mpc.distributed_coupling_tol must be positive.")
        if self.mpc.relaxed_green_quantum_sec <= 0.0:
            raise ValueError("mpc.relaxed_green_quantum_sec must be positive.")
        if self.mpc.relaxed_vsl_quantum_km_h <= 0.0:
            raise ValueError("mpc.relaxed_vsl_quantum_km_h must be positive.")
        if self.mpc.relaxed_rounding_mode not in {"floor", "nearest"}:
            raise ValueError("mpc.relaxed_rounding_mode must be floor or nearest.")
        if self.mpc.wu_faithful_np_coordination_mode not in {"equality", "cap"}:
            raise ValueError("mpc.wu_faithful_np_coordination_mode must be equality or cap.")
        if self.mpc.wu_faithful_nuf_coordination_mode not in {"equality", "cap", "dual"}:
            raise ValueError("mpc.wu_faithful_nuf_coordination_mode must be equality, cap, or dual.")
        if self.mpc.grid_global_refresh_sec <= 0.0:
            raise ValueError("mpc.grid_global_refresh_sec must be positive.")
        if self.mpc.grid_parallel_backend not in {"serial", "thread", "process"}:
            raise ValueError("mpc.grid_parallel_backend must be serial, thread, or process.")
        if self.mpc.grid_parallel_max_workers <= 0:
            raise ValueError("mpc.grid_parallel_max_workers must be positive.")
        if self.mpc.grid_parallel_min_items <= 0:
            raise ValueError("mpc.grid_parallel_min_items must be positive.")
        if self.mpc.grid_parallel_chunk_size <= 0:
            raise ValueError("mpc.grid_parallel_chunk_size must be positive.")
        if self.mpc.stackelberg_prefilter_top_k < 0:
            raise ValueError("mpc.stackelberg_prefilter_top_k must be non-negative.")
        if self.mpc.stackelberg_prefilter_local_top_k < 0:
            raise ValueError("mpc.stackelberg_prefilter_local_top_k must be non-negative.")
        if self.mpc.stackelberg_fallback_full_refresh_sec <= 0.0:
            raise ValueError("mpc.stackelberg_fallback_full_refresh_sec must be positive.")
        if self.mpc.stackelberg_leader_parallel_backend not in {"serial", "thread", "process"}:
            raise ValueError("mpc.stackelberg_leader_parallel_backend must be serial, thread, or process.")
        if self.mpc.stackelberg_leader_parallel_max_workers <= 0:
            raise ValueError("mpc.stackelberg_leader_parallel_max_workers must be positive.")
        if self.mpc.stackelberg_inner_backend_when_outer_process not in {"serial", "thread"}:
            raise ValueError("mpc.stackelberg_inner_backend_when_outer_process must be serial or thread.")
        if self.mpc.stackelberg_allocation_mode not in {"direct", "simplified", "pso"}:
            raise ValueError("mpc.stackelberg_allocation_mode must be direct, simplified, or pso.")
        if self.mpc.wu_faithful_np_predictor_mode not in {
            "legacy",
            "storage_aware",
            "current_interval",
            "phase_substep",
        }:
            raise ValueError(
                "mpc.wu_faithful_np_predictor_mode must be legacy, storage_aware, "
                "current_interval, or phase_substep."
            )
        if self.mpc.wu_np_arrival_mode not in {"horizon", "current_interval"}:
            raise ValueError("mpc.wu_np_arrival_mode must be horizon or current_interval.")
        cap_drop = self.freeway_offramp_capacity_drop
        if cap_drop.lane_reduction < 0.0:
            raise ValueError("freeway_offramp_capacity_drop.lane_reduction must be non-negative.")
        if cap_drop.lane_reduction >= self.network.freeway_lanes:
            raise ValueError("freeway_offramp_capacity_drop.lane_reduction must be less than freeway_lanes.")
        if cap_drop.gamma <= 0.0:
            raise ValueError("freeway_offramp_capacity_drop.gamma must be positive.")
        if cap_drop.b <= 0.0:
            raise ValueError("freeway_offramp_capacity_drop.b must be positive.")
        if self.leader.objective_mode not in {"state_accumulation", "follower_ttt"}:
            raise ValueError("leader.objective_mode must be state_accumulation or follower_ttt.")
        if self.leader.mfd_penalty_mode not in {
            "disabled",
            "protected_exceed",
            "all_urban_halfcap",
            "combined",
        }:
            raise ValueError(
                "leader.mfd_penalty_mode must be disabled, protected_exceed, "
                "all_urban_halfcap, or combined."
            )
        if not 0.0 <= self.leader.mfd_storage_threshold_ratio <= 1.0:
            raise ValueError("leader.mfd_storage_threshold_ratio must be in [0, 1].")
        if self.leader.mfd_storage_weight < 0.0:
            raise ValueError("leader.mfd_storage_weight must be non-negative.")
        if self.leader.mfd_boundary_queue_capacity_veh <= 0.0:
            raise ValueError("leader.mfd_boundary_queue_capacity_veh must be positive.")
        if self.leader.N_P_star_unit != "veh":
            raise ValueError("leader.N_P_star_unit must be veh.")
        if self.leader.N_P_crit_veh <= 0.0:
            raise ValueError("leader.N_P_crit_veh must be positive.")
        if self.leader.N_P_candidate_lower_factor <= 0.0:
            raise ValueError("leader.N_P_candidate_lower_factor must be positive.")
        if self.leader.N_P_candidate_upper_factor < self.leader.N_P_candidate_lower_factor:
            raise ValueError("leader.N_P_candidate_upper_factor must be >= lower factor.")
        if self.leader.N_UF_star_unit not in {"veh_per_hour", "veh_per_control_interval"}:
            raise ValueError("leader.N_UF_star_unit must be veh_per_hour or veh_per_control_interval.")
        if self.leader.non_convergence_objective_residual_scale <= 0.0:
            raise ValueError("leader.non_convergence_objective_residual_scale must be positive.")
        if self.leader.non_convergence_control_residual_scale <= 0.0:
            raise ValueError("leader.non_convergence_control_residual_scale must be positive.")
        if self.freeway_follower.vsl_sequence_horizon_steps <= 0:
            raise ValueError("freeway_follower.vsl_sequence_horizon_steps must be positive.")
        if self.freeway_follower.vsl_sequence_candidate_limit <= 0:
            raise ValueError("freeway_follower.vsl_sequence_candidate_limit must be positive.")
        if self.urban_follower.allocation_pso_particles <= 0:
            raise ValueError("urban_follower.allocation_pso_particles must be positive.")
        if self.urban_follower.allocation_pso_iterations <= 0:
            raise ValueError("urban_follower.allocation_pso_iterations must be positive.")
        if self.evaluation.eps_balance < 0.0:
            raise ValueError("evaluation.eps_balance must be non-negative.")
        if not 0.0 <= self.evaluation.boundary_degenerate_saturation_fraction <= 1.0:
            raise ValueError("evaluation.boundary_degenerate_saturation_fraction must be in [0, 1].")
        if not 0.0 <= self.evaluation.boundary_degenerate_ratio <= 1.0:
            raise ValueError("evaluation.boundary_degenerate_ratio must be in [0, 1].")

    @classmethod
    def from_file(cls, path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> "ExperimentConfig":
        raw = load_jsonish(path)
        if overrides:
            raw = _deep_update(raw, overrides)
        return cls.from_dict(raw)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_updates(self, updates: Mapping[str, Any]) -> "ExperimentConfig":
        return ExperimentConfig.from_dict(_deep_update(self.to_dict(), updates))


@dataclass
class TrafficState:
    freeway_density: Dict[str, List[float]]
    freeway_speed: Dict[str, List[float]]
    freeway_flow: Dict[str, List[float]]
    ramp_queue: Dict[str, float]
    urban_queue: Dict[str, float]
    boundary_queue: Dict[str, float]
    freeway_effective_lanes: Dict[str, List[float]] = field(default_factory=dict)
    # 본선 진입 origin 큐[veh]: CTM receiving 제약으로 segment 0에 못 들어간 본선 수요를
    # 링크별로 보관한다(spec §3.1.2 demand-supply 개정, 차량보존). freeway entry에서만 사용.
    mainline_origin_queue: Dict[str, float] = field(default_factory=dict)
    # 완충 세그먼트(2026-07-19, FW_BUFFER): 코어 밖 상·하류 무제어 METANET 셀 — plant 전용.
    # 혼잡이 경계 가정 대신 실제 셀에서 퍼지고 배수되게 한다(controllers 무접촉).
    freeway_buffer_up_density: Dict[str, List[float]] = field(default_factory=dict)
    freeway_buffer_up_speed: Dict[str, List[float]] = field(default_factory=dict)
    freeway_buffer_down_density: Dict[str, List[float]] = field(default_factory=dict)
    freeway_buffer_down_speed: Dict[str, List[float]] = field(default_factory=dict)
    urban_movement_queue: Dict[str, float] = field(default_factory=dict)
    urban_link_storage: Dict[str, float] = field(default_factory=dict)
    urban_arrival_buffer: Dict[str, Dict[int, float]] = field(default_factory=dict)
    urban_storage_release_buffer: Dict[str, Dict[int, float]] = field(default_factory=dict)
    time_sec: float = 0.0

    @classmethod
    def initial(cls, cfg: ExperimentConfig) -> "TrafficState":
        net = cfg.network
        density = {
            link: [18.0 for _ in range(net.freeway_segments_per_link)]
            for link in net.freeway_links
        }
        speed = {
            link: [net.v_free for _ in range(net.freeway_segments_per_link)]
            for link in net.freeway_links
        }
        flow = {
            link: [
                max(0.0, density[link][i]) * max(0.0, speed[link][i]) * max(0.0, net.freeway_lanes)
                for i in range(net.freeway_segments_per_link)
            ]
            for link in net.freeway_links
        }
        lanes = {
            link: [float(net.freeway_lanes) for _ in range(net.freeway_segments_per_link)]
            for link in net.freeway_links
        }
        return cls(
            freeway_density=density,
            freeway_speed=speed,
            freeway_flow=flow,
            ramp_queue={r: 0.0 for r in net.ramps},
            urban_queue={m: 20.0 for m in net.movement_links},
            boundary_queue={m: 20.0 for m in net.movement_links},
            freeway_effective_lanes=lanes,
            mainline_origin_queue={link: 0.0 for link in net.freeway_links},
            # 게이트 대기열(boundary_in)만 초기 20대를 β로 나눠 갖고, 내부 큐는 0에서 시작한다.
            urban_movement_queue={
                movement: (
                    20.0 * float(spec.get("beta", 1.0))
                    if spec.get("kind") == "boundary_in"
                    else 0.0
                )
                for movement, spec in net.urban_movements.items()
            },
            urban_link_storage=dict(net.urban_link_storage_veh),
            urban_arrival_buffer={},
            urban_storage_release_buffer={link: {} for link in net.urban_link_storage_veh},
        )

    def copy(self) -> "TrafficState":
        return copy.deepcopy(self)

    def ensure_freeway_lane_profile(self, net: NetworkConfig) -> None:
        for link in net.freeway_links:
            count = len(self.freeway_density.get(link, []))
            lanes = self.freeway_effective_lanes.get(link, [])
            if len(lanes) != count:
                self.freeway_effective_lanes[link] = [float(net.freeway_lanes) for _ in range(count)]

    def freeway_vehicle_count_by_link(self, net: NetworkConfig) -> Dict[str, List[float]]:
        self.ensure_freeway_lane_profile(net)
        out: Dict[str, List[float]] = {}
        for link in net.freeway_links:
            out[link] = [
                max(0.0, rho) * net.freeway_segment_length_km * max(lane, 1.0e-9)
                for rho, lane in zip(
                    self.freeway_density.get(link, []),
                    self.freeway_effective_lanes.get(link, []),
                )
            ]
        return out

    def refresh_freeway_flow(self, net: NetworkConfig) -> None:
        self.ensure_freeway_lane_profile(net)
        self.freeway_flow = {
            link: [
                max(0.0, rho) * max(0.0, speed) * max(0.0, lane)
                for rho, speed, lane in zip(
                    self.freeway_density.get(link, []),
                    self.freeway_speed.get(link, []),
                    self.freeway_effective_lanes.get(link, []),
                )
            ]
            for link in net.freeway_links
        }

    def total_freeway_vehicles(self, net: NetworkConfig) -> float:
        return float(
            self.freeway_segment_vehicles(net)
            + sum(self.ramp_queue.values())
            + sum(self.mainline_origin_queue.values())
        )

    def freeway_segment_vehicles(self, net: NetworkConfig) -> float:
        """본선 segment 내부 차량 수[veh]. ramp/origin queue는 포함하지 않는다."""
        return float(sum(sum(values) for values in self.freeway_vehicle_count_by_link(net).values()))

    def total_urban_vehicles(self, net=None) -> float:
        """urban 총 차량 수. net을 주면 링크 in-transit 점유까지 포함한다.

        그리드 라우팅 이후 urban 차량의 상당수가 movement 큐가 아니라 링크 transit에
        있으므로, leader objective처럼 '총 urban 차량'이 필요한 곳은 net을 넘겨야 한다
        (큐만 세면 그리드 과충전이 비용 0으로 보이는 왜곡이 생긴다)."""
        total = float(sum(self.urban_movement_queue.values())) if self.urban_movement_queue else (
            float(sum(self.urban_queue.values()) + sum(self.boundary_queue.values()))
        )
        if net is not None:
            # off-ramp 램프 storage(`OR_*_storage`)는 freeway로 재귀속(design 2026-06-17)하므로
            # urban 총 차량에서 제외한다. leg(off_ramp movement 점큐)는 urban에 유지된다.
            off_ramp_storage_links = set(net.off_ramp_storage_link.values())
            for link, capacity in net.urban_link_storage_veh.items():
                if link in off_ramp_storage_links:
                    continue
                total += max(0.0, capacity - self.urban_link_storage.get(link, capacity))
        return total

    def off_ramp_storage_occupancy_veh(self, net) -> float:
        """off-ramp 램프 storage(`OR_*_storage`)의 in-transit 점유[veh] 합.

        off-ramp 재귀속(design 2026-06-17)에 따라 이 점유는 urban이 아니라 freeway TTT/
        agent/누적으로 귀속된다. 램프 storage link 집합은 `net.off_ramp_storage_link`의 값.
        """
        total = 0.0
        for storage_link in set(net.off_ramp_storage_link.values()):
            capacity = net.urban_link_storage_veh.get(storage_link)
            if capacity is None:
                continue
            total += max(0.0, capacity - self.urban_link_storage.get(storage_link, capacity))
        return float(total)

    def uncontrolled_node_movement_queue_veh(self, net) -> float:
        """비통제 노드 stop-line movement queue 차량 수[veh]."""
        nodes = {str(node) for node in getattr(net, "uncontrolled_nodes", [])}
        if not nodes:
            return 0.0
        total = 0.0
        for movement, spec in net.urban_movements.items():
            if str(spec.get("intersection", "")) in nodes:
                total += max(0.0, self.urban_movement_queue.get(movement, 0.0))
        return float(total)

    def uncontrolled_node_storage_occupancy_veh(self, net) -> float:
        """비통제 노드에 접한 directed urban link 점유 차량 수[veh]."""
        nodes = {str(node) for node in getattr(net, "uncontrolled_nodes", [])}
        if not nodes:
            return 0.0
        links = set()
        for node, legs in getattr(net, "grid_node_legs", {}).items():
            node_id = str(node)
            for leg in legs.values():
                if leg.get("type") != "grid":
                    continue
                downstream = str(leg.get("node", ""))
                if node_id in nodes or downstream in nodes:
                    links.add(f"{node_id}_to_{downstream}")
        total = 0.0
        for link in links:
            capacity = net.urban_link_storage_veh.get(link)
            if capacity is None:
                continue
            total += max(0.0, capacity - self.urban_link_storage.get(link, capacity))
        return float(total)

    def uncontrolled_node_vehicles(self, net) -> float:
        """비통제 내부 노드 주변 차량 수[veh] = 점큐 + 접속 link in-transit 점유."""
        return float(
            self.uncontrolled_node_movement_queue_veh(net)
            + self.uncontrolled_node_storage_occupancy_veh(net)
        )

    def boundary_in_queue_vehicles(self, net) -> float:
        """boundary_in movement(유입 대기) 큐 점유[veh] 합.

        Step D 이후 leader objective에는 더하지 않고, 후보가 경계 대기를 얼마나
        만드는지 해석하기 위한 diagnostic으로만 보고한다.
        """
        total = 0.0
        for movement, spec in net.urban_movements.items():
            if str(spec.get("kind", "")) == "boundary_in":
                total += max(0.0, self.urban_movement_queue.get(movement, 0.0))
        return float(total)

    def boundary_leg_vehicles(self, net) -> float:
        """Leader base에서 제외할 외부 boundary leg 점유[veh].

        boundary_in/out movement queue와 boundary_out sink storage는 외부 네트워크 leg로 보며,
        on-ramp/off-ramp 및 내부 grid link는 freeway-urban coupling 비용이므로 제외하지 않는다.
        """
        boundary_kinds = {"boundary_in", "boundary_out"}
        total = 0.0
        boundary_storage_links = set()
        for movement, spec in net.urban_movements.items():
            kind = str(spec.get("kind", ""))
            if kind in boundary_kinds:
                total += max(0.0, self.urban_movement_queue.get(movement, 0.0))
            if kind == "boundary_out":
                receiving = str(spec.get("receiving_link", ""))
                if receiving:
                    boundary_storage_links.add(receiving)
        for link in boundary_storage_links:
            capacity = net.urban_link_storage_veh.get(link)
            if capacity is not None:
                total += max(0.0, capacity - self.urban_link_storage.get(link, capacity))
        return float(total)

    def objective_urban_vehicles(self, net, exclude_boundary_legs: bool = True) -> float:
        """Leader state-accumulation base용 urban 차량 수[veh]."""
        total = self.total_urban_vehicles(net)
        if exclude_boundary_legs:
            total -= self.boundary_leg_vehicles(net)
        return float(max(0.0, total))

    def protected_accumulation_veh(self, net) -> float:
        """보호영역 누적 N_P = 링크 in-transit 점유(cap−available) + 보호영역 내부 movement 대기열.

        진입 대기열(boundary_in 게이트 큐)·on-ramp 접근 대기열(x_on)은 경계 미터링 큐이므로
        제외한다. 그리드 라우팅 도입으로 내부 교차로 대기열(internal/boundary_out/off_ramp
        kind)이 생겼고, 이들은 물리적으로 보호영역 안에 있으므로 N_P에 포함한다.

        단 off-ramp 램프 storage(`OR_*_storage`)는 freeway로 재귀속하므로 N_P에서 제외한다
        (design 2026-06-17). off_ramp movement 점큐(leg)는 정지선 대기라 urban이므로 유지."""
        total = 0.0
        off_ramp_storage_links = set(net.off_ramp_storage_link.values())
        for link, capacity in net.urban_link_storage_veh.items():
            if link in off_ramp_storage_links:
                continue
            total += max(0.0, capacity - self.urban_link_storage.get(link, capacity))
        protected_kinds = {"internal", "boundary_out", "off_ramp"}
        for movement, spec in net.urban_movements.items():
            if str(spec.get("kind", "")) in protected_kinds:
                total += max(0.0, self.urban_movement_queue.get(movement, 0.0))
        return float(total)

    def boundary_vector(self) -> List[float]:
        return [float(v) for v in self.boundary_queue.values()]


def segment_vsl(control: "ControlAction", link: str, i: int, cfg: ExperimentConfig) -> float:
    """freeway link의 segment i에 적용할 VSL 값을 읽는다(Option C per-segment).

    조회 순서: segment 키 `{link}__seg{i}` → link 키 `{link}` → vsl_set 최대값(no-VSL).
    segment 키가 없으면 link 키로 fallback해 기존 link-uniform 동작과 비트 동일하다.
    """
    fallback = control.vsl.get(link, max(cfg.freeway_follower.vsl_set))
    return float(control.vsl.get(f"{link}__seg{i}", fallback))


@dataclass
class ControlAction:
    N_P_star: float = 0.0
    N_UF_star: float = 0.0
    ramp_metering: Dict[str, float] = field(default_factory=dict)
    vsl: Dict[str, float] = field(default_factory=dict)
    green_times: Dict[str, float] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)
    inflow_outflow_allocation: Dict[str, float] = field(default_factory=dict)
    infeasibility: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "ControlAction":
        """후보 평가 중 이전 control 객체가 오염되지 않도록 dict 필드를 복사한다."""
        return ControlAction(
            N_P_star=float(self.N_P_star),
            N_UF_star=float(self.N_UF_star),
            ramp_metering=dict(self.ramp_metering),
            vsl=dict(self.vsl),
            green_times=dict(self.green_times),
            offsets=dict(self.offsets),
            inflow_outflow_allocation=dict(self.inflow_outflow_allocation),
            infeasibility=dict(self.infeasibility),
            diagnostics=dict(self.diagnostics),
        )

    @classmethod
    def uncontrolled(cls, cfg: ExperimentConfig) -> "ControlAction":
        """고정 신호와 자유 방출을 사용하는 물리적 no-control action을 만든다.

        allocation을 비워 두면 urban plant가 movement saturation flow를 사용한다.
        따라서 equal green fraction이 service capacity에 정확히 한 번만 적용된다.
        """
        net = cfg.network
        phase_green = net.effective_green_total / 2.0
        return cls(
            ramp_metering={r: net.ramp_capacity_veh_h[r] for r in net.ramps},
            vsl={link: max(cfg.freeway_follower.vsl_set) for link in net.freeway_links},
            green_times={
                f"{signal}_{phase}": phase_green
                for signal in net.signals
                for phase in ("p1", "p2")
            },
            offsets={signal: 0.0 for signal in net.signals},
            inflow_outflow_allocation={},
        )

    @classmethod
    def fixed(cls, cfg: ExperimentConfig) -> "ControlAction":
        net = cfg.network
        green = {}
        phase_green = net.effective_green_total / 2.0
        for signal in net.signals:
            green[f"{signal}_p1"] = phase_green
            green[f"{signal}_p2"] = phase_green
        # allocation은 perimeter(경계/램프) 제어 전용 신호다. 내부 movement는 사전충전하지
        # 않는다(green×saturation으로만 제어; allocation으로 throttle되면 안 됨).
        perimeter_kinds = {"boundary_in", "off_ramp", "boundary_out", "on_ramp"}
        allocation = {
            movement: net.movement_capacity_veh_h * 0.5
            for movement, spec in net.urban_movements.items()
            if spec.get("kind") in perimeter_kinds
        }
        # legacy link-level allocation은 movement-level 합과 일치시킨다(분산 coordinator 합산과 정합).
        for link in net.boundary_in_links:
            allocation[link] = sum(
                allocation[movement]
                for movement, spec in net.urban_movements.items()
                if spec.get("origin") == link and spec.get("kind") == "boundary_in"
            )
        for link in net.boundary_out_links:
            allocation[link] = sum(
                allocation[movement]
                for movement, spec in net.urban_movements.items()
                if spec.get("destination") == link and spec.get("kind") == "boundary_out"
            )
        return cls(
            ramp_metering={r: net.ramp_capacity_veh_h[r] for r in net.ramps},
            vsl={link: max(cfg.freeway_follower.vsl_set) for link in net.freeway_links},
            green_times=green,
            offsets={signal: 0.0 for signal in net.signals},
            inflow_outflow_allocation=allocation,
        )

    def control_vector(self, cfg: ExperimentConfig) -> List[float]:
        net = cfg.network
        return (
            [self.ramp_metering.get(r, 0.0) for r in net.ramps]
            + [self.vsl.get(link, max(cfg.freeway_follower.vsl_set)) for link in net.freeway_links]
            + [self.green_times.get(f"{s}_p1", 0.0) for s in net.signals]
            + [self.offsets.get(s, 0.0) for s in net.signals]
            + [self.inflow_outflow_allocation.get(m, 0.0) for m in net.movement_links]
            + [self.inflow_outflow_allocation.get(m, 0.0) for m in net.urban_movements]
        )


@dataclass
class EvaluationResult:
    metrics: Dict[str, float]
    improvement_pct: float
    passed: bool
    control_validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticResult:
    failure_modes: List[str]
    suggestions: List[str]
    dominant_failure_mode: str = "none"


def mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else default
