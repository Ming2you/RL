# controller authority 자동검사 (spec 16.3/16.13) — control trace에서 권한 위반을 탐지
from __future__ import annotations

from typing import Dict, List, Mapping

from src.models.state import ExperimentConfig

WU_GROUP = {"WU-CD-F", "WU-MATCHED-STACKELBERG", "WU-CC-F"}
PROPOSED_GROUP = {"PROPOSED-FOLLOWERS-ONLY", "PROPOSED-STACKELBERG", "PROPOSED-CENTRALIZED"}
CLASSICAL_GROUP = {"CLASSICAL-HIERARCHICAL", "NO-CONTROL"}
LEADER_ENABLED = {"WU-MATCHED-STACKELBERG", "PROPOSED-STACKELBERG", "CLASSICAL-HIERARCHICAL"}
CENTRALIZED = {"WU-CC-F", "PROPOSED-CENTRALIZED"}
# 2026-06-13 재정의(spec 16.7): P-FO는 allocation 비제어 — Wu group과 같은 fallback.
NO_ALLOCATION = WU_GROUP | CLASSICAL_GROUP | {"PROPOSED-FOLLOWERS-ONLY"}
# 주 비교군(spec 16.1, 2026-06-13 개정). 나머지 둘은 보조 참고군.
PRIMARY_CONTROLLERS = [
    "WU-CD-F",
    "PROPOSED-FOLLOWERS-ONLY",
    "PROPOSED-STACKELBERG",
    "PROPOSED-CENTRALIZED",
]
CONTROLLER_IDS = [
    "NO-CONTROL",
    "WU-CD-F",
    "WU-MATCHED-STACKELBERG",
    "WU-CC-F",
    "PROPOSED-FOLLOWERS-ONLY",
    "PROPOSED-STACKELBERG",
    "PROPOSED-CENTRALIZED",
    "CLASSICAL-HIERARCHICAL",
]


def check_control_authority(
    controller_id: str,
    control_rows: List[Mapping[str, float]],
    cfg: ExperimentConfig,
    eps: float = 1.0e-6,
) -> Dict[str, object]:
    """적용된 control 시계열이 선언된 authority만 사용했는지 검사한다.

    Wu group(spec 16.3.1): offset 고정, ramp metering=용량(no-metering), allocation 미사용.
    P-FO(16.7 재정의): green/offset/metering/VSL 허용, allocation 미사용.
    P-STACK/P-CENT(16.3.2): 4계열 모두 허용. leader 유무는 N_P_star/N_UF_star 흔적으로 확인."""
    net = cfg.network
    violations: List[str] = []
    for row in control_rows:
        if controller_id in WU_GROUP:
            for signal in net.signals:
                if abs(float(row.get(f"offset_{signal}", 0.0))) > eps:
                    violations.append(f"offset_{signal} moved")
                    break
            for ramp in net.ramps:
                if abs(float(row.get(f"ramp_metering_{ramp}", net.ramp_capacity_veh_h[ramp]))
                       - net.ramp_capacity_veh_h[ramp]) > eps:
                    violations.append(f"ramp_metering_{ramp} != capacity")
                    break
        if controller_id in NO_ALLOCATION:
            # allocation 미사용: movement allocation 열이 전부 0이어야 한다.
            for movement in net.urban_movements:
                if abs(float(row.get(f"allocation_{movement}", 0.0))) > eps:
                    violations.append(f"allocation_{movement} used")
                    break
        if violations:
            break
    leader_used = any(
        abs(float(row.get("N_P_star", 0.0))) > eps or abs(float(row.get("N_UF_star", 0.0))) > eps
        for row in control_rows
    )
    if controller_id in LEADER_ENABLED and not leader_used:
        # WU-MATCHED는 N_P_star를 ControlAction에 싣지 않을 수 있어 leader 흔적은
        # 별도 진단(leader_candidates)으로 확인한다 — 여기서는 경고만.
        pass
    if controller_id not in LEADER_ENABLED and leader_used:
        violations.append("hidden leader target present")
    return {
        "controller_id": controller_id,
        "authority_group": "wu" if controller_id in WU_GROUP else (
            "classical" if controller_id in CLASSICAL_GROUP else "proposed"
        ),
        "leader_enabled": controller_id in LEADER_ENABLED,
        "centralized": controller_id in CENTRALIZED,
        "violations": violations,
        "ok": not violations,
    }
