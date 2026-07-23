# 그리드 leg 인접 그래프에서 directed link·movement(o,s,d)·turning ratio β를 자동 유도하는 모듈 (docs/grid_routing_proposal.md)
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

# leg 방위와 정반대(직진) 매핑 — proposal §4: 직진 = 들어온 approach의 정반대 leg.
LEG_DIRECTIONS = ("N", "S", "E", "W")
OPPOSITE_LEG = {"N": "S", "S": "N", "E": "W", "W": "E"}
NS_AXIS = {"N", "S"}

# 램프 leg(D·F의 S)는 4갈래: 나가는 on_ramp 2(W/E) + 들어오는 off_ramp 2(W/E) — proposal §1.
RAMP_SIDES = ("W", "E")


def default_grid_node_legs() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """proposal §1 leg 표를 그대로 코드화한 기본 토폴로지.

    leg spec 종류:
      {"type": "grid", "node": X}                      — 내부 도로(양방향 directed 2개)
      {"type": "boundary", "in": .., "out": .., "out_link": ..} — 경계 게이트(in/out 링크)
      {"type": "ramp", "on": {side: ramp}, "off": {side: off_ramp}} — D·F 램프 leg
    """
    return {
        "A": {
            "N": {"type": "boundary", "in": "in_A_top", "out": "out_A_top", "out_link": "A_top_out"},
            "S": {"type": "grid", "node": "D"},
            "E": {"type": "grid", "node": "B"},
            "W": {"type": "boundary", "in": "in_A_left", "out": "out_A_left", "out_link": "A_left_out"},
        },
        "B": {
            "N": {"type": "boundary", "in": "in_B_top", "out": "out_B_top", "out_link": "B_top_out"},
            "S": {"type": "grid", "node": "E"},
            "E": {"type": "grid", "node": "C"},
            "W": {"type": "grid", "node": "A"},
        },
        "C": {
            "N": {"type": "boundary", "in": "in_C_top", "out": "out_C_top", "out_link": "C_top_out"},
            "S": {"type": "grid", "node": "F"},
            "E": {"type": "boundary", "in": "in_C_right", "out": "out_C_right", "out_link": "C_right_out"},
            "W": {"type": "grid", "node": "B"},
        },
        "D": {
            "N": {"type": "grid", "node": "A"},
            "S": {"type": "ramp", "on": {"W": "R_D_W", "E": "R_D_E"}, "off": {"W": "OR_D_W", "E": "OR_D_E"}},
            "E": {"type": "grid", "node": "E"},
            "W": {"type": "boundary", "in": "in_D_left", "out": "out_D_left", "out_link": "D_left_out"},
        },
        "E": {
            "N": {"type": "grid", "node": "B"},
            "E": {"type": "grid", "node": "F"},
            "W": {"type": "grid", "node": "D"},
        },
        "F": {
            "N": {"type": "grid", "node": "C"},
            "S": {"type": "ramp", "on": {"W": "R_F_W", "E": "R_F_E"}, "off": {"W": "OR_F_W", "E": "OR_F_E"}},
            "E": {"type": "boundary", "in": "in_F_right", "out": "out_F_right", "out_link": "F_right_out"},
            "W": {"type": "grid", "node": "E"},
        },
    }


def internal_link_name(from_node: str, to_node: str) -> str:
    return f"{from_node}_to_{to_node}"


def internal_links(grid_node_legs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[str]:
    """leg 인접에서 내부 directed link(7도로 × 2방향 = 14개) 이름을 유도한다."""
    links: List[str] = []
    for node, legs in grid_node_legs.items():
        for leg in legs.values():
            if leg.get("type") == "grid":
                links.append(internal_link_name(node, str(leg["node"])))
    return sorted(links)


def _approach_tokens(legs: Mapping[str, Mapping[str, Any]]) -> List[Tuple[str, str]]:
    """노드의 incoming approach (token, leg방위) 목록. 램프 leg는 off_ramp 2개가 별도 incoming."""
    out: List[Tuple[str, str]] = []
    for direction, leg in legs.items():
        kind = leg.get("type")
        if kind in {"grid", "boundary"}:
            out.append((direction, direction))
        elif kind == "ramp":
            for side in RAMP_SIDES:
                if side in leg.get("off", {}):
                    out.append((f"off{side}", direction))
    return out


def _outgoing_leg_dirs(legs: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """나가는 leg 방위 목록(램프 leg는 on_ramp가 있으면 한 leg로 취급)."""
    out: List[str] = []
    for direction, leg in legs.items():
        kind = leg.get("type")
        if kind in {"grid", "boundary"}:
            out.append(direction)
        elif kind == "ramp" and leg.get("on"):
            out.append(direction)
    return out


def derive_turning_ratios(
    grid_node_legs: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """직진우대 β 자동 유도 — proposal §4.

    - 직진(들어온 leg의 정반대 방위)이 있으면 β(직진 leg)=0.5, 나머지 leg 균등.
    - 직진 leg가 없으면(E의 N approach) 가용 leg 균등. U-turn(같은 leg) 제외, Σ_d β=1.
    - 램프 leg가 받는 몫은 on_ramp W/E에 균등 분할(각 절반) — d token: onW/onE.
    """
    ratios: Dict[str, Dict[str, Dict[str, float]]] = {}
    for node, legs in grid_node_legs.items():
        node_ratios: Dict[str, Dict[str, float]] = {}
        outgoing_dirs = _outgoing_leg_dirs(legs)
        for token, leg_dir in _approach_tokens(legs):
            available = [d for d in outgoing_dirs if d != leg_dir]
            if not available:
                continue
            straight = OPPOSITE_LEG.get(leg_dir, "")
            leg_share: Dict[str, float] = {}
            if straight in available:
                others = [d for d in available if d != straight]
                leg_share[straight] = 0.5 if others else 1.0
                for d in others:
                    leg_share[d] = 0.5 / len(others)
            else:
                for d in available:
                    leg_share[d] = 1.0 / len(available)
            beta: Dict[str, float] = {}
            for d, share in leg_share.items():
                leg = legs[d]
                if leg.get("type") == "ramp":
                    on_sides = [side for side in RAMP_SIDES if side in leg.get("on", {})]
                    for side in on_sides:
                        beta[f"on{side}"] = share / len(on_sides)
                else:
                    beta[d] = share
            node_ratios[token] = beta
        ratios[node] = node_ratios
    return ratios


def _token_leg_dir(token: str) -> str:
    """approach/exit token의 leg 방위(phase 축 판정용). off*/on*은 램프 leg(S)다."""
    if token.startswith("off") or token.startswith("on"):
        return "S"
    return token


def build_urban_movements(
    grid_node_legs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    turning_ratios: Mapping[str, Mapping[str, Mapping[str, float]]],
    signals: List[str],
    ramp_to_freeway: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """movement (o,s,d)를 토폴로지+β에서 자동 생성한다 — proposal §3.

    - phase = incoming approach 축: N·S → p1, E·W → p2 (2-phase 확정, 램프 movement 포함).
    - E(비통제)는 phase=""(green=1 상당)로 신호 없이 통과.
    - kind 우선순위: origin이 경계 게이트 → boundary_in, origin이 off_ramp → off_ramp,
      exit이 on_ramp → on_ramp, exit이 경계 → boundary_out, 그 외 internal.
    """
    signal_set = set(signals)
    movements: Dict[str, Dict[str, Any]] = {}
    for node, legs in grid_node_legs.items():
        node_ratios = turning_ratios.get(node, {})
        for token, leg_dir in _approach_tokens(legs):
            beta_map = node_ratios.get(token, {})
            approach_leg = legs[leg_dir]
            for exit_token, beta in beta_map.items():
                exit_leg_dir = _token_leg_dir(exit_token)
                exit_leg = legs[exit_leg_dir]
                spec: Dict[str, Any] = {
                    "intersection": node,
                    "approach": token,
                    "exit": exit_token,
                    "beta": float(beta),
                    "signal": node,
                }
                # phase: incoming approach 축으로 결정(2-phase NS/EW), E는 비통제.
                if node in signal_set:
                    axis_dir = _token_leg_dir(token)
                    spec["phase"] = f"{node}_p1" if axis_dir in NS_AXIS else f"{node}_p2"
                else:
                    spec["phase"] = ""
                # origin(legacy): 게이트 in링크 / off_ramp 이름 / 내부 incoming link.
                if approach_leg.get("type") == "boundary":
                    spec["origin"] = str(approach_leg["in"])
                elif approach_leg.get("type") == "ramp":
                    side = token[len("off"):]
                    spec["origin"] = str(approach_leg["off"][side])
                    spec["off_ramp"] = str(approach_leg["off"][side])
                else:
                    spec["origin"] = internal_link_name(str(approach_leg["node"]), node)
                # destination/receiving_link: exit leg 타입별.
                if exit_leg.get("type") == "grid":
                    receiving = internal_link_name(node, str(exit_leg["node"]))
                    spec["destination"] = receiving
                    spec["receiving_link"] = receiving
                elif exit_leg.get("type") == "boundary":
                    spec["destination"] = str(exit_leg["out"])
                    spec["receiving_link"] = str(exit_leg["out_link"])
                else:  # ramp exit (onW/onE)
                    side = exit_token[len("on"):]
                    ramp = str(exit_leg["on"][side])
                    spec["ramp"] = ramp
                    spec["destination"] = str(ramp_to_freeway.get(ramp, ramp))
                    spec["receiving_link"] = f"{node}_R_{side}"
                # kind 분류(우선순위: origin 경계 > origin off_ramp > exit ramp > exit 경계 > internal).
                if approach_leg.get("type") == "boundary":
                    spec["kind"] = "boundary_in"
                elif approach_leg.get("type") == "ramp":
                    spec["kind"] = "off_ramp"
                elif exit_leg.get("type") == "ramp":
                    spec["kind"] = "on_ramp"
                elif exit_leg.get("type") == "boundary":
                    spec["kind"] = "boundary_out"
                else:
                    spec["kind"] = "internal"
                movements[f"{node}_{token}_to_{exit_token}"] = spec
    return movements


def approach_sources(
    grid_node_legs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    off_ramp_storage_link: Mapping[str, str],
) -> Dict[str, Tuple[str, str]]:
    """arrival buffer key(=approach source link) → (교차로, approach token) 매핑.

    내부 링크 X_to_Y는 Y에 leg방위로 도착, off-ramp storage 링크는 offW/offE로 도착.
    게이트 수요는 in링크 이름으로 주입된다.
    """
    out: Dict[str, Tuple[str, str]] = {}
    for node, legs in grid_node_legs.items():
        for direction, leg in legs.items():
            kind = leg.get("type")
            if kind == "grid":
                out[internal_link_name(str(leg["node"]), node)] = (node, direction)
            elif kind == "boundary":
                out[str(leg["in"])] = (node, direction)
            elif kind == "ramp":
                for side, off_ramp in leg.get("off", {}).items():
                    storage = off_ramp_storage_link.get(str(off_ramp), f"{off_ramp}_storage")
                    out[storage] = (node, f"off{side}")
    return out
