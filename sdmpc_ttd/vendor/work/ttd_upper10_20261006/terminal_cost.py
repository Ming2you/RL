"""역사 mfd_far_cost_to_go를 그대로 읽는 terminal cost와 player 귀속.

Spec 03/04/06/12/15 적용. terminal H 이후 추가 교통 rollout을 하지 않는다.
원 함수의 gate/계수/분기는 보존하고, AD에서는 기존 Dual 산술로 모든 연결 미분을
전파한다. V는 경험적인 잔여 배수 비용이며 정확한 무한지평 value나 안정성 인증이 아니다.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap
from typing import Any

_FUNCTIONS: dict[bool, Any] = {}


class _ExposeRegions(ast.NodeTransformer):
    """원 total 반환식과 별도로 이미 계산된 두 지역 항만 노출한다."""
    def visit_Return(self, node):
        # 내부 _lane_at 반환식은 그대로 둔다. 최상위 두 반환식만 모양으로 구별한다.
        if isinstance(node.value, ast.Constant) and node.value.value == 0.0:
            node.value = ast.parse("dict(total=0.0, urban=0.0, freeway=0.0)", mode="eval").body
        elif isinstance(node.value, ast.BinOp) and ast.unparse(node.value) == "w * (w_urban * far_urban + fw_contrib)":
            node.value = ast.Call(func=ast.Name(id="dict", ctx=ast.Load()), args=[], keywords=[
                ast.keyword(arg="total", value=node.value),
                ast.keyword(arg="urban", value=ast.parse("w * w_urban * far_urban", mode="eval").body),
                ast.keyword(arg="freeway", value=ast.parse("w * fw_contrib", mode="eval").body),
            ])
        return node


def _function(ad: bool):
    if ad not in _FUNCTIONS:
        from src.controllers.stackelberg_mpc import mfd_far_cost_to_go
        tree = ast.parse(textwrap.dedent(inspect.getsource(mfd_far_cost_to_go)))
        tree = _ExposeRegions().visit(tree)
        namespace = dict(mfd_far_cost_to_go.__globals__)
        if ad:
            from sparse.engine import Transform, load_modules
            from sparse.dual import float_keep, minimum, maximum
            load_modules()
            tree = Transform().visit(tree)
            namespace.update(_ad_float=float_keep, _ad_min=minimum, _ad_max=maximum)
        ast.fix_missing_locations(tree)
        exec(compile(tree, str(inspect.getfile(mfd_far_cost_to_go)) + "::terminal_regions", "exec"), namespace)
        _FUNCTIONS[ad] = namespace["mfd_far_cost_to_go"]
    return _FUNCTIONS[ad]


def terminal_components(cfg, state, ownership, modules=None, prev_state=None):
    """원 V와 region/player 항을 반환한다. cfg·state·소유권은 변경하지 않는다.

    전역 reservoir V는 separable하지 않다. 각 region V를 terminal 실제 물리 재고
    비율로 player에 귀속한다. V_i=V_R*n_i/sum(n_j), 따라서 합은 원 V 그대로다.
    비율까지 미분하므로 외부효과를 계산할 때 귀속 변화의 미분을 누락하지 않는다.
    """
    ad = modules is not None
    if ad:
        ledger_class = modules["src.simulation.player_cost_accounting"].PlayerCostLedger
        from sparse.dual import primal
    else:
        from src.simulation.player_cost_accounting import PlayerCostLedger as ledger_class
        primal = float
    regional = _function(ad)(cfg, state, prev_state)
    ledger = ledger_class(cfg, ownership)
    ids = ownership.active_player_ids
    stocks = {region: {key: 0.0 for key in ids} for region in ("F", "U")}
    # 두 inventory는 core/buffer/ramp와 movement/storage로 물리 공간이 겹치지 않는다.
    for stage in ("freeway", "urban"):
        for _, (bucket, region, count) in ledger.inventory(stage, state).items():
            owner = bucket if bucket in stocks[region] else ownership.passive_cost_owners.get(bucket)
            if owner not in stocks[region]:
                raise ValueError(f"Unassigned terminal stock owner: {bucket}")
            stocks[region][owner] += count
    costs = {key: 0.0 for key in ids}
    for region, value in (("F", regional["freeway"]), ("U", regional["urban"])):
        denominator = sum(stocks[region].values())
        if primal(denominator) <= 0.0:
            if abs(primal(value)) > 1e-12:
                raise ValueError(f"Nonzero terminal value without physical {region} stock")
            continue
        for key in ids:
            costs[key] += value * stocks[region][key] / denominator
    residual = primal(sum(costs.values())) - primal(regional["total"])
    if abs(residual) > 1e-8:
        raise ValueError(f"Terminal player accounting residual {residual}")
    return dict(total=regional["total"], urban=regional["urban"], freeway=regional["freeway"],
                player_costs=costs, regional_player_stock=stocks, accounting_residual=residual)


def evaluate_terminal(cfg, state, ownership, modules=None, prev_state=None):
    """S-DMPC 호출용: raw/Dual veh*h player terminal 비용 사전."""
    return terminal_components(cfg, state, ownership, modules=modules, prev_state=prev_state)["player_costs"]


def metadata(cfg=None):
    from src.controllers.stackelberg_mpc import mfd_far_cost_to_go
    path = Path(inspect.getfile(mfd_far_cost_to_go))
    defaults = dict(leader_mfd_far_enabled=False, leader_mfd_far_weight=1.0,
        leader_mfd_far_urban_weight=1.0, leader_mfd_far_freeway_weight=1.0,
        leader_mfd_far_state_aware=False, leader_mfd_far_real_speed=False,
        leader_mfd_far_freeflow_offset=False, leader_mfd_far_ncrit=1700.0,
        leader_mfd_far_g_free=640.0, leader_mfd_far_g_cong=500.0, leader_mfd_far_g_fw=300.0,
        leader_urban_clf_enabled=False, leader_urban_clf_gated=True,
        leader_urban_clf_nref=450.0, leader_urban_clf_ncrit=550.0,
        leader_urban_clf_njam=1540.0, leader_urban_clf_pmax=640.0,
        leader_urban_clf_gmin=60.0, leader_urban_clf_ttail=0.5)
    effective = {key: getattr(cfg.mpc, key, value) for key, value in defaults.items()} if cfg is not None else None
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        function="mfd_far_cost_to_go", extra_horizon_rollouts=0,
        formula="J=sum_h TTT_h + V_original(x_H,x_(H-1)); V_i=V_region*n_i/sum_j n_j",
        units="nominal veh*h of the historical implementation", configured_gate_and_weights_preserved=True,
        effective_terminal_parameters=effective,
        allocation="terminal physical stock per existing PlayerCostOwnership; passive mapped once",
        allocation_derivative_included=True,
        caveats=["Original empirical drain approximation is not a solved infinite-horizon value function",
                 "Original near-stock V does not separately model all buffer and off-ramp storage discharge",
                 "Original g_fw convention mixes calibrated vehicles-per-interval with state-aware flow in vehicles/hour; its scaling is preserved and is not certified as an exact physical drainage-time estimate",
                 "Aggregate reservoir cost is assigned proportionally; this is not an independently calibrated local value",
                 "Original hard branches/min/max retained; no smoothing or convergence certification"])
