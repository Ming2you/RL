"""명세03/04/12/15: 원 inequality S-DMPC의 목적만 TTT-alpha*TTD로 격리 변경.

원 소스와 물리 모델의 식은 보존한다. 명시적인 objective 필드를 갖는 평가값을
추가하고, 최적화에 사용하는 속성만 AST로 치환한 독립 클래스 계층을 만든다.
TTT 계측 필드와 물리/제어/budget 검사는 원 단위를 유지한다.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import fields, make_dataclass
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import numpy as np


ATTRIBUTES = {
    'total_ttt': 'objective_value',
    'player_costs': 'objective_player_costs',
    'cost_vector': 'objective_cost_vector',
    'gradients': 'objective_gradients',
}
LABELS = {'ttt': 'objective_value'}


class ObjectiveTransform(ast.NodeTransformer):
    """최적화 계층 안에서만 필드와 로그 이름을 변경한다."""
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        node.attr = ATTRIBUTES.get(node.attr, node.attr)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = LABELS.get(node.value, node.value.replace('TTT', 'objective'))
        return node

    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if node.arg:
            node.arg = LABELS.get(node.arg, node.arg.replace('TTT', 'objective'))
        return node


def _clone(symbol, replacements, manifest):
    """원 함수의 제어 흐름을 그대로 컴파일하고 super의 새 클래스 closure도 유지한다."""
    path = Path(inspect.getfile(symbol))
    source = textwrap.dedent(inspect.getsource(symbol))
    tree = ObjectiveTransform().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    namespace = dict(vars(sys.modules[symbol.__module__]))
    namespace.update(replacements)
    code = ast.unparse(tree)
    exec(compile(tree, str(path) + ':TTT_TTD_objective_only', 'exec'), namespace)
    manifest.append(dict(symbol=symbol.__module__ + '.' + symbol.__name__,
                         source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         generated_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                         generated_source=code))
    return namespace[symbol.__name__]


def create_controller_class(alpha=0., objective='ttd', max_candidates=10):
    """역사 runtime 이후 호출. TTD의 alpha[h/km]는 시나리오 전체 NC로 고정한다."""
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('alpha must be finite and nonnegative')
    if objective not in ('ttd', 'ctg') or (objective == 'ctg' and alpha != 0.):
        raise ValueError('objective must be ttd or ctg; ctg requires alpha=0')
    from ttd_accounting import DistanceCapture, install_scalar, install_namespace
    from src.controllers.player_sensitivity_dmpc import PlayerSensitivityDMPC, PlayerRolloutEvaluation
    import prox_controller
    import fixed_controller
    import controller as selected_controller
    import central_controller
    import exception_controller
    import anchor_controller
    import reused_model
    from fixed_policy import contract
    def experiment_contract():
        # 물리 budget 정의는 유지하고 고정 3으로 남아 있던 표시만 실제 상한으로 맞춘다.
        return dict(contract(), budget_candidate_solves=max_candidates)
    from sparse.dual import primal, derivative, Dual

    install_scalar()
    manifest = []
    extra = [('objective_value', float), ('objective_player_costs', dict),
             ('objective_cost_vector', np.ndarray), ('total_ttd_veh_km', float),
             ('player_ttd_veh_km', dict), ('regional_ttd_veh_km', dict),
             ('terminal_cost_veh_h', float), ('player_terminal_costs_veh_h', dict)]
    ObjectiveEvaluation = make_dataclass('ObjectiveEvaluation', extra, bases=(PlayerRolloutEvaluation,))

    def terminal_cfg(cfg):
        # 역사 물리 config는 변경하지 않는다. terminal 계산용 복사에 gate와 공통 가중치만 켠다.
        value = copy.deepcopy(cfg)
        value.mpc.leader_mfd_far_enabled = True
        value.mpc.leader_mfd_far_weight = 1.0
        return value

    def terminal_values(cfg, states, ownership, modules=None):
        if objective == 'ttd':
            return {pid: 0. for pid in ownership.active_player_ids}
        from terminal_cost import evaluate_terminal
        return evaluate_terminal(cfg, states[-1], ownership, modules=modules,
                                 prev_state=states[-2] if len(states) > 1 else None)

    class ObjectiveEvaluator(PlayerSensitivityDMPC):
        def __init__(self, cfg, options):
            super().__init__(cfg, options)
            self.terminal_cfg = terminal_cfg(cfg)

        def evaluate_control(self, *args, **kwargs):
            # 한 번의 원 rollout에서 TTT와 TTD를 함께 관측한다. 가격만으로 추가 예측하지 않는다.
            with DistanceCapture(self.cfg, self.ownership) as capture:
                raw = super().evaluate_control(*args, **kwargs)
            distance = {pid: primal(capture.player_ttd.get(pid, 0.)) for pid in raw.player_costs}
            total_ttd = sum(distance.values())
            terminal = {pid: primal(value) for pid, value in
                        terminal_values(self.terminal_cfg, raw.states, self.ownership).items()}
            terminal_sum = sum(terminal.values())
            costs = {pid: raw.player_costs[pid] - alpha * distance[pid] + terminal[pid] for pid in raw.player_costs}
            objective_value = raw.total_ttt - alpha * total_ttd + terminal_sum
            if abs(sum(costs.values()) - objective_value) > 1e-8:
                raise RuntimeError('TTT-TTD player objective accounting mismatch')
            values = {field.name: getattr(raw, field.name) for field in fields(raw)}
            values['diagnostics'] = dict(raw.diagnostics, objective_contract='TTT-alpha*TTD' if objective == 'ttd' else 'TTT+V_terminal',
                alpha_h_per_km=alpha, raw_total_ttt_veh_h=raw.total_ttt,
                total_ttd_veh_km=total_ttd, objective_value_veh_h=objective_value,
                terminal_cost_veh_h=terminal_sum,
                objective_accounting_residual_veh_h=abs(sum(costs.values()) - objective_value))
            return ObjectiveEvaluation(**values, objective_value=objective_value,
                objective_player_costs=costs, objective_cost_vector=np.asarray(list(costs.values())),
                total_ttd_veh_km=total_ttd, player_ttd_veh_km=distance,
                regional_ttd_veh_km={'freeway': primal(capture.freeway_ttd), 'urban': primal(capture.urban_ttd)},
                terminal_cost_veh_h=terminal_sum, player_terminal_costs_veh_h=terminal)

    class ObjectiveEngine(central_controller.RecordedEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_namespace(self.modules)
            self.terminal_cfg = terminal_cfg(self.cfg)

        def evaluate(self, *args, **kwargs):
            # 같은 활성 경로의 Dual 산술로 TTD 총미분도 전파한다. budget Jacobian은 변경하지 않는다.
            with DistanceCapture(self.cfg, self.ownership) as capture:
                raw = super().evaluate(*args, **kwargs)
            ids = self.ownership.active_player_ids
            distances = [capture.player_ttd.get(pid, 0.) for pid in ids]
            n = len(args[3].axes)
            distance_jac = np.array([[derivative(value).get(j, 0.) for j in range(n)] for value in distances])
            distance_vector = np.array([primal(value) for value in distances])
            terminal = terminal_values(self.terminal_cfg, raw.states, self.ownership, self.modules)
            terminal_vector = np.array([primal(terminal[pid]) for pid in ids])
            terminal_jac = np.array([[derivative(terminal[pid]).get(j, 0.) for j in range(n)] for pid in ids])
            objective_vector = raw.cost_vector - alpha * distance_vector + terminal_vector
            objective_gradients = raw.gradients - alpha * distance_jac + terminal_jac
            diagnostics = copy.deepcopy(raw.diagnostics)
            # TTD에만 연결된 kink도 기존 FD fallback 판정에 포함한다. 평활화는 하지 않는다.
            additions = distances if objective == 'ttd' and alpha else list(terminal.values()) if objective == 'ctg' else []
            if additions:
                traces = [value.trace for value in additions if isinstance(value, Dual)]
                if traces:
                    extra_trace = traces[0].result(additions)
                    for key in ('output_connected_exact_tie_axes', 'output_connected_stencil_crossing_axes'):
                        diagnostics['trace'][key] = sorted(set(diagnostics['trace'][key]) | set(extra_trace[key]))
            diagnostics.update(alpha_h_per_km=alpha, ttd_gradient_units='veh*km / normalized control',
                               objective_gradient_units='veh*h / normalized control')
            self.last_result = SimpleNamespace(**{**vars(raw), 'diagnostics': diagnostics},
                objective_value=raw.total_ttt-alpha*float(distance_vector.sum())+float(terminal_vector.sum()),
                objective_cost_vector=objective_vector, objective_gradients=objective_gradients,
                total_ttd_veh_km=float(distance_vector.sum()), player_ttd_vector_veh_km=distance_vector,
                ttd_gradients=distance_jac, terminal_cost_veh_h=float(terminal_vector.sum()),
                terminal_gradients=terminal_jac)
            return self.last_result

    # 상속 순서를 원 구현과 일치시킨다. 원 모듈/클래스/파일은 수정하지 않는다.
    group = _clone(prox_controller.GroupProxSDMPC, {'PlayerSensitivityDMPC': ObjectiveEvaluator}, manifest)
    fixed = _clone(fixed_controller.FixedNPBandSDMPC, {'GroupProxSDMPC': group, 'contract': experiment_contract}, manifest)
    selected = _clone(selected_controller.SelectedDualSDMPC, {'FixedNPBandSDMPC': fixed}, manifest)
    # 별도 실험: 원 상위 후보를 보존하고 10개 탐색을 위한 내부 격자만 추가한다.
    from upper_search import decide as upper_decide
    selected.decide = _clone(upper_decide, {}, manifest)
    central = _clone(central_controller.CentralKKTSDMPC,
                     {'SelectedDualSDMPC': selected, 'FixedNPBandSDMPC': fixed, 'RecordedEngine': ObjectiveEngine}, manifest)
    choose_archive = _clone(exception_controller.choose_archive, {}, manifest)
    exception = _clone(exception_controller.BudgetExceptionSDMPC,
                       {'CentralKKTSDMPC': central, 'choose_archive': choose_archive, 'contract': experiment_contract}, manifest)
    anchor = _clone(anchor_controller.PFOAnchorSDMPC, {'BudgetExceptionSDMPC': exception}, manifest)
    build = _clone(reused_model.build, {}, manifest)

    class TTTTDAnchorSDMPC(anchor):
        def central_system(self, y, budget):
            return build(self, y, budget)

        def __init__(self, cfg, options):
            super().__init__(cfg, options)
            self.alpha_h_per_km = alpha
            self.objective_arm = objective
            self.objective_adapter_manifest = manifest
            self.terminal_settings = {key: value for key, value in vars(self.evaluator.terminal_cfg.mpc).items()
                                      if key.startswith(('leader_mfd_far_', 'leader_urban_clf_'))}

    return TTTTDAnchorSDMPC


def bootstrap():
    """배포본: 과거 출력 파일 대신 포함된 설정에서 runtime을 복원한다."""
    from sdmpc_ttd.runtime import bootstrap as portable_bootstrap
    return portable_bootstrap()
