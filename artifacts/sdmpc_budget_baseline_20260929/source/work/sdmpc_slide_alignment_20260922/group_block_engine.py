"""역사 결합 예측 프로그램의 격리된 활성 경로 자동미분.

원 모듈을 monkeypatch하지 않는다. 별도 namespace에서 원 명령과 분기를
실행하고 float 변환 및 scalar math/numpy primitive만 AD에 맞춘다.
원 source hash와 변경한 primitive를 결과에 명시한다.
"""
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np

from sparse.dual import Dual, Trace, MathProxy, NumpyProxy, float_keep, primal, derivative
from blocks import minimum, maximum, install_metanet_blocks


ROOT = Path(__file__).resolve().parents[2]
HIST = ROOT / 'work/sdmpc_matrix_14400_20260912/historical_tree'
PREFIX = '_sdmpc_group_block_ad_20260922'
MODULES = ('src.models.state', 'src.models.metanet', 'src.models.urban_queue_model',
           'src.simulation.player_cost_accounting', 'src.simulation.coupling')


class Transform(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'float':
            node.func.id = '_ad_float'
        elif isinstance(node.func, ast.Name) and node.func.id in ('min', 'max'):
            node.func.id = '_ad_' + node.func.id
        return node

    def visit_ImportFrom(self, node):
        if node.level == 0 and node.module in MODULES:
            node.module = PREFIX + '.' + node.module
        elif node.level == 1 and node.module in ('state', 'metanet', 'urban_queue_model'):
            node.module = PREFIX + '.src.models.' + node.module
            node.level = 0
        elif node.level == 1 and node.module == 'demand':
            node.module = 'src.models.demand'
            node.level = 0
        return node


def load_modules():
    for package in (PREFIX, PREFIX+'.src', PREFIX+'.src.models', PREFIX+'.src.simulation'):
        if package not in sys.modules:
            module = ModuleType(package)
            module.__path__ = []
            sys.modules[package] = module
    modules, hashes = {}, {}
    for original in MODULES:
        path = HIST / (original.replace('.', '/') + '.py')
        source = path.read_text(encoding='utf-8')
        name = PREFIX + '.' + original
        if name not in sys.modules:
            tree = Transform().visit(ast.parse(source, str(path)))
            ast.fix_missing_locations(tree)
            module = ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = name.rpartition('.')[0]
            module.__dict__['_ad_float'] = float_keep
            module.__dict__['_ad_min'] = minimum
            module.__dict__['_ad_max'] = maximum
            sys.modules[name] = module
            exec(compile(tree, str(path), 'exec'), module.__dict__)
            # 이 격리 dictionary에서만 imported primitive를 교체한다.
            if 'math' in module.__dict__:
                module.math = MathProxy()
            if 'np' in module.__dict__:
                module.np = NumpyProxy()
            if original == 'src.models.metanet':
                install_metanet_blocks(module)
        modules[original] = sys.modules[name]
        hashes[original] = hashlib.sha256(path.read_bytes()).hexdigest()
    return modules, hashes


@dataclass
class TangentResult:
    cost_vector: np.ndarray
    budget_vector: np.ndarray
    gradients: np.ndarray
    budget_jacobian: np.ndarray
    total_ttt: float
    total_gradient: np.ndarray
    states: list
    diagnostics: dict


class SparseRolloutEngine:
    def __init__(self, cfg, horizon_steps=3):
        self.cfg, self.horizon_steps = cfg, horizon_steps
        self.modules, self.source_hashes = load_modules()
        self.State = self.modules['src.models.state'].TrafficState
        ledger = self.modules['src.simulation.player_cost_accounting']
        self.Ledger = ledger.PlayerCostLedger
        self.ownership = ledger.PlayerCostOwnership.from_config(cfg)
        self.coupled = self.modules['src.simulation.coupling'].run_coupled_interval

    def evaluate(self, state, forecast, control, coordinates, *, active_columns=None):
        """`control` 기준점에서 정규화 controller 좌표로 미분한다.

        `control = coordinates.decode(z,vsl)`로 상태·미분의 기준점을 일치시킨다.
        offset 미분은 같은 previous에 대한 상대 좌표이며 `control`에는 이미
        실제 물리 offset이 저장되어 있다.
        """
        if len(forecast) < self.horizon_steps:
            raise ValueError('forecast does not cover the unchanged full horizon')
        started, cpu = time.perf_counter(), time.process_time()
        trace = Trace(coordinates.fd)
        active = set(range(len(coordinates.axes)) if active_columns is None else active_columns)
        ctrl = copy.deepcopy(control)
        for j, axis in enumerate(coordinates.axes):
            if j not in active:
                continue
            kind, key, scale = axis[:3]
            def seed(value, sign=1.0):
                return Dual(value, {j: sign*scale}, trace)
            if kind == 'meter':
                ctrl.ramp_metering[key] = seed(ctrl.ramp_metering[key])
            elif kind == 'green':
                ctrl.green_times[key+'_p1'] = seed(ctrl.green_times[key+'_p1'])
                ctrl.green_times[key+'_p2'] = seed(ctrl.green_times[key+'_p2'], -1.0)
            elif kind == 'offset':
                ctrl.offsets[key] = seed(ctrl.offsets[key])
            elif kind == 'vsl':
                ctrl.vsl[key] = seed(ctrl.vsl[key])
            elif kind == 'vsl_group':
                # 한 group의 네 segment에 같은 제어축 tangent를 전파한다.
                link,group=key.split('__group')
                for index in range(4*int(group),4*(int(group)+1)):
                    segment_key=f'{link}__seg{index}'
                    ctrl.vsl[segment_key]=seed(ctrl.vsl[segment_key])
            else:
                raise ValueError(f'Unsupported control axis: {kind}')
        for link in self.cfg.network.freeway_links:
            ctrl.vsl[link] = min((ctrl.vsl[f'{link}__seg{i}']
                                 for i in range(self.cfg.network.freeway_segments_per_link)), key=primal)
        current = self.State(**copy.deepcopy(vars(state)))
        ledger = self.Ledger(self.cfg, self.ownership)
        total, service = 0.0, 0.0
        states = []
        for demand in list(forecast)[:self.horizon_steps]:
            result = self.coupled(current, ctrl, demand, self.cfg, cost_observer=ledger.observe)
            total += result.freeway_ttt + result.urban_ttt
            service += result.diagnostics['inbound_service_veh'] - result.diagnostics['outbound_service_veh']
            current.time_sec += self.cfg.simulation.T_c_sec
            states.append(copy.deepcopy(current))
        command = sum(ctrl.ramp_metering.get(r, 0.0) for r in self.cfg.network.ramps)
        costs = [ledger.player_costs[key] for key in self.ownership.active_player_ids]
        n = len(coordinates.axes)
        def jac(values):
            return np.array([[derivative(value).get(j, 0.0) for j in range(n)] for value in values])
        return TangentResult(np.array([primal(v) for v in costs]), np.array([primal(service), primal(command)]),
                             jac(costs), jac([service, command]), primal(total), jac([total])[0], states,
                             dict(wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
                                  trace=trace.result(costs+[service, command]), source_hashes=self.source_hashes,
                                  active_columns=sorted(active), full_coupled_rollouts=1,
                                  differentiation='three direct METANET block tangents plus original active-path scalar AD',
                                  transformed_primitives=['float casts preserve tangent', 'math.exp', 'scalar numpy.clip',
                                                          'diagnostic numpy arrays retain object scalars'],
                                  original_nonlinear_execution_gate_required=True))


def primal_tree(value):
    """미분을 삭제하는 암묵적 cast 없이 primal 상태의 동등성을 검사한다."""
    if isinstance(value, Dual):
        return value.value
    if isinstance(value, dict):
        return {key: primal_tree(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [primal_tree(val) for val in value]
    if hasattr(value, '__dataclass_fields__'):
        return primal_tree(vars(value))
    return value
