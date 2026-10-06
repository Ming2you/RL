"""기존 TTD S-DMPC를 역사 출력 파일 없이 초기화하는 배포 경계.

별도 Python process에서 먼저 bootstrap()을 호출한다. 기존 RL 저장소의 src와
이 runtime의 src를 같은 process에서 섞으면 즉시 중단한다.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PACKAGE = Path(__file__).resolve().parent
VENDOR = PACKAGE/'vendor'
HIST = VENDOR/'work/sdmpc_matrix_14400_20260912/historical_tree'
CONFIG = PACKAGE/'config'
SCENARIOS = ('sweet_155_w', 'sweet_170_w', 'sweet_170_skew15_w',
             'sweet_170_incident_w', 'sweet_190_w')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def plain(value):
    from dataclasses import asdict, is_dataclass
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(plain(value), indent=2, ensure_ascii=False)+'\n').encode())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def environment(mode='SDMPC'):
    if mode != 'SDMPC':
        raise ValueError('This package exposes the TTD S-DMPC experiment only')
    spec = read(VENDOR/'work/sdmpc_matrix_14400_20260912/historical_environment.json')
    for key in spec['controller_environment_keys'] + spec['launch_unset']:
        os.environ.pop(key, None)
    os.environ.update(spec['environment'])
    for key in ('CENT_SLSQP', 'CENT_SLSQP_MAXEVAL'):
        os.environ.pop(key, None)
    os.environ['PFO_SPLIT'] = '2'
    os.environ['SDMPC_BUDGET_VARIANT'] = 'upper'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMEXPR_MAX_THREADS', 'BLIS_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['OMP_DYNAMIC'] = os.environ['MKL_DYNAMIC'] = 'FALSE'
    return {key: os.environ.get(key) for key in spec['controller_environment_keys']}


def pin(mask=None):
    """원 실험처럼 논리 CPU 하나를 선택한다. Linux에서는 허용 CPU 중 선택한다."""
    if os.name != 'nt':
        if not hasattr(os, 'sched_getaffinity'):
            raise RuntimeError('CPU affinity unavailable; single-core timing cannot be asserted')
        available = os.sched_getaffinity(0)
        cpu = min(available) if mask is None else int(mask).bit_length()-1
        if mask is not None and (mask <= 0 or mask & (mask-1)):
            raise ValueError('cpu-mask must select one CPU')
        if cpu not in available:
            raise ValueError('Requested CPU unavailable')
        os.sched_setaffinity(0, {cpu})
        assert os.sched_getaffinity(0) == {cpu}
        return dict(pid=os.getpid(), logical_cpu_index=cpu, mask=1 << cpu)
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    proc = kernel.GetCurrentProcess()
    before, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(proc, ctypes.byref(before), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    selected = int(mask) if mask is not None else before.value & -before.value
    if selected <= 0 or selected & (selected-1) or not selected & before.value:
        raise ValueError('Unavailable single CPU mask')
    if not kernel.SetProcessAffinityMask(proc, selected):
        raise ctypes.WinError(ctypes.get_last_error())
    after = ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(proc, ctypes.byref(after), ctypes.byref(system)) or after.value != selected:
        raise RuntimeError('CPU affinity verification failed')
    return dict(pid=os.getpid(), before_mask=before.value, mask=selected, logical_cpu_index=selected.bit_length()-1)


def verify_sources():
    rows = read(PACKAGE/'source_manifest.json')
    for row in rows:
        if hashlib.sha256((VENDOR/row['path']).read_bytes()).hexdigest() != row['packaged_sha256']:
            raise RuntimeError('Packaged source changed: '+row['path'])
    return len(rows)


def bootstrap():
    """원 수식·solver를 그대로 import하고 effective config를 복원한다."""
    verify_sources()
    # top-level 이름을 쓰는 연구 코드이므로 이미 로드된 다른 runtime을 금지한다.
    isolated = {'anchor_controller', 'fixed_controller', 'fixed_policy', 'policy', 'central',
                'controller', 'prox_controller', 'multiplier', 'central_controller',
                'reused_model', 'sdmpc_objective', 'ttd_accounting', 'group_block_engine',
                'blocks', 'band_math', 'local_qp', 'exception_controller', 'upper_search',
                'terminal_cost', 'historical_config'}
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and (name == 'src' or name.startswith(('src.', 'sparse.')) or name in isolated):
            if not Path(file).resolve().is_relative_to(VENDOR):
                raise RuntimeError('Mixed runtime import; use a fresh process: '+name)
    environment()
    folders = ['ttd_upper10_20261006', 'sdmpc_budget_ablation_20260922',
               'sdmpc_central_kkt_reuse_20260922', 'sdmpc_selected_dual_20260922',
               'sdmpc_relative_band_20260922', 'sdmpc_group_proxlinear_20260922',
               'sdmpc_slide_alignment_20260922', 'sdmpc_block_speed_ordered_20260919',
               'sdmpc_sparse_local_20260916', 'sdmpc_matrix_14400_20260912']
    paths = [str(VENDOR/'work'/p) for p in folders] + [str(HIST)]
    sys.path[:] = paths + [p for p in sys.path if p not in paths]
    from historical_config import restore_historical_config, to_plain_dict
    from src.controllers.player_sensitivity_dmpc import PlayerSDMPCOptions
    from src.models.state import ControlAction, TrafficState
    from src.models.demand import DemandStep
    from src.simulation.simulator import MixedTrafficSimulator
    name = '_sdmpc_ttd_historical_factory'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HIST/'work/run_claude_style_five_controller.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    factory = sys.modules[name]
    cfg = restore_historical_config(read(CONFIG/'factory_config.json'))
    opts = PlayerSDMPCOptions(**read(CONFIG/'solver_options.json'))
    from anchor_controller import PFOAnchorSDMPC
    rc = SimpleNamespace(ControlAction=ControlAction, TrafficState=TrafficState,
        DemandStep=DemandStep, PlayerSDMPCOptions=PlayerSDMPCOptions,
        MixedTrafficSimulator=MixedTrafficSimulator, make_controller=factory.make_controller,
        to_plain_dict=to_plain_dict, restore_historical_config=restore_historical_config,
        FrozenProfile=FrozenProfile)
    return rc, cfg, opts, (), PFOAnchorSDMPC


class FrozenProfile:
    """재생성한 수요 시간표를 고정하고 외삽을 금지한다."""
    def __init__(self, rows, dt):
        from src.models.demand import DemandStep
        self.dt, self.table = dt, {}
        for row in rows:
            value = copy.deepcopy(row)
            timestamp = float(value.pop('time_sec'))
            if not math.isfinite(timestamp) or timestamp in self.table:
                raise ValueError('Invalid frozen forecast time')
            value['freeway_lane_loss'] = {link: {int(i): loss for i, loss in losses.items()}
                for link, losses in value.get('freeway_lane_loss', {}).items()}
            self.table[timestamp] = DemandStep(**value)
        if sorted(self.table) != [i*dt for i in range(len(self.table))]:
            raise ValueError('Frozen forecast not contiguous from zero')

    def at(self, timestamp):
        if float(timestamp) not in self.table:
            raise ValueError('Frozen forecast extrapolation forbidden')
        return copy.deepcopy(self.table[float(timestamp)])

    def horizon(self, timestamp, depth):
        return [self.at(timestamp+i*self.dt) for i in range(depth)]


def protocol(rc, cfg, scenario):
    """원 실험의 입력을 결정적으로 재생성하고 fingerprint를 대조한다."""
    if scenario not in SCENARIOS:
        raise ValueError('Unknown scenario: '+scenario)
    from src.models.demand import DemandProfile, ScenarioConfig
    raw = read(CONFIG/'scenarios.json')[scenario]
    profile = DemandProfile(cfg, ScenarioConfig(**raw))
    count = read(CONFIG/'forecast_count.json')[scenario]
    rows = [dict(time_sec=float(i*cfg.simulation.T_c_sec),
                 **rc.to_plain_dict(profile.at(i*cfg.simulation.T_c_sec))) for i in range(count)]
    result = dict(config=rc.to_plain_dict(cfg), scenario=raw, forecast=rows,
                  initial_state=rc.to_plain_dict(rc.TrafficState.initial(cfg)))
    expected = read(CONFIG/'protocol_fingerprints.json')[scenario]
    for name, value in result.items():
        if digest(value) != expected[name]:
            raise RuntimeError('Historical input fingerprint mismatch: '+scenario+'/'+name)
    return result


def calibrate(rc, cfg, scenario, profile):
    """14,400초 무제어를 재실행해 alpha를 고정한다. 실제·예측 성과를 구분한다."""
    from ttd_accounting import DistanceCapture, install_scalar
    install_scalar()
    sim = rc.MixedTrafficSimulator(cfg)
    total_distance = 0.
    for k in range(80):
        with DistanceCapture(cfg) as capture:
            sim.step(rc.ControlAction.uncontrolled(cfg), profile.at(sim.state.time_sec), k)
        total_distance += float(capture.total_ttd)
    if not (math.isfinite(sim.total_ttt) and math.isfinite(total_distance)
            and sim.total_ttt > 0 and total_distance > 0 and sim.state.time_sec == 14400):
        raise RuntimeError('Invalid NC normalization')
    recomputed_alpha = sim.total_ttt/total_distance
    frozen_alpha = read(CONFIG/'alpha.json')[scenario]
    if abs(recomputed_alpha-frozen_alpha) > 1e-15:
        raise RuntimeError('NC calibration differs from the frozen experimental weight')
    return dict(experiment_complete=True, scenario=scenario, elapsed_seconds=14400, seed=42,
                ttt=sim.total_ttt, ttd=total_distance, alpha_h_per_km=frozen_alpha,
                recomputed_alpha_h_per_km=recomputed_alpha,
                alpha_source='frozen experiment coefficient; regenerated NC ratio checked at 1e-15')


def prepare_inputs(directory, scenario):
    rc, cfg, _, _, _ = bootstrap()
    data = protocol(rc, cfg, scenario)
    directory = Path(directory)
    for name, value in data.items():
        write(directory/'protocols'/scenario/(name+'.json'), value)
    profile = FrozenProfile(data['forecast'], cfg.simulation.T_c_sec)
    norm = calibrate(rc, cfg, scenario, profile)
    path = directory/'normalization.json'
    write(path, norm)
    return directory/'protocols', path, norm['alpha_h_per_km']


def make_runtime(scenario='sweet_170_w', max_candidates=3):
    """학습 환경 연결용 구성 요소. caller가 step/commit을 명시적으로 수행한다."""
    from dataclasses import replace
    if max_candidates not in (3, 10):
        raise ValueError('Validated candidate caps: 3 or 10')
    rc, cfg, options, _, _ = bootstrap()
    options = replace(options, max_candidates=max_candidates)
    data = protocol(rc, cfg, scenario)
    profile = FrozenProfile(data['forecast'], cfg.simulation.T_c_sec)
    norm = calibrate(rc, cfg, scenario, profile)
    from sdmpc_objective import create_controller_class
    solver = create_controller_class(norm['alpha_h_per_km'], 'ttd', max_candidates)(cfg, options)
    return SimpleNamespace(rc=rc, cfg=cfg, options=options, profile=profile,
        normalization=norm, alpha=norm['alpha_h_per_km'], solver=solver,
        plant=rc.MixedTrafficSimulator(cfg),
        warm_controller=rc.make_controller('WU-FAITHFUL-FOLLOWER', cfg),
        previous=rc.ControlAction.uncontrolled(cfg))
