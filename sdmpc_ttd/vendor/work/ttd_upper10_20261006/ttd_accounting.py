"""역사 교통 모델의 실제 방출 유량에 대한 모델 거리 생산량[veh km].

Spec 03(교통 모델), 12(코딩), 15(모델 선택의 명시)를 읽고 구현했다.
이는 미시 궤적의 부분 주행거리가 아니라 고정 길이 링크의 완료 거리이다.
고속도로: 실제 제한된 segment 방출량 * segment 길이.
도시: 원점 물리 링크를 떠나는 차량수 * (storage 용량 * 평균 차량 길이).
점큐 자체 및 길이가 없는 경계는 0, 도착 버퍼 예약은 거리로 더하지 않는다.
원 source/상태/TTT/제약은 수정하지 않고 현재 process의 함수에 관찰 호출만 삽입한다.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import importlib
from pathlib import Path
import sys
from typing import Any


METRIC_NAME = "completed_link_distance_production_v1"
_CAPTURE: ContextVar[Any] = ContextVar("ttd_capture_20260930", default=None)


def storage_distance_km(cfg, link: str) -> float:
    """도시 모델 자체의 최대 travel-distance(용량 * 차량 길이)를 고정 길이로 사용."""
    if link not in cfg.network.urban_link_storage_veh:
        return 0.0
    return (float(cfg.network.urban_link_storage_veh[link])
            * float(cfg.network.urban_avg_vehicle_length_m) / 1000.0)


def movement_distance_km(cfg, movement: str, spec=None) -> float:
    """movement의 원점 링크만 완료한다. 다음 receiving 링크는 여기서 계산하지 않는다."""
    spec = cfg.network.urban_movements[movement] if spec is None else spec
    if str(spec.get("kind", "")) == "off_ramp":
        return offramp_distance_km(cfg, str(spec.get("off_ramp", spec.get("origin", ""))))
    return storage_distance_km(cfg, str(spec.get("origin", "")))


def offramp_distance_km(cfg, offramp: str) -> float:
    """off-ramp storage를 빠져나올 때 해당 연결 링크 거리를 한 번 계산."""
    return storage_distance_km(cfg, cfg.network.off_ramp_storage_link.get(offramp, ""))


@dataclass
class DistanceCapture:
    cfg: Any
    ownership: Any = None
    freeway_ttd: Any = field(default=0.0, init=False)
    urban_ttd: Any = field(default=0.0, init=False)
    terms: dict[str, Any] = field(default_factory=dict, init=False)
    event_counts: dict[str, int] = field(default_factory=dict, init=False)
    raw_player_ttd: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.ownership is None:
            from src.simulation.player_cost_accounting import PlayerCostOwnership
            self.ownership = PlayerCostOwnership.from_config(self.cfg)
        self.lengths = {key: storage_distance_km(self.cfg, key)
                        for key in self.cfg.network.urban_link_storage_veh}

    def __enter__(self):
        # 중첩 후보 평가가 외부 plant 회계를 오염하지 않도록 가장 안쪽 회계만 활성화.
        self._token = _CAPTURE.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        _CAPTURE.reset(self._token)

    @property
    def total_ttd(self):
        return self.freeway_ttd + self.urban_ttd

    @property
    def player_ttd(self):
        # 순수 TTT ledger와 동일한 passive 비용 귀속을 사용한다.
        result = {key: 0.0 for key in self.ownership.active_player_ids}
        for key, value in self.raw_player_ttd.items():
            owner = key if key in result else self.ownership.passive_cost_owners.get(key)
            if owner not in result:
                raise ValueError(f"Unassigned TTD owner: {key}")
            result[owner] += value
        return result

    def add(self, term: str, region: str, owner: str, value):
        # Dual을 float로 변환하지 않아 condensed 연결 동역학의 총미분을 보존한다.
        self.terms[term] = self.terms.get(term, 0.0) + value
        self.raw_player_ttd[owner] = self.raw_player_ttd.get(owner, 0.0) + value
        self.event_counts[region] = self.event_counts.get(region, 0) + 1
        if region == "F":
            self.freeway_ttd += value
        elif region == "U":
            self.urban_ttd += value
        else:
            raise ValueError(region)


def _core(link, index, flow, dt_h, length_km):
    cap = _CAPTURE.get()
    if cap is not None:
        cap.add(f"segment:{link}:{index}", "F", cap.ownership.segment_owner[(link, index)],
                flow * dt_h * length_km)


def _buffer(link, flows_out, dt_h, length_km):
    cap = _CAPTURE.get()
    if cap is not None:
        # 완충은 원 TTT와 같이 해당 freeway origin owner에 귀속. core와 다른 공간이다.
        cap.add(f"buffer:{link}", "F", cap.ownership.origin_owner[link],
                sum(flows_out) * dt_h * length_km)


def _movement(movement, actual, spec):
    cap = _CAPTURE.get()
    if cap is not None:
        length = cap.lengths.get(str(spec.get("origin", "")), 0.0)
        cap.add(f"movement:{movement}", "U", cap.ownership.movement_owner[movement], actual * length)


def _storage(link, actual):
    cap = _CAPTURE.get()
    if cap is not None:
        # off-ramp는 기존 TTT 소유권을 따라 F, 도시 내부/출구 링크는 U.
        off = cap.ownership.offramp_storage_owner
        owner = off[link] if link in off else cap.ownership.urban_storage_owner[link]
        cap.add(f"storage:{link}", "F" if link in off else "U", owner,
                actual * cap.lengths[link])


class _Events(ast.NodeTransformer):
    """원 수치식을 바꾸지 않고 실제 방출값이 결정된 위치에만 회계 호출 삽입."""
    def __init__(self):
        self.scope = ""
        self.counts = {key: 0 for key in ("core", "buffer", "movement", "sink", "off")}

    def visit_FunctionDef(self, node):
        previous, self.scope = self.scope, node.name
        node = self.generic_visit(node)
        self.scope = previous
        return node

    def call(self, key, code, node):
        self.counts[key] += 1
        return [ast.copy_location(ast.parse(code).body[0], node), node]

    def visit_Assign(self, node):
        if (self.scope == "freeway_substep" and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "vehicle_raw"):
            return self.call("core", "_ttd_core(link, i, q_out, dt_h, net.freeway_segment_length_km)", node)
        return self.generic_visit(node)

    def visit_AugAssign(self, node):
        target = node.target.id if isinstance(node.target, ast.Name) else ""
        value = node.value.id if isinstance(node.value, ast.Name) else ""
        if self.scope == "urban_substep" and target == "total_departures_veh" and value == "actual":
            return self.call("movement", "_ttd_movement(movement, actual, specs[movement])", node)
        if self.scope == "urban_substep" and target == "boundary_out_sink_veh":
            return self.call("sink", "_ttd_storage(link, departed)", node)
        if self.scope == "_drain_offramp_storage" and target == "released_total" and value == "actual":
            return self.call("off", "_ttd_storage(storage_link, actual)", node)
        return self.generic_visit(node)

    def visit_Return(self, node):
        if self.scope == "_adv_chain":
            return self.call("buffer", "_ttd_buffer(link, flows_out, dt_h, _buf_len)", node)
        return self.generic_visit(node)


class _RetainAD(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ("float", "min", "max"):
            node.func.id = "_ad_" + node.func.id
        return node


def install_namespace(modules: dict[str, Any]) -> dict:
    """Scalar 또는 이미 생성된 격리 AD namespace에 같은 방출 이벤트를 설치한다."""
    plans = []
    counts = {key: 0 for key in ("core", "buffer", "movement", "sink", "off")}
    hashes = {}
    for name, functions in (("src.models.metanet", {"freeway_substep"}),
                            ("src.models.urban_queue_model", {"urban_substep", "_drain_offramp_storage"})):
        module = modules[name]
        if getattr(module, "_ttd_events_installed", False):
            continue
        path = Path(module.__file__)
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, str(path))
        tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        transformer = _Events()
        tree = transformer.visit(tree)
        if "_ad_float" in vars(module):
            tree = _RetainAD().visit(tree)
        ast.fix_missing_locations(tree)
        for key in counts:
            counts[key] += transformer.counts[key]
        hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        plans.append((module, compile(tree, str(path), "exec"), functions, transformer.counts))
    # 소스가 바뀌어 이벤트 위치가 없어지면 일부만 설치하지 않고 즉시 중단한다.
    for module, code, functions, actual in plans:
        expected = ({"core": 1, "buffer": 1, "movement": 0, "sink": 0, "off": 0}
                    if "freeway_substep" in functions else
                    {"core": 0, "buffer": 0, "movement": 2, "sink": 1, "off": 1})
        if actual != expected:
            raise ValueError(f"TTD source event mismatch: {module.__name__}: {actual} != {expected}")
    replacements = []
    for module, code, functions, actual in plans:
        previous = {key: vars(module)[key] for key in functions}
        module.__dict__.update(_ttd_core=_core, _ttd_buffer=_buffer,
                               _ttd_movement=_movement, _ttd_storage=_storage)
        exec(code, module.__dict__)
        replacements.extend((old, vars(module)[key]) for key, old in previous.items())
        module._ttd_events_installed = True
    # coupling의 from-import alias도 같은 namespace 내부에서만 교체한다.
    for module in modules.values():
        for key, value in list(vars(module).items()):
            for old, new in replacements:
                if value is old:
                    setattr(module, key, new)
    return {"metric": METRIC_NAME, "source_sha256": hashes, "events": counts,
            "installed_functions": len(replacements)}


def install_scalar() -> dict:
    """현재 process의 역사 src 패키지만 계측하며 파일은 변경하지 않는다."""
    for name in ("src.models.metanet", "src.models.urban_queue_model", "src.simulation.coupling"):
        importlib.import_module(name)
    modules = {name: module for name, module in list(sys.modules.items())
               if module is not None and name.startswith("src.")}
    return install_namespace(modules)
