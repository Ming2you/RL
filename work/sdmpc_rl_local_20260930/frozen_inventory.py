"""Reuse frozen accounting definitions without booting a physical runtime."""
import ast
import copy
import importlib.util
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2] / "artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_tree/src"


@lru_cache(maxsize=1)
def accounting():
    # state.py has only standard-library imports. Keep it out of the physical src namespace.
    spec = importlib.util.spec_from_file_location("local_frozen_state", ROOT / "models/state.py")
    state = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = state
    spec.loader.exec_module(state)
    namespace = dict(TrafficState=state.TrafficState, ExperimentConfig=state.ExperimentConfig,
                     fields=fields, is_dataclass=is_dataclass, Any=Any, Mapping=Mapping)
    # Compile the unchanged pure definitions, excluding runtime/solver/simulator imports.
    for path, name in ((ROOT / "simulation/player_cost_accounting.py", "freeway_buffer_vehicle_counts"),
                       (ROOT / "controllers/sensitivity_dmpc.py", "physical_inventory"),
                       (ROOT.parents[1] / "historical_config.py", "to_plain_dict")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(nodes) != 1:
            raise ValueError("Frozen accounting definition missing or ambiguous: " + name)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return state.TrafficState, namespace["physical_inventory"], namespace["to_plain_dict"]


def complete_config(config):
    return accounting()[2](config)


def inventory(state, config):
    state_type, compute, _ = accounting()
    required = ("freeway_density", "freeway_effective_lanes", "ramp_queue", "mainline_origin_queue",
                "urban_queue", "boundary_queue", "urban_movement_queue", "urban_link_storage",
                "freeway_buffer_up_density", "freeway_buffer_down_density")
    if any(name not in state for name in required):
        raise ValueError("Missing physical inventory state")
    cfg = SimpleNamespace(network=SimpleNamespace(**config["network"]))
    return compute(state_type(**copy.deepcopy(state)), cfg)
