"""Synthetic wrapper contracts; no historical imports or traffic solves."""
import copy
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from budget_controller import BudgetController
from budget_env import BudgetEnv, Observation
import budget_runtime


@dataclass
class Network:
    urban_link_storage_veh: dict = field(default_factory=lambda: {"u": 100.})
    urban_movements: dict = field(default_factory=dict)
    boundary_queue_max_veh: float = 100.
    urban_avg_vehicle_length_m: float = 5.
    urban_avg_speed_km_h: float = 20.
    freeway_lanes: int = 3
    freeway_links: tuple = ("f",)
    freeway_segments_per_link: int = 1
    signals: tuple = ()
    cycle_length: float = 60.
    total_ramp_capacity: float = 6000.
    ramp_queue_max_veh: float = 180.


@dataclass
class Simulation:
    T_u_h: float = 1. / 3600.
    T_u_sec: float = 1.


@dataclass
class Config:
    network: Network = field(default_factory=Network)
    simulation: Simulation = field(default_factory=Simulation)


@dataclass
class Options:
    horizon_steps: int = 3
    max_iterations: int = 6
    max_candidates: int = 3


@dataclass
class State:
    time_sec: float = 900.
    ramp_queue: dict = field(default_factory=lambda: {"r": 2.})
    mainline_origin_queue: dict = field(default_factory=dict)
    boundary_queue: dict = field(default_factory=dict)
    urban_movement_queue: dict = field(default_factory=dict)
    urban_arrival_buffer: dict = field(default_factory=dict)
    urban_storage_release_buffer: dict = field(default_factory=dict)

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Control:
    point: list = field(default_factory=lambda: [0., 0.])
    offsets: dict = field(default_factory=dict)
    N_P_star: float = 0.
    N_UF_star: float = 1000.

    def copy(self):
        return copy.deepcopy(self)


class Coordinates:
    def __init__(self, *args):
        pass

    def encode(self, control):
        return np.asarray(control.point).copy()

    def quantize(self, point):
        return point.copy()

    def decode(self, point):
        return Control(point.tolist())


class Lower:
    def __init__(self, cfg, options):
        self.cfg, self.options = cfg, options
        self.dual = np.array([[2., 3.], [0., 0.]])
        self.last_budget = np.array([0., 1000.])
        self.coords = Coordinates()
        self.counts = {"scalar_rollouts": 0}
        self.execution_archives = []

    def begin(self, *args):
        self.execution_archives = []
        self.derivative_rows = []

    def evaluate(self, point):
        return SimpleNamespace(total_ttt=10. + point[1], budget_vector=np.array([point[0], 1000.]),
                               physical_valid=True, control_valid=True, states=["DO_NOT_EXPORT"])

    def execution_check(self, point, budget):
        ev = self.evaluate(point)
        residual = ev.budget_vector - budget
        return dict(physical_control_valid=True, budget_feasible=bool(np.all(residual <= 0.)),
                    achieved=ev.budget_vector.tolist(), original_residual=residual.tolist(),
                    scaled_excess=np.maximum(residual, 0.).tolist(), point=list(point),
                    budget=list(budget), ttt=ev.total_ttt)


@dataclass
class Log:
    freeway_ttt: float = 2.
    urban_ttt: float = 3.
    diagnostics: dict = field(default_factory=lambda: {"conservation": 0.})


class Simulator:
    def __init__(self, cfg):
        self.cfg, self.state = cfg, State()
        self.total_ttt = self.freeway_ttt = self.urban_ttt = 0.
        self.logs = []

    def copy(self):
        return copy.deepcopy(self)

    def step(self, *args):
        self.state.time_sec += 180.
        self.freeway_ttt += 2.
        self.urban_ttt += 3.
        self.total_ttt += 5.
        log = Log()
        self.logs.append(log)
        return log


def runtime():
    return dict(cfg=Config(), options=Options(), baseline=Lower, coordinates=Coordinates,
                audit_rows=lambda rows: None, snapshot_identity={"manifest_sha256": "a" * 64},
                rc=SimpleNamespace(inventory=lambda state, cfg: {"vehicles": 2.},
                                   to_plain_dict=config_fixture_plain))


def config_fixture_plain(value):
    """Stub of historical_config.to_plain_dict, including public dynamic fields."""
    if is_dataclass(value):
        declared = {item.name for item in fields(value)}
        out = {name: config_fixture_plain(getattr(value, name)) for name in declared}
        out.update({name: config_fixture_plain(item) for name, item in vars(value).items()
                    if name not in declared and not name.startswith("_")})
        return out
    if isinstance(value, dict):
        return {str(key): config_fixture_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [config_fixture_plain(item) for item in value]
    if isinstance(value, set):
        return sorted(config_fixture_plain(item) for item in value)
    return value


def environment(rt=None):
    env = BudgetEnv.__new__(BudgetEnv)
    env.rt = rt or runtime()
    env.cfg, env.reward_scale = env.rt["cfg"], 100.
    env.profile_hash = "same-profile"
    env.sim, env.warm, env.previous = Simulator(env.cfg), SimpleNamespace(memory=[7.]), Control()
    env.k, env.warmup_ttt, env.last_fallback = 5, 0., False
    env.last_requested = env.last_executed = np.array([0., 1000.])
    env.last_slack = np.zeros(2)
    env.forecast = [SimpleNamespace(freeway_mainline={"f": 1.}, urban_boundary={}, ramp_arrival={},
                                   incident_capacity_factor=1., freeway_lane_loss={})] * 3
    env.profile = SimpleNamespace(horizon=lambda *args: env.forecast)
    env.controller = BudgetController(env.rt)
    with patch.dict(sys.modules, {"fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]])}):
        env.controller.prepare_reference(env.sim.state, env.forecast, env.previous, Control())
    env.reference_timing = dict(pfo_wall_seconds=1., pfo_cpu_seconds=.5,
                                reference_wall_seconds=2., reference_cpu_seconds=1.,
                                forecast_wall_seconds=.1, forecast_cpu_seconds=.05)
    env.observation_timing = dict(observation_wall_seconds=.1, observation_cpu_seconds=.05)
    env.observer = Observation(env.cfg)
    env.observer.encode(env)
    return env


class CheckpointTests(unittest.TestCase):
    def test_dynamic_receiving_config_and_options_rejected_before_replacement(self):
        original = environment()
        original.cfg.network.terminal_zero_gradient = True
        original.cfg.mpc = Options()
        original.cfg.mpc.leader_rollout_box_walk = True
        original.rt["options"].runtime_tolerance = .01
        checkpoint = original.checkpoint()
        for location in ("network", "mpc", "options", "runtime_cfg"):
            with self.subTest(location=location):
                receiver = environment()
                receiver.cfg.network.terminal_zero_gradient = True
                receiver.cfg.mpc = copy.deepcopy(original.cfg.mpc)
                receiver.rt["options"].runtime_tolerance = .01
                if location == "network":
                    receiver.cfg.network.terminal_zero_gradient = False
                elif location == "mpc":
                    receiver.cfg.mpc.leader_rollout_box_walk = False
                elif location == "options":
                    receiver.rt["options"].runtime_tolerance = .02
                else:
                    receiver.rt["cfg"] = copy.deepcopy(receiver.cfg)
                    receiver.rt["cfg"].network.terminal_zero_gradient = False
                sim, lower, warm = receiver.sim, receiver.controller.lower, receiver.warm
                with patch.dict(receiver.rt, baseline=Mock(side_effect=AssertionError("lower constructed"))), \
                        self.assertRaisesRegex(ValueError, "contract"):
                    receiver.restore(checkpoint)
                self.assertIs(receiver.sim, sim)
                self.assertIs(receiver.controller.lower, lower)
                self.assertIs(receiver.warm, warm)

    def test_dynamic_saved_simulator_config_rejected_before_replacement(self):
        original = environment()
        original.cfg.network.terminal_zero_gradient = True
        checkpoint = original.checkpoint()
        checkpoint["sim"].cfg.network.terminal_zero_gradient = False
        receiver = environment()
        receiver.cfg.network.terminal_zero_gradient = True
        sim, lower, warm = receiver.sim, receiver.controller.lower, receiver.warm
        with patch.dict(receiver.rt, baseline=Mock(side_effect=AssertionError("lower constructed"))), \
                self.assertRaisesRegex(ValueError, "config contract"):
            receiver.restore(checkpoint)
        self.assertIs(receiver.sim, sim)
        self.assertIs(receiver.controller.lower, lower)
        self.assertIs(receiver.warm, warm)

    def test_dynamic_contract_is_detached_and_same_config_restore_is_exact(self):
        original = environment()
        original.cfg.network.terminal_zero_gradient = True
        original.cfg.mpc = Options()
        original.cfg.mpc.runtime_options = {"enabled": [True]}
        original.rt["options"].runtime_tolerance = .01
        original.cfg.network._diagnostic_cache = "not behavioral public config"
        expected = original.observer.encode(original)
        checkpoint = pickle.loads(pickle.dumps(original.checkpoint()))
        self.assertTrue(checkpoint["contract"]["cfg"]["network"]["terminal_zero_gradient"])
        self.assertEqual(checkpoint["contract"]["cfg"]["mpc"]["runtime_options"], {"enabled": [True]})
        self.assertEqual(checkpoint["contract"]["options"]["runtime_tolerance"], .01)
        self.assertNotIn("_diagnostic_cache", checkpoint["contract"]["cfg"]["network"])
        receiver = environment()
        receiver.cfg.network.terminal_zero_gradient = True
        receiver.cfg.mpc = copy.deepcopy(original.cfg.mpc)
        receiver.rt["options"].runtime_tolerance = .01
        with patch.dict(sys.modules, {"fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]])}), \
                patch.object(BudgetEnv, "prepare", side_effect=AssertionError("PFO repeated")):
            np.testing.assert_array_equal(receiver.restore(checkpoint), expected)
        self.assertTrue(receiver.sim.cfg.network.terminal_zero_gradient)
        self.assertTrue(receiver.controller.lower.cfg.network.terminal_zero_gradient)
        original.cfg.mpc.runtime_options["enabled"][0] = False
        self.assertEqual(checkpoint["contract"]["cfg"]["mpc"]["runtime_options"], {"enabled": [True]})

    def test_mismatches_rejected_before_replacing_live_state(self):
        checkpoint = environment().checkpoint()
        for field_name in ("reward", "cfg", "options", "source", "observation", "names"):
            with self.subTest(field=field_name):
                receiver = environment()
                if field_name == "reward":
                    receiver.reward_scale = 200.
                elif field_name == "cfg":
                    receiver.cfg.network.freeway_lanes = 4
                elif field_name == "options":
                    receiver.rt["options"].max_iterations = 5
                elif field_name == "source":
                    receiver.rt["snapshot_identity"]["manifest_sha256"] = "b" * 64
                elif field_name == "observation":
                    receiver.observer.delay += 1
                else:
                    receiver.observer.names = ("wrong",)
                sim, lower, warm = receiver.sim, receiver.controller.lower, receiver.warm
                with patch.dict(receiver.rt, baseline=Mock(side_effect=AssertionError("lower constructed"))), \
                        self.assertRaisesRegex(ValueError, "contract"):
                    receiver.restore(checkpoint)
                self.assertIs(receiver.sim, sim)
                self.assertIs(receiver.controller.lower, lower)
                self.assertIs(receiver.warm, warm)

    def test_same_contract_restore_is_exact_without_pfo(self):
        original = environment()
        expected = original.observer.encode(original)
        checkpoint = pickle.loads(pickle.dumps(original.checkpoint()))
        receiver = environment()
        receiver.observer = Observation(receiver.cfg)
        with patch.dict(sys.modules, {"fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]])}), \
                patch.object(BudgetEnv, "prepare", side_effect=AssertionError("PFO repeated")):
            restored = receiver.restore(checkpoint)
        np.testing.assert_array_equal(restored, expected)
        self.assertEqual(receiver.warm.memory, [7.])
        np.testing.assert_array_equal(receiver.controller.lower.dual, checkpoint["dual"])
        self.assertEqual(receiver.reference_timing, checkpoint["reference_timing"])
        receiver.warm.memory.append(8.)
        self.assertEqual(checkpoint["warm"].memory, [7.])

    def test_terminal_contract_and_legacy_checkpoint(self):
        original = environment()
        original.k, original.sim.state.time_sec = 80, 14400.
        original.controller.prepared = False
        checkpoint = original.checkpoint()
        receiver = environment()
        obs = receiver.restore(checkpoint)
        self.assertFalse(receiver.controller.prepared)
        np.testing.assert_array_equal(obs, np.zeros(len(receiver.observer.names)))
        checkpoint["contract"]["version"] = 1
        with self.assertRaisesRegex(ValueError, "contract"):
            receiver.restore(checkpoint)
        checkpoint.pop("contract")
        with self.assertRaisesRegex(ValueError, "contract"):
            receiver.restore(checkpoint)

    def test_saved_normalization_and_names_tamper_rejected(self):
        checkpoint = environment().checkpoint()
        for field_name in ("version", "implementation_sha256", "names"):
            with self.subTest(field=field_name):
                changed = copy.deepcopy(checkpoint)
                changed["contract"]["observation"][field_name] = "changed"
                receiver = environment()
                receiver.observer = Observation(receiver.cfg)
                with self.assertRaisesRegex(ValueError, "contract"):
                    receiver.restore(changed)

    def test_saved_simulator_config_mismatch_rejected(self):
        checkpoint = environment().checkpoint()
        checkpoint["sim"].cfg.network.freeway_lanes = 4
        receiver = environment()
        sim = receiver.sim
        with self.assertRaisesRegex(ValueError, "config contract"):
            receiver.restore(checkpoint)
        self.assertIs(receiver.sim, sim)


class SnapshotBootTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="wrapper-contract-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        path = self.source / "work/sdmpc_upper_ttt_20260922/matrix_common.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"raise AssertionError('snapshot imported')\n")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = dict(source_relative=path.relative_to(self.source).as_posix(),
                     snapshot_relative=path.relative_to(self.root).as_posix(),
                     size_bytes=path.stat().st_size, sha256=file_hash)
        entry.update({key: file_hash for key in ("source_sha256_before", "copied_sha256",
                                                "destination_sha256_after", "source_sha256_after")})
        self.manifest = dict(schema_version=1, complete=True, file_count=1,
                             total_bytes=entry["size_bytes"], files=[entry])
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        self.payload = path

    def test_tampered_source_rejected_before_imports(self):
        self.payload.write_bytes(self.payload.read_bytes().replace(b"snapshot", b"tampered"))
        with patch("builtins.__import__", wraps=__import__) as imports:
            with self.assertRaisesRegex(ValueError, "Snapshot file mismatch"):
                budget_runtime.boot(self.source)
        self.assertFalse(any(call.args[0] in ("matrix_common", "runtime", "src")
                             for call in imports.call_args_list))

    def test_original_source_without_manifest_rejected(self):
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            budget_runtime.boot(self.source / "work")

    def test_boot_retains_identity_and_completion_verifier(self):
        expected = self.source / "work/sdmpc_externality_ablation_20260923"
        modules = {name: SimpleNamespace(__file__=str(expected / (name + ".py"))) for name in
                   ("fixed_policy", "band_math", "prox_controller", "fixed_controller", "central",
                    "exception_controller", "anchor_controller", "externality_policy")}
        modules["matrix_common"] = SimpleNamespace(pin=lambda mask: mask, environment=lambda *args: None)
        modules["runtime"] = SimpleNamespace(load=lambda step: (None, Config(), Options()), restore_state=Mock())
        modules["anchor_controller"].PFOAnchorSDMPC = Lower
        modules["fixed_controller"].GridCoordinates = Coordinates
        modules["externality_policy"].audit_rows = Mock()
        modules["reused_model"] = SimpleNamespace()
        for root in (self.root, self.source):
            with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)), \
                    patch.object(sys, "dont_write_bytecode", True), patch.dict(budget_runtime.os.environ):
                rt = budget_runtime.boot(root)
            self.assertEqual(rt["verify"](), rt["snapshot_identity"])
        self.assertEqual(rt["root"], self.source.resolve())
        self.assertEqual(rt["snapshot_identity"]["manifest_sha256"],
                         hashlib.sha256((self.root / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(rt["verify"](), rt["snapshot_identity"])
        self.manifest["extra_metadata"] = "changed after boot"
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "identity"):
            rt["verify"]()

    def test_completion_verifier_rejects_payload_tamper(self):
        identity = budget_runtime.verify_frozen_source(self.source)
        self.payload.write_bytes(self.payload.read_bytes().replace(b"snapshot", b"tampered"))
        with self.assertRaisesRegex(ValueError, "Snapshot file mismatch"):
            budget_runtime.verify_frozen_source(self.source, identity)


if __name__ == "__main__":
    unittest.main()
