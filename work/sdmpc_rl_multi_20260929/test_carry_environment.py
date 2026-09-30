"""Carry environment regressions using the v1 fake runtime, without plant solves."""
import copy
from dataclasses import dataclass, field, fields, is_dataclass
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from budget_controller import BudgetController, InvalidReference
from budget_env import BudgetEnv, Observation
from budget_runtime import plain


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
        self.lower = np.array([-100., 0.])
        self.upper = np.array([100., 1.])

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
        self.last_budget = np.array([-20., 1500.])
        self.coords = Coordinates()
        self.counts = {"scalar_rollouts": 0}
        self.execution_archives = []

    def begin(self, *args):
        self.execution_archives = []
        self.derivative_rows = []

    def evaluate(self, point):
        return SimpleNamespace(total_ttt=10. + point[1],
                               budget_vector=np.array([point[0], 1000.]),
                               physical_valid=True, control_valid=True,
                               states=["DO_NOT_EXPORT"])

    def execution_check(self, point, budget):
        ev = self.evaluate(point)
        residual = ev.budget_vector - budget
        return dict(physical_control_valid=True, budget_feasible=bool(np.all(residual <= 0.)),
                    achieved=ev.budget_vector.tolist(), original_residual=residual.tolist(),
                    scaled_excess=np.maximum(residual, 0.).tolist(), point=list(point),
                    budget=list(budget), ttt=ev.total_ttt)

    def solve(self, budget, seed, incoming):
        point = np.array([budget[0], .25])
        check = self.execution_check(point, budget)
        control = self.coords.decode(point)
        control.N_P_star, control.N_UF_star = budget
        self.execution_archives.append({})
        return dict(point=point, budget=budget.copy(), control=control,
                    evaluation=self.evaluate(point), feasible=check["budget_feasible"],
                    dual=incoming + 1., converged=True, stationarity=0.,
                    original_residual=check["original_residual"], rows=[], local_rows=[])


@dataclass
class Log:
    freeway_ttt: float = 2.
    urban_ttt: float = 3.
    diagnostics: dict = field(default_factory=lambda: {"conservation": 0.})


class Simulator:
    def __init__(self, cfg):
        self.cfg, self.state = cfg, State()
        self.total_ttt, self.freeway_ttt, self.urban_ttt = 5., 2., 3.
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


class Warm:
    def __init__(self):
        self.memory, self.calls = [7.], 0

    def solve(self, state, unused, forecast, previous):
        self.calls += 1
        self.memory.append(state.time_sec)
        state.time_sec = -1.
        return SimpleNamespace(control=Control([3., .5]))


def config_fixture_plain(value):
    """Match the frozen config serializer's public dynamic-field behavior."""
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


def runtime():
    return dict(cfg=Config(), options=Options(), baseline=Lower, coordinates=Coordinates,
                audit_rows=lambda rows: None, snapshot_identity={"manifest_sha256": "a" * 64},
                rc=SimpleNamespace(inventory=lambda state, cfg: {"vehicles": 2.},
                                   to_plain_dict=config_fixture_plain))


def environment(k=6, prepared=True, guard_mode="physical"):
    env = BudgetEnv.__new__(BudgetEnv)
    env.rt, env.guard_mode = runtime(), guard_mode
    env.cfg, env.reward_scale = env.rt["cfg"], 100.
    env.profile_hash = "same-profile"
    env.sim, env.warm, env.previous = Simulator(env.cfg), Warm(), Control()
    env.k, env.warmup_ttt, env.last_fallback = k, 5., False
    env.sim.state.time_sec = k * 180.
    env.last_requested = np.array([-25., 500.])
    env.last_executed = np.array([-20., 1500.])
    env.last_slack = np.zeros(2)
    env.forecast = [SimpleNamespace(freeway_mainline={"f": 1.}, urban_boundary={}, ramp_arrival={},
                                   incident_capacity_factor=1., freeway_lane_loss={})] * 3
    env.profile = SimpleNamespace(horizon=lambda *args: env.forecast)
    env.controller = BudgetController(env.rt, guard_mode)
    if k == 5:
        env.controller.lower.last_budget = None
    env.observer = Observation(env.cfg)
    env.observation_timing = dict(observation_wall_seconds=.1, observation_cpu_seconds=.05)
    if prepared:
        with fake_modules():
            env.prepare()
        env.observer.encode(env)
    return env


def fake_modules():
    return patch.dict(sys.modules, {
        "fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]]),
        "exception_controller": SimpleNamespace(choose_archive=lambda rows: None),
    })


class PreparationTests(unittest.TestCase):
    def test_regular_prepare_uses_previous_without_pfo(self):
        env = environment(prepared=False)
        with fake_modules(), patch.object(env.warm, "solve", side_effect=AssertionError("PFO called")), \
                patch.object(env.controller, "prepare_reference", wraps=env.controller.prepare_reference) as prepare:
            env.prepare()
        prepare.assert_called_once_with(env.sim.state, env.forecast, env.previous, env.previous,
                                        reference_source="previous")
        self.assertEqual(env.controller.reference["source"], "previous")
        self.assertEqual(env.reference_timing["pfo_calls"], 0)
        self.assertEqual(env.reference_timing["pfo_wall_seconds"], 0.)
        self.assertEqual(env.reference_timing["pfo_cpu_seconds"], 0.)
        self.assertFalse(env.reference_timing["reference_recovery"])

    def test_initial_prepare_calls_pfo_once_with_copied_state(self):
        env = environment(k=5, prepared=False)
        with fake_modules():
            env.prepare()
        self.assertEqual(env.warm.calls, 1)
        self.assertEqual(env.sim.state.time_sec, 900.)
        self.assertEqual(env.controller.reference["source"], "pfo_initial")
        self.assertEqual(env.reference_timing["pfo_calls"], 1)
        self.assertFalse(env.reference_timing["reference_recovery"])
        np.testing.assert_array_equal(env.controller.action_anchor, env.controller.reference["budget"])

    def test_invalid_reference_alone_recovers_with_pfo(self):
        env = environment(prepared=False)
        prepare = env.controller.prepare_reference

        def previous_invalid(*args, **kwargs):
            if kwargs["reference_source"] == "previous":
                raise InvalidReference("synthetic current-state violation")
            return prepare(*args, **kwargs)

        with fake_modules(), patch.object(env.controller, "prepare_reference", side_effect=previous_invalid) as calls:
            env.prepare()
        self.assertEqual([call.kwargs["reference_source"] for call in calls.call_args_list],
                         ["previous", "pfo_recovery"])
        self.assertEqual(env.warm.calls, 1)
        self.assertEqual(env.controller.reference["source"], "pfo_recovery")
        self.assertTrue(env.reference_timing["reference_recovery"])
        self.assertEqual(env.reference_timing["pfo_calls"], 1)
        np.testing.assert_array_equal(env.controller.action_anchor, [-20., 1500.])

    def test_unexpected_prepare_exceptions_propagate_without_pfo(self):
        for error in (ValueError("bug"), RuntimeError("bug"), TypeError("bug")):
            with self.subTest(error=type(error).__name__):
                env = environment(prepared=False)
                with patch.object(env.controller, "prepare_reference", side_effect=error), \
                        self.assertRaises(type(error)) as raised:
                    env.prepare()
                self.assertIs(raised.exception, error)
                self.assertEqual(env.warm.calls, 0)
                self.assertEqual(env.sim.logs, [])

    def test_invalid_recovery_and_initial_reference_fail_closed(self):
        for k in (5, 6):
            with self.subTest(k=k):
                env = environment(k=k, prepared=False)
                with patch.object(env.controller, "prepare_reference", side_effect=InvalidReference("invalid")), \
                        self.assertRaises(InvalidReference):
                    env.prepare()
                self.assertEqual(env.warm.calls, 1)
                self.assertFalse(env.controller.prepared)
                self.assertEqual(env.k, k)
                self.assertEqual(env.sim.logs, [])
                with self.assertRaisesRegex(RuntimeError, "Missing reference"):
                    env.step(np.zeros(2))

    def test_pfo_failure_propagates_without_plant_step(self):
        env = environment(prepared=False)
        with patch.object(env.controller, "prepare_reference", side_effect=InvalidReference("invalid")), \
                patch.object(env.warm, "solve", side_effect=RuntimeError("PFO failed")), \
                self.assertRaisesRegex(RuntimeError, "PFO failed"):
            env.prepare()
        self.assertFalse(env.controller.prepared)
        self.assertEqual(env.sim.logs, [])


class ObservationAndStepTests(unittest.TestCase):
    def test_reference_identity_and_carried_budget_are_observed(self):
        env = environment()
        names = env.observer.names
        self.assertTrue(any(name.startswith("reference/coordinate/") for name in names))
        self.assertFalse(any("pfo" in name.lower() for name in names))
        for index, source in enumerate(("previous", "pfo_initial", "pfo_recovery")):
            with self.subTest(source=source):
                env.controller.reference["source"] = source
                values = dict(zip(names, env.observer.encode(env)))
                np.testing.assert_allclose([values[f"memory/action_anchor/{i}"] for i in range(2)],
                                           [-.02, .15])
                np.testing.assert_allclose([values[f"memory/reference_budget/{i}"] for i in range(2)],
                                           [0., .1])
                self.assertEqual([values[f"memory/reference_source/{i}"] for i in range(3)],
                                 [float(i == index) for i in range(3)])
                self.assertAlmostEqual(values["memory/reference_TTT/0"], .01)
                self.assertAlmostEqual(values["memory/previous_requested/0"], -.025)

    def test_failed_request_remains_in_audit_and_next_observation(self):
        env = environment()
        action = np.array([-.5, -1.])
        with fake_modules():
            obs, reward, terminal, audit = env.step(action, actor_seconds=.2, actor_cpu_seconds=.1)
        self.assertEqual(audit["selection_source"], "reference_fallback")
        np.testing.assert_array_equal(audit["action_requested"], action)
        np.testing.assert_array_equal(audit["B_requested"][0], [-45., 500.])
        np.testing.assert_array_equal(audit["B_executed"], [0., 1000.])
        np.testing.assert_array_equal(env.last_requested, [-45., 500.])
        np.testing.assert_array_equal(env.last_executed, [0., 1000.])
        np.testing.assert_array_equal(env.controller.action_anchor, env.last_executed)
        values = dict(zip(env.observer.names, obs))
        self.assertAlmostEqual(values["memory/previous_requested/0"], -.045)
        self.assertEqual(values["memory/fallback/0"], 1.)
        self.assertFalse(terminal)
        self.assertEqual(reward, -.05)

    def test_interval_reward_accounting_and_true_terminal(self):
        for k in (6, 78, 79):
            with self.subTest(k=k):
                env = environment(k=k)
                before = env.sim.total_ttt
                with fake_modules(), patch.object(env.warm, "solve", side_effect=AssertionError("PFO called")), \
                        patch.object(env, "prepare", wraps=env.prepare) as prepare:
                    obs, reward, terminal, audit = env.step(np.zeros(2), actor_seconds=.2, actor_cpu_seconds=.1)
                self.assertEqual(reward, -(env.sim.total_ttt - before) / 100.)
                self.assertEqual(audit["interval_ttt"], 5.)
                self.assertEqual(audit["plant_log"]["freeway_ttt"] + audit["plant_log"]["urban_ttt"], 5.)
                self.assertEqual(terminal, k == 79)
                self.assertEqual(audit["terminated"], terminal)
                self.assertIs(audit["truncated"], False)
                self.assertEqual(audit["step"], k)
                self.assertEqual(audit["control_step"], k - 5)
                self.assertEqual(prepare.call_count, int(not terminal))
                if terminal:
                    self.assertEqual(env.sim.state.time_sec, 14400.)
                    np.testing.assert_array_equal(obs, np.zeros(len(env.observer.names)))
                    with self.assertRaisesRegex(RuntimeError, "terminal"):
                        env.step(np.zeros(2))
                    with self.assertRaisesRegex(RuntimeError, "terminal"):
                        env.prepare()
                phases = ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor")
                for clock in ("wall", "cpu"):
                    self.assertAlmostEqual(audit[f"decision_{clock}_seconds"],
                                           sum(audit[f"{p}_{clock}_seconds"] for p in phases))

    def test_bad_interval_accounting_is_not_a_successful_transition(self):
        for total, log in ((-1., Log()), (float("nan"), Log()), (5., Log(4., 4.))):
            with self.subTest(total=total, log=log):
                env = environment()

                def bad_step(*args):
                    env.sim.total_ttt += total
                    return log

                with fake_modules(), patch.object(env.sim, "step", side_effect=bad_step), \
                        self.assertRaisesRegex(RuntimeError, "TTT accounting"):
                    env.step(np.zeros(2))
                self.assertEqual(env.k, 6)


class CheckpointTests(unittest.TestCase):
    def assert_rejected_before_replacement(self, receiver, checkpoint, message="contract"):
        sim, controller, warm = receiver.sim, receiver.controller, receiver.warm
        with patch.dict(receiver.rt, baseline=Mock(side_effect=AssertionError("lower constructed"))), \
                patch.object(receiver, "close", side_effect=AssertionError("live state closed")), \
                self.assertRaisesRegex(ValueError, message):
            receiver.restore(checkpoint)
        self.assertIs(receiver.sim, sim)
        self.assertIs(receiver.controller, controller)
        self.assertIs(receiver.warm, warm)

    def test_contract_mismatches_rejected_before_replacing_live_state(self):
        checkpoint = environment().checkpoint()
        changes = {
            "guard": lambda env: setattr(env, "guard_mode", "h3"),
            "reward": lambda env: setattr(env, "reward_scale", 200.),
            "config": lambda env: setattr(env.cfg.network, "freeway_lanes", 4),
            "options": lambda env: setattr(env.rt["options"], "max_iterations", 5),
            "source": lambda env: env.rt["snapshot_identity"].update(manifest_sha256="b" * 64),
            "delay": lambda env: setattr(env.observer, "delay", env.observer.delay + 1),
            "names": lambda env: setattr(env.observer, "names", ("wrong",)),
        }
        for label, change in changes.items():
            with self.subTest(label=label):
                receiver = environment()
                change(receiver)
                self.assert_rejected_before_replacement(receiver, checkpoint)

    def test_dynamic_config_options_and_saved_simulator_mismatches(self):
        original = environment()
        original.cfg.network.terminal_zero_gradient = True
        original.cfg.mpc = Options()
        original.cfg.mpc.leader_rollout_box_walk = True
        original.rt["options"].runtime_tolerance = .01
        for location in ("network", "mpc", "options", "runtime_cfg", "saved_sim"):
            with self.subTest(location=location):
                checkpoint = original.checkpoint()
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
                elif location == "runtime_cfg":
                    receiver.rt["cfg"] = copy.deepcopy(receiver.cfg)
                    receiver.rt["cfg"].network.terminal_zero_gradient = False
                else:
                    checkpoint["sim"].cfg.network.terminal_zero_gradient = False
                self.assert_rejected_before_replacement(receiver, checkpoint)

    def test_saved_schema_tamper_rejected_with_fresh_observer(self):
        for field_name in ("version", "implementation_sha256", "names"):
            with self.subTest(field=field_name):
                checkpoint = environment().checkpoint()
                checkpoint["contract"]["observation"][field_name] = "changed"
                receiver = environment()
                receiver.observer = Observation(receiver.cfg)
                self.assert_rejected_before_replacement(receiver, checkpoint)

    def test_profile_missing_reference_and_legacy_contract_rejected(self):
        for label in ("profile", "reference", "legacy", "missing_contract"):
            with self.subTest(label=label):
                checkpoint = environment().checkpoint()
                message = "contract"
                if label == "profile":
                    checkpoint["profile_hash"] = "other"
                    message = "demand"
                elif label == "reference":
                    checkpoint["prepared"] = None
                    message = "prepared reference"
                elif label == "legacy":
                    checkpoint["contract"]["version"] = 2
                else:
                    checkpoint.pop("contract")
                self.assert_rejected_before_replacement(environment(), checkpoint, message)

    def test_serialized_restore_and_next_step_are_exact_without_pfo(self):
        for source in ("previous", "pfo_initial", "pfo_recovery"):
            with self.subTest(source=source):
                original = environment(k=5 if source == "pfo_initial" else 6)
                original.controller.reference["source"] = source
                original.reference_timing.update(pfo_calls=int(source != "previous"),
                                                 reference_recovery=source == "pfo_recovery")
                original.cfg.network.public_switch = True
                original.rt["options"].runtime_tolerance = .01
                checkpoint = pickle.loads(pickle.dumps(original.checkpoint()))
                expected = original.observer.encode(original)
                receiver = environment()
                receiver.cfg.network.public_switch = True
                receiver.rt["options"].runtime_tolerance = .01
                receiver.observer = Observation(receiver.cfg)
                with fake_modules(), patch.object(Warm, "solve", side_effect=AssertionError("PFO repeated")), \
                        patch.object(receiver, "prepare", side_effect=AssertionError("prepare on restore")):
                    np.testing.assert_array_equal(receiver.restore(checkpoint), expected)
                self.assertEqual(receiver.warm.memory, original.warm.memory)
                self.assertEqual(receiver.reference_timing, checkpoint["reference_timing"])
                np.testing.assert_array_equal(receiver.controller.lower.dual, checkpoint["dual"])
                np.testing.assert_array_equal(receiver.controller.action_anchor, original.controller.action_anchor)
                with fake_modules(), patch.object(Warm, "solve", side_effect=AssertionError("PFO repeated")), \
                        patch("budget_env.time.perf_counter", return_value=20.), \
                        patch("budget_env.time.process_time", return_value=10.):
                    left = original.step(np.array([.25, 0.]), actor_seconds=.2, actor_cpu_seconds=.1)
                    right = receiver.step(np.array([.25, 0.]), actor_seconds=.2, actor_cpu_seconds=.1)
                np.testing.assert_array_equal(left[0], right[0])
                self.assertEqual(left[1:3], right[1:3])
                self.assertEqual(plain(left[3]), plain(right[3]))
                self.assertEqual(plain(original.sim.state), plain(receiver.sim.state))
                receiver.warm.memory.append(8.)
                self.assertNotEqual(receiver.warm.memory, checkpoint["warm"].memory)

    def test_terminal_restore_returns_zero_observation_without_preparation(self):
        original = environment()
        original.k, original.sim.state.time_sec = 80, 14400.
        original.controller.prepared = False
        checkpoint = pickle.loads(pickle.dumps(original.checkpoint()))
        receiver = environment()
        with patch.object(Warm, "solve", side_effect=AssertionError("PFO repeated")), \
                patch.object(BudgetController, "prepare_reference", side_effect=AssertionError("reference prepared")):
            obs = receiver.restore(checkpoint)
        self.assertFalse(receiver.controller.prepared)
        np.testing.assert_array_equal(obs, np.zeros(len(receiver.observer.names)))


if __name__ == "__main__":
    unittest.main()
