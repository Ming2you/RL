import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np

from retention_actor import RetentionActor, RecoveryActor
from p10_analysis import validate_experience
import p10_worker as worker


NAMES = [f"state/urban_movement_queue/{i}" for i in range(78)] + [
    "memory/action_anchor/0", "memory/action_anchor/1", "memory/previous_executed/0",
    "memory/previous_slack/0", "memory/remaining/0", "memory/fallback/0"]
NAMES += [f"unused/{i}" for i in range(2367 - len(NAMES))]


def spec(retain=False, end=30):
    return dict(format="sdmpc-retention-p10-v1", retain_nuf_during_bind=retain,
        recovery=dict(format="sdmpc-recovery-actor-p9-v1", restore_nuf=True, abort_on_fallback=False,
            base=dict(format="sdmpc-perimeter-feedback-actor-v1", params=dict(
                uq_on=-1e12, uq_off=-1e12, delta=50., margin=0., bind_start=16, bind_end=end,
                pre_mode="hold", post_mode="return"))))


def obs(step, np_anchor=-100., nuf=6000., fallback=False):
    out = np.zeros(2367, np.float32)
    values = {"memory/action_anchor/0": np_anchor/1000., "memory/action_anchor/1": nuf/10000.,
              "memory/previous_executed/0": np_anchor/1000., "memory/remaining/0": (76-step)/75.,
              "memory/fallback/0": float(fallback)}
    for key, value in values.items():
        out[NAMES.index(key)] = value
    return out


class FakeEnv:
    def __init__(self):
        self.k, self.reward_scale, self.training_seed, self.profile_hash = 20, 100., 8701, "test-profile"
        self.sim = SimpleNamespace(state=SimpleNamespace(time_sec=3600.))
        self.observer = SimpleNamespace(names=NAMES)

    def step(self, action, *args, **kwargs):
        self.k += 1
        terminal = self.k == 80
        self.sim.state.time_sec += 180.
        row = dict(control_step=self.k-6, action=list(action), interval_ttt=1., total_ttt=float(self.k))
        return (np.zeros(2367, np.float32) if terminal else obs(self.k-4)), -.01, terminal, row


class P10Tests(unittest.TestCase):
    def test_no_retention_equals_recovery_at_both_window_lengths(self):
        for end in (25, 30):
            a, b = RetentionActor(spec(False, end), NAMES), RecoveryActor(spec(False, end)["recovery"], NAMES)
            for step in range(1, 76):
                o = obs(step, -100.-step*5., 6000.-(step % 7)*250., step % 4 == 0)
                np.testing.assert_array_equal(a.act(o), b.act(o))

    def test_retention_changes_only_nuf_while_binding(self):
        a, b = RetentionActor(spec(True), NAMES), RetentionActor(spec(False), NAMES)
        np.testing.assert_array_equal(a.act(obs(16)), b.act(obs(16)))
        x,y = a.act(obs(17, -300., 3500., True)), b.act(obs(17, -300., 3500., True))
        np.testing.assert_array_equal(x, [-1., 1.])
        np.testing.assert_array_equal(y, [-1., 0.])

    def test_retention_target_is_latched_and_recovers_after_window(self):
        a = RetentionActor(spec(True, 25), NAMES)
        a.act(obs(16, -100., 5500.))
        for step in range(17, 26):
            np.testing.assert_array_equal(a.act(obs(step, -300., 3500.)), [-1., 1.])
        np.testing.assert_array_equal(a.act(obs(26, -300., 3500.)), [1., 1.])
        np.testing.assert_array_equal(a.act(obs(27, -100., 5500.)), [0., 0.])

    def test_no_cross_episode_state_or_spec_mutation(self):
        s = spec(True)
        before = copy.deepcopy(s)
        a, b = RetentionActor(s, NAMES), RetentionActor(s, NAMES)
        for step in range(16, 76):
            o = obs(step, -100.-step, 5500.-step*11.)
            np.testing.assert_array_equal(a.memory(), b.memory())
            np.testing.assert_array_equal(a.act(o), b.act(o))
        self.assertEqual(s, before)

    def test_invalid_settings_rejected(self):
        s = spec()
        s["retain_nuf_during_bind"] = 1
        with self.assertRaises(ValueError): RetentionActor(s, NAMES)
        s = spec(); s["recovery"]["abort_on_fallback"] = True
        with self.assertRaises(ValueError): RetentionActor(s, NAMES)

    def test_full_capture_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            worker.OUT = Path(temp)
            worker.SCENARIO = "sweet_155_w"
            worker.STOP_ROOTS = (Path(temp),)
            worker.EXPECTED_TTT = 80.
            original = worker.probe.compact
            worker.probe.compact = lambda row: row
            try:
                env = FakeEnv()
                rows = []
                option = SimpleNamespace(name="carry", spec=dict(name="carry"), act=lambda env, o: np.zeros(2, np.float32))
                worker.run_to_end(env, obs(16), option, rows, time.time()+60)
                xp = Path(temp) / "experience/carry.npz"
                meta = json.loads(xp.with_suffix(".json").read_text())
                with np.load(xp, allow_pickle=False) as payload:
                    arrays = {k: payload[k].copy() for k in payload.files}
                branch = dict(rows=rows, ttt=80., option=dict(name="carry"))
                carry = dict(scenario="sweet_155_w", seed=8701, profile_sha256="test-profile")
                validate_experience(arrays, meta, branch, carry)
                bad = copy.deepcopy(arrays); bad["terminal"][-1] = False
                with self.assertRaises(ValueError): validate_experience(bad, meta, branch, carry)
                bad = copy.deepcopy(arrays); bad["next_obs"][0,0] = 1.
                with self.assertRaises(AssertionError): validate_experience(bad, meta, branch, carry)
                badmeta = copy.deepcopy(meta); badmeta["training_seed"] = None
                with self.assertRaises(ValueError): validate_experience(arrays, badmeta, branch, carry)
                bad = copy.deepcopy(arrays); bad["reward"][10] += .1
                with self.assertRaises(AssertionError): validate_experience(bad, meta, branch, carry)
            finally:
                worker.probe.compact = original

    def test_stop_precedes_any_simulator_step(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "STOP").touch()
            worker.STOP_ROOTS = (Path(temp),)
            env = FakeEnv()
            with self.assertRaisesRegex(RuntimeError, "STOP"):
                worker.run_to_end(env, obs(16), SimpleNamespace(), [], time.time()+60)
            self.assertEqual(env.k, 20)


if __name__ == "__main__":
    unittest.main()
