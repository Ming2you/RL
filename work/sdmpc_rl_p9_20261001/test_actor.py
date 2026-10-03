import copy
import unittest
import numpy as np
from actor import RecoveryActor, PerimeterActor


NAMES = [f"state/urban_movement_queue/{i}" for i in range(78)] + [
    "memory/action_anchor/0", "memory/action_anchor/1", "memory/previous_executed/0",
    "memory/previous_slack/0", "memory/remaining/0", "memory/fallback/0"]
NAMES += [f"unused/{i}" for i in range(2367 - len(NAMES))]
BASE = dict(format="sdmpc-perimeter-feedback-actor-v1", params=dict(
    uq_on=-1e12, uq_off=-1e12, delta=50., margin=0., bind_start=16, bind_end=30,
    pre_mode="hold", post_mode="return"))


def spec(nuf=True, abort=False):
    return dict(format="sdmpc-recovery-actor-p9-v1", base=copy.deepcopy(BASE),
                restore_nuf=nuf, abort_on_fallback=abort)


def obs(step, np_anchor=-100., nuf_anchor=6000., fallback=False):
    out = np.zeros(2367, np.float32)
    values = {"memory/action_anchor/0": np_anchor/1000., "memory/action_anchor/1": nuf_anchor/10000.,
              "memory/previous_executed/0": np_anchor/1000., "memory/remaining/0": (76-step)/75.,
              "memory/fallback/0": float(fallback)}
    for key, value in values.items():
        out[NAMES.index(key)] = value
    return out


class ActorTests(unittest.TestCase):
    def test_legacy_action_equivalence(self):
        a, b = RecoveryActor(spec(False, False), NAMES), PerimeterActor(BASE, NAMES)
        for step in range(1, 76):
            o = obs(step, -100. - step*4.7, 6000.-(step % 11)*100., step % 3 == 0)
            np.testing.assert_array_equal(a.act(o), b.act(o))

    def test_return_restores_both_and_is_clipped(self):
        a = RecoveryActor(spec(), NAMES)
        a.act(obs(16))
        for step in range(17, 31):
            np.testing.assert_array_equal(a.act(obs(step, -500., 3500.)), [-1., 0.])
        np.testing.assert_array_equal(a.act(obs(31, -500., 3500.)), [1., 1.])
        np.testing.assert_array_equal(a.act(obs(32, -100., 6000.)), [0., 0.])

    def test_pre_window_fallback_ignored(self):
        a = RecoveryActor(spec(True, True), NAMES)
        a.act(obs(15))
        np.testing.assert_array_equal(a.act(obs(16, fallback=True)), [-1., 0.])
        self.assertFalse(a.aborted)

    def test_fallback_returns_next_decision_and_latches(self):
        a = RecoveryActor(spec(True, True), NAMES)
        a.act(obs(16))
        np.testing.assert_array_equal(a.act(obs(17, -400., 3500., True)), [1., 1.])
        self.assertTrue(a.aborted)
        np.testing.assert_array_equal(a.act(obs(18, -350., 4500.)), [1., 1.])

    def test_no_latent_state_between_episodes(self):
        sequence = [obs(step, -50.-step*10., 6000.-step*30., step == 24) for step in range(16, 76)]
        actors = [RecoveryActor(spec(True, True), NAMES) for _ in range(2)]
        outputs = [[a.act(o) for o in sequence] for a in actors]
        np.testing.assert_array_equal(*outputs)

    def test_invalid_input_and_nonsequential_rejected(self):
        a = RecoveryActor(spec(), NAMES)
        with self.assertRaises(ValueError):
            a.act(obs(16).astype(np.float64))
        a.act(obs(16))
        with self.assertRaises(ValueError):
            a.act(obs(18))

    def test_identity_changes_with_behavior(self):
        hashes = {RecoveryActor(spec(n, a), NAMES).sha256 for n in (False, True) for a in (False, True)}
        self.assertEqual(len(hashes), 4)


if __name__ == "__main__":
    unittest.main()
