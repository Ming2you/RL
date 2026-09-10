"""Versioned response equivalence cannot silently change old replay or commits."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rl_leader.response_dqn_collect import (
    _candidate_trial, _validate_policy_equivalence, commit_action, evaluate_executable_responses,
)
from rl_leader.response_dqn_data import merge_frozen_response_replays
from src.tests.test_response_dqn_semantics import _replay


MODE = 'post_commit_continuation_v1'


class TrialEnv:
    response_equivalence_mode = MODE

    def __init__(self):
        self.previous = np.array([1, 2], np.float32)
        self.step = 0
        self.controller = SimpleNamespace(nash_solver=object())

    def prepare_pstack_anchor_context(self):
        return SimpleNamespace(state_fingerprint='state')

    def step_prepared_optimizer_anchor(self, context):
        self.step += 1
        return np.array([self.step]), -1., False, {'validity_gate_pass': True}, None

    def step_anchored_candidate(self, residual, context):
        return self.step_prepared_optimizer_anchor(context)[:4]

    def response_vector(self, previous):
        return previous

    def _follower_runtime_fingerprint(self, follower):
        return 'legacy'


def identity(env, *, interval_reward, terminal):
    return {'step': env.step, 'reward': interval_reward, 'done': terminal}


class ContinuationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.catalog = SimpleNamespace(size=2, actions=[SimpleNamespace(action_id=i) for i in range(2)],
                                       residual=lambda i: np.array([i]))

    def test_new_and_legacy_replays_must_not_merge(self):
        for old in (_replay(), _replay(response_equivalence_mode='legacy_follower_runtime_v1')):
            new = _replay(response_equivalence_mode=MODE)
            with self.assertRaisesRegex(ValueError, 'response_equivalence_mode'):
                merge_frozen_response_replays([old, new], source='test')
        merged = merge_frozen_response_replays([new, new], source='test')
        self.assertEqual(merged.manifest['response_equivalence_mode'], MODE)

    def test_continuation_validation_errors_are_not_masked_as_infeasible_actions(self):
        with patch('rl_leader.response_dqn_collect._continuation_identity', side_effect=ValueError('unsupported')):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                _candidate_trial(TrialEnv(), object(), self.catalog, 1)

    def test_identity_is_based_on_committed_clone_and_source_is_untouched(self):
        source = TrialEnv()
        with patch('rl_leader.response_dqn_collect._continuation_identity', side_effect=identity):
            evaluated = evaluate_executable_responses(source, np.array([0]), self.catalog)
            self.assertEqual(source.step, 0)
            self.assertEqual(evaluated.response_equivalence_mode, MODE)
            self.assertEqual(evaluated.response_mask.groups, ((0, 1),))
            commit_action(source, evaluated, self.catalog, 0)
            self.assertEqual(source.step, 1)

    def test_commit_mismatch_fails_instead_of_recording_wrong_transition(self):
        source = TrialEnv()
        with patch('rl_leader.response_dqn_collect._continuation_identity', side_effect=identity):
            evaluated = evaluate_executable_responses(source, np.array([0]), self.catalog)
            source.step = 4
            with self.assertRaisesRegex(ValueError, 'preview.*commit'):
                commit_action(source, evaluated, self.catalog, 0)

    def test_legacy_path_does_not_use_new_identity(self):
        source = TrialEnv()
        source.response_equivalence_mode = 'legacy_follower_runtime_v1'
        with patch('rl_leader.response_dqn_collect._continuation_identity', side_effect=AssertionError):
            result = _candidate_trial(source, object(), self.catalog, 1)
        self.assertEqual(result[1], 'legacy')

    def test_unknown_mode_rejected_before_simulator_preview(self):
        source = TrialEnv()
        source.response_equivalence_mode = 'unrecognized'
        with self.assertRaisesRegex(ValueError, 'equivalence'):
            evaluate_executable_responses(source, np.array([0]), self.catalog)
        self.assertEqual(source.step, 0)

    def test_legacy_policy_cannot_silently_use_new_mask_semantics(self):
        with self.assertRaisesRegex(ValueError, 'equivalence'):
            _validate_policy_equivalence([SimpleNamespace()], SimpleNamespace(response_equivalence_mode=MODE))
        _validate_policy_equivalence([SimpleNamespace(response_equivalence_mode=MODE)],
                                     SimpleNamespace(response_equivalence_mode=MODE))

    def test_model_checkpoint_retains_mask_semantics(self):
        import tempfile
        from pathlib import Path
        import torch
        from rl_leader.response_dqn import ResponseDQNConfig, load_trained_response_dqn, train_response_dqn_member
        torch.set_num_threads(1)
        replay = _replay(response_equivalence_mode=MODE)
        model = train_response_dqn_member(
            replay, np.eye(2, dtype=np.float32), catalog_fingerprint='catalog-test',
            config=ResponseDQNConfig(gradient_steps=1, hidden=(8,)), bootstrap=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'new.pt'
            model.save(path)
            restored = load_trained_response_dqn(path)
            self.assertEqual(restored.response_equivalence_mode, MODE)
            payload = model.checkpoint()
            payload.pop('response_equivalence_mode')
            torch.save(payload, path)
            self.assertEqual(load_trained_response_dqn(path).response_equivalence_mode, 'legacy_follower_runtime_v1')


if __name__ == '__main__':
    unittest.main()
