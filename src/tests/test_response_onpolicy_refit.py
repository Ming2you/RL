from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.tests.test_response_dqn_semantics import _replay
from work.run_response_onpolicy_refit import fit_gate_passed, replay_fit, run
from work.run_response_reference_coverage import train_variant


class OnPolicyRefitTests(unittest.TestCase):
    def model(self, *, mode='legacy_follower_runtime_v1', gamma=1, following=None):
        return SimpleNamespace(
            response_equivalence_mode=mode,
            config=SimpleNamespace(gamma=gamma, reward_scale=1, min_action_support=1),
            action_support_counts=np.array([2, 0]),
            q_values=Mock(side_effect=[np.array([[-5., 0.], [-4., 0.]]),
                                      np.array([[-4., 999.], [1e6, 1e6]]) if following is None else following]),
        )

    def test_terminal_target_ignores_future_and_nonterminal_keeps_bootstrap(self):
        replay = replace(_replay(), action_id=np.array([0, 0]))
        fit = replay_fit([self.model()], replay)
        self.assertEqual(fit['terminal_exact_target_scaled'], [-2])
        self.assertEqual(fit['terminal_behavior_mae_scaled'], 2)
        self.assertEqual(fit['all_behavior_td_mae_scaled'], 1)
        other = replay_fit([self.model(following=np.array([[-3., 999.], [-1e6, -1e6]]))], replay)
        self.assertEqual(other['terminal_behavior_mae_scaled'], 2)
        self.assertEqual(other['all_behavior_td_mae_scaled'], 1.5)

    def test_incompatible_mode_or_gamma_rejects(self):
        for model in (self.model(mode='other'), self.model(gamma=.99)):
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                replay_fit([model], _replay())

    def test_nonfinite_q_rejects(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            replay_fit([self.model(following=np.full((2, 2), np.nan))], _replay())

    def test_missing_true_terminal_rejects(self):
        with self.assertRaisesRegex(ValueError, 'true terminal'):
            replay_fit([self.model()], replace(_replay(), done=np.zeros(2)))

    def test_empty_supported_successor_rejects(self):
        model = self.model()
        model.action_support_counts[:] = 0
        with self.assertRaisesRegex(ValueError, 'empty supported'):
            replay_fit([model], _replay())

    def test_gate_requires_both_prespecified_improvements(self):
        before = {'terminal_behavior_mae_scaled': 2., 'all_behavior_td_mae_scaled': 1.}
        after = {'terminal_behavior_mae_scaled': 1., 'all_behavior_td_mae_scaled': 1.2}
        gate = {'terminal_mae_max_ratio': .5, 'all_td_mae_max_ratio': 1.25}
        self.assertTrue(fit_gate_passed(before, after, gate))
        for field, value in (('terminal_behavior_mae_scaled', 1.01), ('all_behavior_td_mae_scaled', 1.26)):
            self.assertFalse(fit_gate_passed(before, {**after, field: value}, gate))
        with self.assertRaises(ValueError):
            fit_gate_passed(before, after, {**gate, 'terminal_mae_max_ratio': float('nan')})

    def test_stop_prevents_reading_inputs_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'STOP').touch()
            with patch('work.run_response_onpolicy_refit.train_variant') as train:
                with self.assertRaises(InterruptedError):
                    run({'output_dir': str(root), 'start_only_after_output_dir': 'unused', 'additional_stop_files': []})
                train.assert_not_called()

    def test_changed_input_hash_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / 'old'
            previous.mkdir()
            (previous / 'status.json').write_text(json.dumps({'phase': 'ablation_complete'}))
            (previous / 'process.json').write_text(json.dumps({'state': 'exited'}))
            source = Path(directory) / 'source'
            source.write_text('changed')
            with patch('work.run_response_onpolicy_refit.train_variant') as train:
                with self.assertRaisesRegex(ValueError, 'input hash changed'):
                    run({'output_dir': str(Path(directory) / 'new'), 'start_only_after_output_dir': str(previous),
                         'additional_stop_files': [], 'implementation_files': [], 'input_sha256': {str(source): '0' * 64}})
                train.assert_not_called()

    def test_constant_feature_flag_is_not_passed_as_a_value(self):
        module = 'work.run_response_reference_coverage'
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                with patch(module + '.subprocess.run') as launch, \
                     patch(module + '._read_json', return_value={}), \
                     patch(module + '.validate_training_recipe'):
                    train_variant(Path(directory) / 'data.npz', Path(directory) / 'model', {
                        'common_training': {'group_resampling': False, 'mask_constant_features': enabled,
                                            'gradient_steps': 6000},
                    }, .1)
                command = launch.call_args.args[0]
                self.assertEqual('--mask-constant-features' in command, enabled)
                self.assertNotIn('True', command)
                self.assertNotIn('False', command)

    def test_stop_propagates_to_trainer_and_failure_becomes_pause(self):
        module = 'work.run_response_reference_coverage'
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / 'STOP'
            def interrupted(command, *, check):
                self.assertIn('--stop-file', command)
                self.assertIn(str(stop), command)
                stop.touch()
                raise subprocess.CalledProcessError(1, command)
            with patch(module + '.subprocess.run', side_effect=interrupted) as launch:
                with self.assertRaisesRegex(InterruptedError, 'during training'):
                    train_variant(Path(directory) / 'data.npz', Path(directory) / 'model', {
                        'common_training': {'group_resampling': False},
                    }, .1, stop_files=[stop])
                launch.assert_called_once()
                with self.assertRaisesRegex(InterruptedError, 'before training'):
                    train_variant(Path(directory) / 'data.npz', Path(directory) / 'other', {
                        'common_training': {'group_resampling': False},
                    }, .1, stop_files=[stop])
                launch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
