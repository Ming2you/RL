"""Separate extended optimization from changes to the conservative objective."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from rl_leader.response_dqn import ResponseDQNConfig
from src.tests.test_response_nonlinear_coverage import episode
from work.run_response_cost_refinement import path_rankings, run, verify_long_control_prefix


def model(steps=6000):
    return SimpleNamespace(seed=17, config=ResponseDQNConfig(gamma=1, value_parameterization='finite_horizon_cost_v1',
                            conservative_alpha=.1, gradient_steps=steps), training_losses=list(range(steps)),
                           normalizer=SimpleNamespace(mean=np.zeros(2)), candidate_features=np.eye(10),
                           action_support_counts=np.ones(10, dtype=int), value_contract='same')


class CostRefinementTests(unittest.TestCase):
    def test_long_control_reproduces_loss_prefix_and_only_extends_steps(self):
        old, new = model(), model(24000)
        self.assertEqual(verify_long_control_prefix([old], [new])['matched_loss_prefix_steps'], 6000)
        new.training_losses[100] += 1
        with self.assertRaises(AssertionError):
            verify_long_control_prefix([old], [new])
        for change in ({'seed': 18}, {'config': replace(new.config, conservative_alpha=.01)},
                       {'normalizer': SimpleNamespace(mean=np.ones(2))}, {'value_contract': 'different'}):
            bad = model(24000)
            for k, v in change.items():
                setattr(bad, k, v)
            with self.subTest(change=change), self.assertRaises((ValueError, AssertionError)):
                verify_long_control_prefix([old], [bad])

    def test_rankings_keep_bootstrap_separate_from_frozen_return(self):
        source = episode()
        arrays = {name: getattr(source, name)[1:].copy() for name in source.__dataclass_fields__ if name != 'manifest'}
        arrays['action_id'][0] = 6
        branch = replace(source, **arrays, manifest={**source.manifest, 'transition_count': 74})
        q = np.full((75, 10), -10.)
        q[0, 5], q[1, 6] = -8, -9
        nxt = np.full((1, 10), -5.)
        m = model()
        m.q_values = Mock(side_effect=[q, nxt])
        result = path_rankings([m], source, branch)
        self.assertEqual(result['source_step0_q5_minus_q0_scaled'], 2)
        self.assertEqual(result['source_step1_q6_minus_q0_scaled'], 1)
        self.assertAlmostEqual(result['quadratic_first_ensemble_greedy_td_target_scaled'], -5.01)
        self.assertAlmostEqual(result['quadratic_recorded_tail_return_scaled'], -.74)
        self.assertIn('never a training label', result['interpretation'])
        with self.assertRaises(ValueError):
            path_rankings([m], source, source)

    def test_stop_prevents_inputs_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'STOP').touch()
            with patch('work.run_response_cost_refinement.train_variant') as train:
                with self.assertRaises(InterruptedError):
                    run({'output_dir': str(root), 'start_only_after_output_dir': 'unused', 'additional_stop_files': []})
                train.assert_not_called()

    def test_changed_input_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / 'old'
            previous.mkdir()
            (previous / 'status.json').write_text(json.dumps({'phase': 'ablation_complete'}))
            (previous / 'process.json').write_text(json.dumps({'state': 'exited'}))
            source = Path(directory) / 'source'
            source.write_text('changed')
            with patch('work.run_response_cost_refinement.train_variant') as train:
                with self.assertRaisesRegex(ValueError, 'input hash changed'):
                    run({'output_dir': str(Path(directory) / 'new'), 'start_only_after_output_dir': str(previous),
                         'additional_stop_files': [], 'implementation_files': [], 'input_sha256': {str(source): '0' * 64}})
                train.assert_not_called()

    def test_both_long_endpoints_use_same_data_without_calibration_selection(self):
        module = 'work.run_response_cost_refinement'
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, previous = Path(directory) / 'new', Path(directory) / 'old'
            data = Path(directory) / 'data.npz'
            data.write_bytes(b'fixed-data')
            replay = SimpleNamespace(size=706, manifest={'catalog_fingerprint': 'test', 'response_equivalence_mode': 'post_commit_continuation_v1'})
            plan = {'output_dir': str(root), 'start_only_after_output_dir': str(previous),
                    'additional_stop_files': ['older/STOP'], 'additional_lock_dirs': ['older'],
                    'implementation_files': [], 'input_sha256': {}, 'data': str(data), 'expected_transitions': 706,
                    'diagnostic_replays': {'source_policy_in_training': 'source', 'quadratic_branch_in_training': 'branch'},
                    'control_model_dir': 'old/model', 'control_evaluation_dir': 'old/evaluation', 'training_spec': 'old/spec',
                    'evaluation_episode_base': 40000, 'hypothesis': 'duration and CQL', 'gradient_steps': 24000,
                    'variants': [{'name': 'long01', 'conservative_alpha': .1}, {'name': 'long001', 'conservative_alpha': .01}]}
            common = {'seed': 17, 'value_parameterization': 'finite_horizon_cost_v1', 'gradient_steps': 6000}
            stack.enter_context(patch(module + '._read_json', side_effect=[{'phase': 'ablation_complete'}, {'state': 'exited'}, {'common_training': common}]))
            stack.enter_context(patch(module + '.load_frozen_response_replay', return_value=replay))
            stack.enter_context(patch(module + '.validate_sequential_td_replay'))
            stack.enter_context(patch(module + '.validate_complete_episode'))
            stack.enter_context(patch(module + '._models', return_value=[model()]))
            stack.enter_context(patch(module + '.replay_fit', return_value={'terminal_mae': 999}))
            stack.enter_context(patch(module + '.path_rankings', return_value={}))
            stack.enter_context(patch(module + '.cost_head_checks', return_value={}))
            prefix = stack.enter_context(patch(module + '.verify_long_control_prefix', return_value={}))
            train = stack.enter_context(patch(module + '.train_variant', side_effect=lambda data, folder, spec, alpha, **kw: folder))
            evaluate = stack.enter_context(patch(module + '.evaluate_comparison'))
            run(plan)
            prefix.assert_called_once()
            self.assertEqual([c.args[3] for c in train.call_args_list], [.1, .01])
            for call in train.call_args_list:
                self.assertEqual(call.args[0], data)
                self.assertEqual(call.args[2]['common_training'], {**common, 'gradient_steps': 24000})
                self.assertIn('older/STOP', call.kwargs['stop_files'])
            evaluate.assert_called_once_with(['--config', str(root / 'evaluation_spec.json')])
            spec = json.loads((root / 'evaluation_spec.json').read_text())
            self.assertEqual([v['name'] for v in spec['variants']], ['long01', 'long001'])


if __name__ == '__main__':
    unittest.main()
