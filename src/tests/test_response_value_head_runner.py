"""The value-head comparison must remain paired, stoppable and fail closed."""
from dataclasses import replace
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.tests.test_response_dqn_semantics import _replay
from work.run_response_reference_coverage import train_variant
from work.run_response_value_head_ablation import cost_head_checks, run


class ValueHeadRunnerTests(unittest.TestCase):
    def test_stop_before_inputs_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'STOP').touch()
            with patch('work.run_response_value_head_ablation.train_variant') as train:
                with self.assertRaises(InterruptedError):
                    run({'output_dir': str(root), 'start_only_after_output_dir': 'unused',
                         'additional_stop_files': []})
                train.assert_not_called()

    def test_changed_input_blocks_training(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / 'old'
            previous.mkdir()
            (previous / 'status.json').write_text(json.dumps({'phase': 'ablation_complete'}))
            (previous / 'process.json').write_text(json.dumps({'state': 'exited'}))
            source = Path(directory) / 'source'
            source.write_text('changed')
            with patch('work.run_response_value_head_ablation.train_variant') as train:
                with self.assertRaisesRegex(ValueError, 'input hash changed'):
                    run({'output_dir': str(Path(directory) / 'new'), 'start_only_after_output_dir': str(previous),
                         'additional_stop_files': [], 'implementation_files': [], 'input_sha256': {str(source): '0' * 64}})
                train.assert_not_called()

    def test_value_head_string_is_passed_to_training_cli(self):
        module = 'work.run_response_reference_coverage'
        with tempfile.TemporaryDirectory() as directory:
            with patch(module + '.subprocess.run') as launch, patch(module + '._read_json', return_value={}), \
                 patch(module + '.validate_training_recipe'):
                train_variant(Path(directory) / 'data.npz', Path(directory) / 'model', {
                    'common_training': {'group_resampling': False, 'value_parameterization': 'finite_horizon_cost_v1'},
                }, .1)
            command = launch.call_args.args[0]
            index = command.index('--value-parameterization')
            self.assertEqual(command[index + 1], 'finite_horizon_cost_v1')

    def test_structural_check_requires_nonpositive_and_zero_terminal_successor(self):
        replay = _replay()
        current = np.full((2, 2), -1.)
        following = np.array([[-1., -2.], [0., 0.]])
        def model(q, next_q):
            return SimpleNamespace(q_values=Mock(side_effect=[q, next_q]))
        result = cost_head_checks([model(current, following)], replay)
        self.assertTrue(result['terminal_successor_q_exactly_zero'])
        for q, next_q in ((-current, following), (current, following - 1),
                          (current, following + 1), (current * np.nan, following)):
            with self.subTest(q=q, next_q=next_q), self.assertRaises(ValueError):
                cost_head_checks([model(q, next_q)], replay)
        with self.assertRaisesRegex(ValueError, 'terminal rows'):
            cost_head_checks([model(current, following)], replace(replay, done=np.zeros(2)))

    def test_both_heads_share_data_and_recipe_and_evaluate_without_fit_gate(self):
        module = 'work.run_response_value_head_ablation'
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, previous = Path(directory) / 'new', Path(directory) / 'old'
            manifest = {'catalog_fingerprint': 'test', 'response_equivalence_mode': 'post_commit_continuation_v1'}
            base = SimpleNamespace(size=339, manifest=manifest)
            policy = SimpleNamespace(size=75, manifest=manifest, control_step=np.arange(75), reward=-np.ones(75))
            merged = SimpleNamespace(size=414, manifest=manifest)
            variants = [{'name': 'free', 'value_parameterization': 'free_q', 'conservative_alpha': .1},
                        {'name': 'cost', 'value_parameterization': 'finite_horizon_cost_v1', 'conservative_alpha': .1}]
            plan = {'output_dir': str(root), 'start_only_after_output_dir': str(previous),
                    'additional_stop_files': ['older/STOP'], 'additional_lock_dirs': ['older'],
                    'implementation_files': [], 'input_sha256': {}, 'base_replay': 'base.npz',
                    'policy_replay': 'policy.npz', 'control_evaluation_dir': str(previous / 'evaluation'),
                    'expected_transitions': 414, 'training_spec': 'training.json',
                    'evaluation_episode_base': 37000, 'hypothesis': 'head only', 'variants': variants}
            summary = {'scope': 'ungated_full_run', 'epsilon': 0, 'first_action': None,
                       'lcb_guard': False, 'response_preview': True, 'start_control_step': 0,
                       'end_control_step': 74, 'transitions': 75, 'replay': 'policy.npz',
                       'prefix_ttt': 10, 'total_ttt': 85}
            common = {'gamma': 1, 'mask_constant_features': True, 'seed': 17, 'gradient_steps': 6000}
            stack.enter_context(patch(module + '._read_json', side_effect=[
                {'phase': 'ablation_complete'}, {'state': 'exited'}, summary,
                {'common_training': common, 'data': 'old.npz'}]))
            stack.enter_context(patch(module + '.load_frozen_response_replay', side_effect=[base, policy]))
            stack.enter_context(patch(module + '.merge_frozen_response_replays', return_value=merged))
            stack.enter_context(patch(module + '.validate_sequential_td_replay'))
            stack.enter_context(patch(module + '.validate_complete_episode'))
            saved = stack.enter_context(patch(module + '.persist_merged_replay', return_value='digest'))
            train = stack.enter_context(patch(module + '.train_variant', side_effect=lambda data, path, spec, alpha, **kw: path))
            stack.enter_context(patch(module + '._models', return_value=[]))
            # Intentionally poor calibration must not suppress either full evaluation.
            stack.enter_context(patch(module + '.replay_fit', return_value={'terminal_behavior_mae_scaled': 999}))
            stack.enter_context(patch(module + '.cost_head_checks', return_value={'valid': True}))
            evaluate = stack.enter_context(patch(module + '.evaluate_comparison'))
            run(plan)
            saved.assert_called_once_with(merged, root / 'training_replay.npz')
            self.assertEqual(train.call_count, 2)
            for call, variant in zip(train.call_args_list, variants):
                self.assertEqual(call.args[0], root / 'training_replay.npz')
                self.assertEqual(call.args[2]['common_training'], {**common, 'value_parameterization': variant['value_parameterization']})
                self.assertEqual(call.kwargs['stop_files'], [str(root / 'STOP'), str(previous / 'STOP'), 'older/STOP'])
            evaluate.assert_called_once_with(['--config', str(root / 'evaluation_spec.json')])
            spec = json.loads((root / 'evaluation_spec.json').read_text())
            self.assertEqual([v['name'] for v in spec['variants']], ['free', 'cost'])


if __name__ == '__main__':
    unittest.main()
