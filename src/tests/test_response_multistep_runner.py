"""Freeze data and evaluation rules while comparing backup horizons."""
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.tests.test_response_nonlinear_coverage import episode
from work.run_response_cql_ablation import variant_training_spec, validate_checkpoint_value_heads
from work.run_response_multistep_ablation import greedy_path_coverage, run, verify_policy_source


class MultistepRunnerTests(unittest.TestCase):
    def test_backup_override_keeps_other_variant_configuration_and_legacy_default(self):
        spec = {'data': 'old', 'common_training': {'gamma': 1, 'value_parameterization': 'free_q'}}
        variant = variant_training_spec(spec, {'backup_horizon': 5, 'value_parameterization': 'finite_horizon_cost_v1'})
        self.assertEqual(variant['common_training'], {'gamma': 1, 'value_parameterization': 'finite_horizon_cost_v1', 'backup_horizon': 5})
        self.assertNotIn('backup_horizon', spec['common_training'])
        validate_checkpoint_value_heads([SimpleNamespace(config=SimpleNamespace())], spec)
        with self.assertRaisesRegex(ValueError, 'backup horizon'):
            validate_checkpoint_value_heads([SimpleNamespace(config=SimpleNamespace(value_parameterization='finite_horizon_cost_v1'))], variant)
        for bad in (0, -1, True, 1.5, '5'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                variant_training_spec(spec, {'backup_horizon': bad})

    def test_source_is_complete_unforced_and_reconciles_reward(self):
        r = episode()
        summary = {'scope': 'ungated_full_run', 'epsilon': 0, 'first_action': None, 'lcb_guard': False,
                   'response_preview': True, 'replay': 'source.npz', 'prefix_ttt': 100., 'total_ttt': 175.}
        verify_policy_source(r, summary, 'source.npz')
        for field, value in (('scope', 'fixed_prefix'), ('first_action', 0), ('epsilon', .1),
                             ('lcb_guard', True), ('response_preview', False), ('replay', 'wrong')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_policy_source(r, {**summary, field: value}, 'source.npz')
        with self.assertRaises(AssertionError):
            verify_policy_source(r, {**summary, 'total_ttt': 176.}, 'source.npz')
        with self.assertRaises(ValueError):
            verify_policy_source(replace(r, control_step=r.control_step + 1), summary, 'source.npz')

    def test_endpoint_lengths_cut_at_member_greedy_disagreement(self):
        r = episode()
        q = np.zeros((75, 10))
        q[2, 1] = 1.
        model = SimpleNamespace(seed=7, q_values=Mock(return_value=q),
                                action_support_counts=np.ones(10, int), config=SimpleNamespace(min_action_support=1))
        report = greedy_path_coverage([model], r)
        self.assertEqual(report['verified_links'], 74)
        self.assertEqual(sum(report['members'][0]['length_counts']), 75)
        self.assertGreater(report['members'][0]['length_counts'][1], 1)
        self.assertIn('not actual sampled', report['interpretation'])

    def test_stop_prevents_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'STOP').touch()
            with patch('work.run_response_multistep_ablation.train_variant') as train:
                with self.assertRaises(InterruptedError):
                    run({'output_dir': str(root), 'start_only_after_output_dir': 'unused', 'additional_stop_files': []})
                train.assert_not_called()

    def test_hash_change_blocks_merge_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / 'old'
            previous.mkdir()
            (previous / 'status.json').write_text(json.dumps({'phase': 'ablation_complete'}))
            (previous / 'process.json').write_text(json.dumps({'state': 'exited'}))
            source = Path(directory) / 'source'
            source.write_text('changed')
            with patch('work.run_response_multistep_ablation.train_variant') as train:
                with self.assertRaisesRegex(ValueError, 'input hash changed'):
                    run({'output_dir': str(Path(directory) / 'new'), 'start_only_after_output_dir': str(previous),
                         'additional_stop_files': [], 'implementation_files': [], 'input_sha256': {str(source): '0' * 64}})
                train.assert_not_called()

    def test_both_horizons_share_data_and_are_evaluated_even_with_poor_fit(self):
        module = 'work.run_response_multistep_ablation'
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, previous = Path(directory) / 'new', Path(directory) / 'old'
            manifest = {'catalog_fingerprint': 'test', 'response_equivalence_mode': 'post_commit_continuation_v1'}
            base = SimpleNamespace(size=706, manifest=manifest)
            policy = SimpleNamespace(size=75, manifest=manifest)
            merged = SimpleNamespace(size=856, manifest=manifest, done=np.r_[np.ones(12), np.zeros(844)],
                                     action_support_counts=lambda: np.ones(10, int))
            plan = {'output_dir': str(root), 'start_only_after_output_dir': str(previous), 'additional_stop_files': ['older/STOP'],
                    'additional_lock_dirs': ['older'], 'implementation_files': [], 'input_sha256': {}, 'base_replay': 'base',
                    'policy_evaluation_dirs': ['high/eval', 'low/eval'], 'expected_transitions': 856, 'training_spec': 'old/spec',
                    'control_evaluation_dir': 'old/eval', 'source_policy_replay': 'source', 'quadratic_branch_replay': 'branch',
                    'evaluation_episode_base': 41000, 'hypothesis': 'backup depth',
                    'variants': [{'name': 'one', 'backup_horizon': 1, 'conservative_alpha': .01},
                                 {'name': 'five', 'backup_horizon': 5, 'conservative_alpha': .01}]}
            common = {'seed': 17, 'value_parameterization': 'finite_horizon_cost_v1', 'gradient_steps': 24000, 'group_resampling': False}
            stack.enter_context(patch(module + '._read_json', side_effect=lambda p: {'phase': 'ablation_complete', 'state': 'exited', 'common_training': common}))
            stack.enter_context(patch(module + '.load_frozen_response_replay', side_effect=lambda p: base if str(p) == 'base' else policy))
            stack.enter_context(patch(module + '.validate_sequential_td_replay'))
            stack.enter_context(patch(module + '.validate_complete_episode'))
            verify = stack.enter_context(patch(module + '.verify_policy_source'))
            merge = stack.enter_context(patch(module + '.merge_frozen_response_replays', return_value=merged))
            stack.enter_context(patch(module + '.build_sequential_links', return_value=np.arange(856)))
            stack.enter_context(patch(module + '.persist_merged_replay', return_value='digest'))
            stack.enter_context(patch(module + '._models', return_value=[SimpleNamespace(seed=17, training_losses=[999.])]))
            stack.enter_context(patch(module + '.replay_fit', return_value={'error': 999.}))
            stack.enter_context(patch(module + '.path_rankings', return_value={}))
            stack.enter_context(patch(module + '.greedy_path_coverage', return_value={}))
            stack.enter_context(patch(module + '.cost_head_checks', return_value={}))
            train = stack.enter_context(patch(module + '.train_variant', side_effect=lambda data, folder, spec, alpha, **kw: folder))
            evaluate = stack.enter_context(patch(module + '.evaluate_comparison'))
            run(plan)
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(merge.call_args.args[0], [base, policy, policy])
            self.assertEqual([c.args[2]['common_training']['backup_horizon'] for c in train.call_args_list], [1, 5])
            for call in train.call_args_list:
                self.assertEqual(call.args[0], root / 'training_replay.npz')
                self.assertEqual(call.args[3], .01)
                self.assertIn('older/STOP', call.kwargs['stop_files'])
                self.assertEqual(call.args[2]['common_training']['gradient_steps'], 24000)
            evaluate.assert_called_once_with(['--config', str(root / 'evaluation_spec.json')])
            spec = json.loads((root / 'evaluation_spec.json').read_text())
            self.assertEqual([v['backup_horizon'] for v in spec['variants']], [1, 5])
            self.assertEqual(spec['control_response_workers'], 4)


if __name__ == '__main__':
    unittest.main()
