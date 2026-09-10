"""Targeted branches must preserve source states, true terminals and scope."""
from dataclasses import replace
from concurrent.futures import Future
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rl_leader.response_dqn_data import FrozenResponseReplay, make_replay_manifest
from work.run_response_nonlinear_coverage import branch_opportunity, run, verify_branch


def episode():
    n, actions = 75, 10
    observation = np.stack([np.arange(n + 1), np.ones(n + 1)], axis=1).astype(np.float32)
    response = np.zeros((n + 1, actions, 1), np.float32)
    response[:, 6] = 2
    response[:, 9] = 3
    manifest = make_replay_manifest(transition_count=n, action_count=actions, catalog_fingerprint='test',
                                    observation_schema={'dimension': 2}, response_contract='test', scenario='test', source='test')
    manifest.update(reward_semantics='interval_negative_ttt', done_semantics='environment_terminal')
    return FrozenResponseReplay(
        observation=observation[:-1], next_observation=observation[1:], action_id=np.zeros(n, np.int64),
        reward=-np.ones(n, np.float32), done=np.r_[np.zeros(n - 1), 1].astype(np.float32),
        option_steps=np.ones(n, np.int64), action_mask=np.ones((n, actions), bool), next_action_mask=np.ones((n, actions), bool),
        response_features=response[:-1], next_response_features=response[1:], event_group=np.repeat('test', n),
        episode=np.zeros(n, np.int64), control_step=np.arange(n), manifest=manifest,
    ).validate()


class NonlinearCoverageTests(unittest.TestCase):
    def test_opportunity_requires_distinct_unobserved_executable_response(self):
        policy = episode()
        branch = {'step': 7, 'first_action': 9}
        self.assertEqual(branch_opportunity(policy, policy, branch)['global_action_support'], 0)
        actions = policy.action_id.copy()
        actions[7] = 9
        with self.assertRaisesRegex(ValueError, 'already observed'):
            branch_opportunity(replace(policy, action_id=actions), policy, branch)
        features = policy.response_features.copy()
        features[7, 9] = features[7, 0]
        with self.assertRaisesRegex(ValueError, 'equals a linear'):
            branch_opportunity(policy, replace(policy, response_features=features), branch)
        mask = policy.action_mask.copy()
        mask[7, 9] = False
        with self.assertRaisesRegex(ValueError, 'executable representative'):
            branch_opportunity(policy, replace(policy, action_mask=mask), branch)

    def branch(self):
        policy = episode()
        step, action = 7, 9
        arrays = {name: getattr(policy, name)[step:].copy() for name in policy.__dataclass_fields__ if name != 'manifest'}
        arrays['action_id'][0] = action
        tail = replace(policy, **arrays, manifest={**policy.manifest, 'transition_count': 75 - step})
        summary = {'scope': 'fixed_prefix_continuation', 'epsilon': 0, 'first_action': action,
                   'lcb_guard': False, 'response_preview': True, 'start_control_step': step,
                   'end_control_step': 74, 'transitions': 75 - step, 'prefix_ttt': 10 + step,
                   'total_ttt': 85, 'terminal_inventory': 3, 'wall_seconds': 1}
        return policy, tail, summary, {'step': step, 'first_action': action}

    def test_branch_is_sequential_not_terminalized_or_full_policy_evidence(self):
        policy, tail, summary, branch = self.branch()
        result = verify_branch(summary, tail, policy, {'prefix_ttt': 10, 'total_ttt': 85}, branch)
        self.assertEqual(result['delta_ttt_vs_frozen_cost_policy'], 0)
        self.assertEqual(result['transitions'], 68)
        self.assertIn('Not ungated', result['interpretation'])
        with self.assertRaises(ValueError):
            verify_branch(summary, replace(tail, done=np.ones(tail.size)), policy, {}, branch)

    def test_branch_rejects_changed_scope_prefix_or_initial_response(self):
        policy, tail, summary, branch = self.branch()
        control = {'prefix_ttt': 10, 'total_ttt': 85}
        for key, value in {'scope': 'ungated_full_run', 'epsilon': .1, 'first_action': 0,
                           'start_control_step': 0, 'prefix_ttt': 30, 'total_ttt': 50}.items():
            with self.subTest(key=key), self.assertRaises((ValueError, AssertionError)):
                verify_branch({**summary, key: value}, tail, policy, control, branch)
        for field in ('observation', 'response_features'):
            changed = getattr(tail, field).copy()
            changed[0] += 1
            with self.subTest(field=field), self.assertRaises(AssertionError):
                verify_branch(summary, replace(tail, **{field: changed}), policy, control, branch)
        changed = tail.action_id.copy()
        changed[0] = 0
        with self.assertRaisesRegex(ValueError, 'aliased'):
            verify_branch(summary, replace(tail, action_id=changed), policy, control, branch)

    def test_stop_prevents_inputs_collection_and_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'STOP').touch()
            with patch('work.run_response_nonlinear_coverage.train_variant') as train:
                with self.assertRaises(InterruptedError):
                    run({'output_dir': str(root), 'start_only_after_output_dir': 'unused', 'additional_stop_files': []})
                train.assert_not_called()

    def test_changed_input_blocks_prefix_and_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / 'old'
            previous.mkdir()
            (previous / 'status.json').write_text(json.dumps({'phase': 'ablation_complete'}))
            (previous / 'process.json').write_text(json.dumps({'state': 'exited'}))
            source = Path(directory) / 'source'
            source.write_text('changed')
            with patch('work.run_response_nonlinear_coverage.capture_matched_prefix') as capture:
                with self.assertRaisesRegex(ValueError, 'input hash changed'):
                    run({'output_dir': str(Path(directory) / 'new'), 'start_only_after_output_dir': str(previous),
                         'additional_stop_files': [], 'implementation_files': [], 'input_sha256': {str(source): '0' * 64}})
                capture.assert_not_called()

    def test_paired_refit_uses_only_targeted_arm_additional_rows(self):
        module = 'work.run_response_nonlinear_coverage'
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root, previous = Path(directory) / 'new', Path(directory) / 'old'
            policy = episode()
            policy.manifest['catalog'] = {}
            policy.manifest['response_equivalence_mode'] = 'post_commit_continuation_v1'
            refit, targeted = SimpleNamespace(size=564), SimpleNamespace(size=706)
            branches = [{'name': 'quadratic', 'step': 1, 'first_action': 6}, {'name': 'combo', 'step': 7, 'first_action': 9}]
            plan = {'output_dir': str(root), 'start_only_after_output_dir': str(previous),
                    'additional_stop_files': ['older/STOP'], 'additional_lock_dirs': ['older'],
                    'implementation_files': [], 'input_sha256': {}, 'refit_replays': ['base', 'free', 'policy'],
                    'policy_replay': 'policy', 'control_evaluation_dir': str(previous / 'evaluation'),
                    'control_model_dir': 'old/model', 'expected_refit_transitions': 564, 'expected_targeted_transitions': 706,
                    'response_workers_per_actor': 4, 'branches': branches, 'environment_config': 'environment',
                    'collection_seed': 123, 'collection_episode_base': 38000, 'evaluation_episode_base': 39000,
                    'training_spec': 'training', 'hypothesis': 'coverage only'}
            summary = {'scope': 'ungated_full_run', 'epsilon': 0, 'first_action': None, 'lcb_guard': False,
                       'replay': 'policy', 'prefix_ttt': 10, 'total_ttt': 85, 'trace_segments': ['trace']}
            common = {'seed': 17, 'gamma': 1, 'gradient_steps': 6000, 'workers': 3}
            stack.enter_context(patch(module + '._read_json', side_effect=[
                {'phase': 'ablation_complete'}, {'state': 'exited'}, summary, {},
                {'checkpoints': ['old0', 'old1', 'old2']}, {'common_training': common}]))
            stack.enter_context(patch(module + '.load_frozen_response_replay', return_value=policy))
            merges = stack.enter_context(patch(module + '.merge_frozen_response_replays', side_effect=[refit, targeted]))
            stack.enter_context(patch(module + '.validate_sequential_td_replay'))
            stack.enter_context(patch(module + '.validate_complete_episode'))
            stack.enter_context(patch(module + '.branch_opportunity', return_value={}))
            stack.enter_context(patch(module + '.StructuredActionCatalog.from_manifest'))
            stack.enter_context(patch(module + '.capture_matched_prefix', return_value={1: Path('one.pkl'), 7: Path('seven.pkl')}))
            pool = stack.enter_context(patch(module + '.ProcessPoolExecutor'))
            def submit(worker, payload):
                future = Future()
                future.set_result({})
                return future
            pool.return_value.__enter__.return_value.submit.side_effect = submit
            stack.enter_context(patch(module + '.verify_branch', return_value={}))
            saved = stack.enter_context(patch(module + '.persist_merged_replay', side_effect=['refit_hash', 'targeted_hash']))
            train = stack.enter_context(patch(module + '.train_variant', side_effect=lambda data, folder, spec, alpha, **kw: folder))
            stack.enter_context(patch(module + '._models', return_value=[]))
            stack.enter_context(patch(module + '.replay_fit', return_value={'terminal_behavior_mae_scaled': 999}))
            stack.enter_context(patch(module + '.cost_head_checks', return_value={}))
            evaluate = stack.enter_context(patch(module + '.evaluate_comparison'))
            run(plan)
            pool.assert_called_once_with(max_workers=2)
            for call in pool.return_value.__enter__.return_value.submit.call_args_list:
                payload = call.args[1]
                self.assertEqual(payload['response_workers'], 4)
                self.assertEqual(payload['checkpoints'], ['old0', 'old1', 'old2'])
                self.assertFalse(payload['full_run'])
            self.assertEqual(merges.call_args_list[-1].args[0], [refit, policy, policy])
            self.assertEqual([call.args[0].size for call in saved.call_args_list], [564, 706])
            for call in train.call_args_list:
                self.assertEqual(call.args[2]['common_training'], {**common, 'value_parameterization': 'finite_horizon_cost_v1'})
                self.assertIn('older/STOP', call.kwargs['stop_files'])
            evaluate.assert_called_once_with(['--config', str(root / 'evaluation_spec.json')])
            spec = json.loads((root / 'evaluation_spec.json').read_text())
            self.assertEqual([v['name'] for v in spec['variants']], ['refit_only', 'targeted'])
            self.assertEqual([v['data_sha256'] for v in spec['variants']], ['refit_hash', 'targeted_hash'])


if __name__ == '__main__':
    unittest.main()
