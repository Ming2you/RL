import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from work.run_response_continuation_cycle import _probe_worker, run, validate_probe_branches


MODULE = 'work.run_response_continuation_cycle'


class ProbeGateTest(unittest.TestCase):
    def test_equal_next_anchor_accepts_alias_pair(self):
        rows = [{'action': i, 'identity': {'a': 1}, 'next_anchor_identity': {'b': 2},
                 'next_anchor_response': [1, 2]} for i in (0, 1)]
        self.assertEqual(validate_probe_branches([[0, 1]], rows), 1)
        self.assertEqual(validate_probe_branches([[0], [1]], rows), 0)
        for field in ('identity', 'next_anchor_identity', 'next_anchor_response'):
            bad = copy.deepcopy(rows)
            bad[1][field] = {} if field != 'next_anchor_response' else [8, 8]
            with self.assertRaises((ValueError, AssertionError)):
                validate_probe_branches([[0, 1]], bad)

    def test_missing_future_evidence_rejected(self):
        rows = [{'action': i, 'identity': {'a': 1}} for i in (0, 1)]
        with self.assertRaisesRegex(ValueError, 'lack next-anchor'):
            validate_probe_branches([[0, 1]], rows)

    def test_worker_preserves_next_anchor_divergence_evidence(self):
        observation = np.array([1., 2.])
        features = np.array([[3.], [3.]])
        legacy = SimpleNamespace(control_step=np.array([17]), observation=observation[None],
                                 response_features=features[None], action_mask=np.array([[1, 1]]),
                                 manifest={'catalog': {}})
        current = SimpleNamespace(response_features=features, anchor_context=None,
                                  response_mask=SimpleNamespace(groups=[[0, 1]]),
                                  candidate_responses=[SimpleNamespace(valid=True, action_id=i,
                                      follower_memory_fingerprint='same') for i in (0, 1)])
        branches = [{'action': i, 'preview_commit_matches': True, 'identity': {'same': 1},
                     'next_anchor_identity': {'different': i}, 'next_anchor_response': [1.]} for i in (0, 1)]
        with patch(MODULE + '._configure_torch_threads'), \
             patch(MODULE + '.load_verified_snapshot', return_value=SimpleNamespace(control_step=17)), \
             patch(MODULE + '.restore_env_snapshot', return_value=(SimpleNamespace(), observation)), \
             patch(MODULE + '.load_frozen_response_replay', return_value=legacy), \
             patch(MODULE + '.StructuredActionCatalog.from_manifest'), \
             patch(MODULE + '.evaluate_executable_responses', return_value=current), \
             patch(MODULE + '._response_process_pool') as pool, \
             patch(MODULE + '._shutdown_response_process_pools') as shutdown:
            pool.return_value.map.return_value = branches
            result = _probe_worker({'environment': {}, 'legacy_replay': 'unused', 'workers': 4, 'stops': []})
        self.assertEqual(result['branches'], branches)
        self.assertEqual(result['preview_commit_mismatched_actions'], [])
        self.assertEqual(result['next_anchor_verified_alias_pairs'], 0)
        self.assertIn('diverge at the next anchor', result['next_anchor_validation_error'])
        shutdown.assert_called_once_with(wait=True)

    def test_persisted_future_mismatch_blocks_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root, previous = Path(directory) / 'new', Path(directory) / 'old'
            (root / 'probes').mkdir(parents=True)
            previous.mkdir()
            (previous / 'status.json').write_text(json.dumps({'phase': 'ablation_complete'}))
            (previous / 'process.json').write_text(json.dumps({'state': 'exited'}))
            environment = Path(directory) / 'environment.json'
            environment.write_text('{}')
            probes = [{'step': step, 'snapshot': 'unused', 'legacy_replay': 'unused'} for step in (17, 18)]
            for probe in probes:
                (root / 'probes' / f"step_{probe['step']:04d}.json").write_text(json.dumps({
                    'preview_commit_mismatched_actions': [], 'next_anchor_verified_alias_pairs': 1,
                    'next_anchor_validation_error': 'divergent future',
                }))
            plan = {'output_dir': str(root), 'start_only_after_output_dir': str(previous),
                    'additional_stop_files': [], 'implementation_files': [], 'input_sha256': {},
                    'environment_config': str(environment), 'response_workers_per_actor': 4, 'probes': probes}
            with patch(MODULE + '.ProcessPoolExecutor') as pool:
                run(plan)
                pool.assert_not_called()
            self.assertEqual(json.loads((root / 'status.json').read_text())['phase'],
                             'probe_next_anchor_mismatch_requires_diagnosis')


if __name__ == '__main__':
    unittest.main()
