"""Gate corrected-equivalence collection on exact branch/next-anchor probes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from rl_leader.response_ddqn_recovery import restore_env_snapshot
from rl_leader.response_dqn_catalog import StructuredActionCatalog
from rl_leader.response_dqn_collect import (
    _encoded_continuation, _response_process_pool, _shutdown_response_process_pools,
    evaluate_executable_responses,
)
from rl_leader.response_dqn_data import load_frozen_response_replay, merge_frozen_response_replays
from rl_leader.response_dqn_mask import CONTINUATION_EQUIVALENCE
from rl_leader.run_sequential_response_ddqn import (
    _atomic_json, _collect_actor_worker, load_verified_snapshot, validate_complete_episode,
)
from rl_leader.train_response_dqn import _configure_torch_threads
from work.run_response_cql_ablation import _pinned_json, _read_json, main as evaluate_comparison
from work.run_response_reference_coverage import persist_merged_replay, train_variant, verify_reference


def _check_stop(paths):
    if any(Path(path).exists() for path in paths):
        raise InterruptedError('STOP requested')


def _probe_branch(payload):
    source, context, catalog, action, expected_identity, check_future, stops = payload
    _check_stop(stops)
    env, context = copy.deepcopy((source, context))
    if action == 0:
        _, reward, done, info, _ = env.step_prepared_optimizer_anchor(context)
    else:
        _, reward, done, info = env.step_anchored_candidate(catalog.residual(action), context)
    response = env.response_vector(env.previous).tolist()
    identity = _encoded_continuation(env, reward, done, info.get('validity_gate_pass', False))
    result = {'action': action, 'identity': json.loads(identity), 'reward': float(reward),
              'response': response, 'preview_commit_matches': identity == expected_identity}
    if identity != expected_identity:
        result['expected_identity'] = json.loads(expected_identity)
        return result
    if check_future and not done:
        _check_stop(stops)
        following = env.prepare_pstack_anchor_context()
        _, reward, done, info, _ = env.step_prepared_optimizer_anchor(following)
        result['next_anchor_identity'] = json.loads(_encoded_continuation(
            env, reward, done, info.get('validity_gate_pass', False),
        ))
        result['next_anchor_response'] = env.response_vector(env.previous).tolist()
    return result


def validate_probe_branches(groups, branches):
    by_id = {item['action']: item for item in branches}
    verified = 0
    for group in groups:
        if len(group) < 2:
            continue
        first = by_id[group[0]]
        for action in group[1:]:
            other = by_id[action]
            if first['identity'] != other['identity']:
                raise ValueError('merged branches have different post-commit identities')
            if 'next_anchor_identity' not in first or 'next_anchor_identity' not in other:
                raise ValueError('merged branches lack next-anchor evidence')
            if first['next_anchor_identity'] != other['next_anchor_identity']:
                raise ValueError('merged branches diverge at the next anchor')
            np.testing.assert_array_equal(first['next_anchor_response'], other['next_anchor_response'])
            verified += 1
    return verified


def _probe_worker(payload):
    _configure_torch_threads(1)
    try:
        _check_stop(payload['stops'])
        snapshot = load_verified_snapshot(payload['environment'])
        env, observation = restore_env_snapshot(snapshot)
        env.response_equivalence_mode = CONTINUATION_EQUIVALENCE
        legacy = load_frozen_response_replay(payload['legacy_replay'])
        index = np.flatnonzero(legacy.control_step == snapshot.control_step)
        if len(index) != 1:
            raise ValueError('legacy parity replay must contain exactly one matched step')
        index = int(index[0])
        np.testing.assert_array_equal(observation, legacy.observation[index])
        catalog = StructuredActionCatalog.from_manifest(legacy.manifest['catalog'])
        current = evaluate_executable_responses(
            env, observation, catalog, workers=payload['workers'], backend='process',
        )
        np.testing.assert_array_equal(current.response_features, legacy.response_features[index])
        if not all(item.valid for item in current.candidate_responses):
            raise ValueError('a candidate failed validity during the matched-state probe')
        groups = current.response_mask.groups
        alias_actions = {action for group in groups if len(group) > 1 for action in group}
        tasks = [(env, current.anchor_context, catalog, item.action_id, item.follower_memory_fingerprint,
                  item.action_id in alias_actions, payload['stops']) for item in current.candidate_responses]
        branches = list(_response_process_pool(payload['workers']).map(_probe_branch, tasks))
        mismatches = [item['action'] for item in branches if not item['preview_commit_matches']]
        verified, future_error = 0, None
        if not mismatches:
            try:
                verified = validate_probe_branches(groups, branches)
            except (ValueError, AssertionError) as exc:
                # Keep both divergent branches available for diagnosis.
                future_error = f'{type(exc).__name__}: {exc}'
        return {
            'step': snapshot.control_step, 'legacy_groups': int(legacy.action_mask[index].sum()),
            'continuation_groups': [list(group) for group in groups],
            'preview_commit_verified_actions': len(branches), 'next_anchor_verified_alias_pairs': verified,
            'physical_preview_matches_legacy': True, 'branches': branches,
            'preview_commit_mismatched_actions': mismatches,
            'next_anchor_validation_error': future_error,
        }
    finally:
        _shutdown_response_process_pools(wait=True)


def run(plan, *, probe_only=False):
    root = Path(plan['output_dir'])
    previous = Path(plan['start_only_after_output_dir'])
    stops = [str(root / 'STOP'), str(previous / 'STOP'), *plan['additional_stop_files']]
    _check_stop(stops)
    if _read_json(previous / 'status.json')['phase'] != 'ablation_complete' or _read_json(previous / 'process.json')['state'] != 'exited':
        raise ValueError('previous experiment has not completed')
    _pinned_json(root / 'cycle_plan.json', plan)
    sources = [Path(path) for path in plan['implementation_files']]
    _pinned_json(root / 'implementation_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for path, digest in plan['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest().lower() != digest.lower():
            raise ValueError(f'input hash changed: {path}')
    environment = _read_json(plan['environment_config'])
    workers = plan['response_workers_per_actor']
    if workers != 4 or len(plan['probes']) != 2:
        raise ValueError('this cycle requires two actors with four preview workers each')
    _atomic_json(root / 'status.json', {'phase': 'probing_continuation_equivalence'})
    probe_paths = [root / 'probes' / f"step_{item['step']:04d}.json" for item in plan['probes']]
    tasks = [{
        'environment': {**environment, 'snapshot': item['snapshot']},
        'legacy_replay': item['legacy_replay'], 'workers': workers, 'stops': stops,
    } for item in plan['probes']]
    pending = [(task, path) for task, path in zip(tasks, probe_paths) if not path.exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_probe_worker, task): path for task, path in pending}
            for future in as_completed(futures):
                path, result = futures[future], future.result()
                _pinned_json(path, result)
                print(json.dumps({'event': 'probe_verified', 'step': result['step'],
                                  'groups': result['continuation_groups'],
                                  'alias_pairs': result['next_anchor_verified_alias_pairs']}), flush=True)
    probes = [_read_json(path) for path in probe_paths]
    if any(item.get('preview_commit_mismatched_actions') for item in probes):
        _atomic_json(root / 'status.json', {'phase': 'probe_commit_mismatch_requires_diagnosis'})
        return
    if any(item.get('next_anchor_validation_error') for item in probes):
        _atomic_json(root / 'status.json', {'phase': 'probe_next_anchor_mismatch_requires_diagnosis'})
        return
    if sum(item['next_anchor_verified_alias_pairs'] for item in probes) < 1:
        _atomic_json(root / 'status.json', {'phase': 'probe_no_merge_requires_diagnosis'})
        return
    if probe_only:
        _atomic_json(root / 'status.json', {'phase': 'probe_complete'})
        return
    _check_stop(stops)
    catalog = load_frozen_response_replay(plan['probes'][0]['legacy_replay']).manifest['catalog']
    payloads = [{
        'config': environment, 'catalog': catalog, 'directory': str(root / 'collection' / item['name']),
        'seed': plan['collection_seed'] + index, 'episode': plan['collection_episode_base'] + index,
        'response_workers': workers, 'checkpoints': [], 'epsilon': item['epsilon'],
        'first_action': item['first_action'], 'full_run': item['full_run'],
        'response_equivalence_mode': CONTINUATION_EQUIVALENCE,
        'stop_file': stops[0], 'additional_stop_files': stops[1:],
    } for index, item in enumerate(plan['collection'])]
    _atomic_json(root / 'status.json', {'phase': 'collecting_corrected_equivalence', 'episodes': len(payloads)})
    # Verify both references before spending the exploratory collection budget.
    for reference_phase in (True, False):
        batch = [(payload, item) for payload, item in zip(payloads, plan['collection'])
                 if ('reference' in item) == reference_phase]
        _check_stop(stops)
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_collect_actor_worker, payload): (payload, item) for payload, item in batch}
            for future in as_completed(futures):
                payload, item = futures[future]
                result = future.result()
                replay = load_frozen_response_replay(Path(payload['directory']) / 'replay.npz')
                validate_complete_episode(replay)
                if 'reference' in item:
                    verify_reference(result, replay, item['reference'], 1e-4)
                print(json.dumps({'event': 'corrected_episode_complete', 'name': item['name'], **result}), flush=True)
    _check_stop(stops)
    merged = merge_frozen_response_replays([
        load_frozen_response_replay(Path(payload['directory']) / 'replay.npz') for payload in payloads
    ], source=__name__)
    data = root / 'training_replay.npz'
    digest = persist_merged_replay(merged, data)
    spec = {**_read_json(plan['training_spec']),
            'response_equivalence_mode': CONTINUATION_EQUIVALENCE, 'data': str(data), 'data_sha256': digest,
            'output_dir': str(root), 'start_only_after_output_dir': str(previous),
            'required_previous_phase': 'ablation_complete', 'additional_stop_files': stops[2:],
            'control_evaluation_dir': plan['control_evaluation_dir'], 'control_response_workers': 8,
            'evaluation_episode_base': plan['evaluation_episode_base'],
            'hypothesis': plan['hypothesis']}
    _atomic_json(root / 'status.json', {'phase': 'training', 'transitions': merged.size})
    model_dir = train_variant(data, root / 'corrected/model', spec, .1)
    _check_stop(stops)
    spec['variants'] = [{'name': 'corrected', 'conservative_alpha': .1, 'model_dir': str(model_dir)}]
    # The old terminal-fit gate is not a gate for a different, newly constructed dataset.
    spec.pop('pre_evaluation_fit_gate', None)
    spec_path = root / 'evaluation_spec.json'
    _pinned_json(spec_path, spec)
    evaluate_comparison(['--config', str(spec_path)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    _configure_torch_threads(1)
    plan = _read_json(args.config)
    try:
        run(plan, probe_only=args.probe_only)
    except InterruptedError:
        _atomic_json(Path(plan['output_dir']) / 'status.json', {'phase': 'paused'})
    except Exception as exc:
        _atomic_json(Path(plan['output_dir']) / 'status.json', {'phase': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
