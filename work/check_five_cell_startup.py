"""Exercise real preview/commit and checkpoint resume in all five scenes.

Artifacts are separate from the research run. This performs two real controlled
intervals per scenario, never relabels truncation as terminal, and trains no model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from work.run_five_cell_shared import ROOT, read_json, prepare, job, write_json
from work.five_cell_worker import collect_five_cell_episode


def probe(payload):
    progress = Path(payload['directory']) / 'progress.json'
    before = read_json(progress)['transitions'] if progress.exists() else 0
    try:
        collect_five_cell_episode(payload, max_new_transitions=1)
    except InterruptedError:
        after = read_json(progress)
        if after['transitions'] != before + 1 or after['terminal']:
            raise ValueError('startup probe must persist exactly one nonterminal transition')
        return {'scenario': payload['scenario'], 'transitions': after['transitions'],
                'terminal': after['terminal'], 'total_ttt_so_far': after['total_ttt_so_far']}
    raise ValueError('startup probe unexpectedly completed an episode')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('work/five_cell_shared_v1.json'))
    parser.add_argument('--name', default='startup_smoke_v1')
    args = parser.parse_args()
    os.chdir(ROOT)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    plan = read_json(args.config)
    plan['output_dir'] = 'results/five_cell_shared_checks/' + args.name
    root, contracts, catalog, sources = prepare(plan, args.config)
    payloads = [job(plan, root, contracts, catalog, sources, scenario=s, purpose='baseline', index=0)
                for s in plan['scenarios']]
    started = time.perf_counter()
    rounds = []
    for index in range(2):
        values = []
        with ProcessPoolExecutor(max_workers=plan['actors']) as pool:
            futures = [pool.submit(probe, payload) for payload in payloads]
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps({'phase': 'checkpoint_resume_smoke', **result}), flush=True)
                values.append(result)
        rounds.append(values)
    result = {'passed': True, 'scenarios': plan['scenarios'], 'rounds': rounds,
              'wall_seconds': time.perf_counter() - started, 'models_trained': 0}
    write_json(root / 'smoke_result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
