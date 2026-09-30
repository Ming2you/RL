"""Export completed SDMPC research data without running or changing experiments."""
import argparse
import json
from pathlib import Path

from work import research_snapshot as archive

FAMILIES = (
    'sdmpc_rl_budget_20260929', 'sdmpc_rl_carry_20260929',
    'sdmpc_rl_multi_20260929', 'sdmpc_rl_recovery_20260929',
    'sdmpc_rl_balanced_goal_20260930',
)


def select(repo):
    files = {p.relative_to(repo).as_posix(): p for family in FAMILIES
             for p in (repo / 'results' / family).rglob('*') if p.is_file()}
    selected = {name for name in files if 'checkpoints' not in Path(name).parts
                and not name.endswith('.lock')}
    # Completion/data manifests and latest pointers retain the exact checkpoints
    # needed by current authenticators, while redundant interval history stays local.
    for name, path in files.items():
        if path.name not in ('settings.json', 'completion.json', 'latest.json'):
            continue
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if path.name == 'latest.json' and isinstance(value.get('path'), str):
            target = (path.parent / value['path']).resolve()
            selected.add(target.relative_to(repo).as_posix())
        for relative in value.get('outputs_sha256', {}):
            target = (path.parent / relative).resolve()
            selected.add(target.relative_to(repo).as_posix())
        settings = value.get('settings', value)
        for relative in settings.get('data', {}).get('files', {}):
            if relative.startswith('results/'):
                selected.add(relative)
    for name in selected:
        path = archive.safe_target(repo, name)
        if name not in files or not path.is_file():
            raise ValueError('missing or out-of-scope dependency: ' + name)
    omitted = [{'path': name, 'bytes': path.stat().st_size,
                'sha256': archive.sha256(path),
                'reason': 'local-only redundant checkpoint or historical lock'}
               for name, path in sorted(files.items()) if name not in selected]
    return sorted(selected), omitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('build', 'verify', 'verify-local', 'restore'))
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--bundle', type=Path,
                        default=Path('artifacts/sdmpc_research_handoff_20260930'))
    args = parser.parse_args()
    repo, bundle = args.repo.resolve(), args.bundle.resolve()
    if args.mode == 'build':
        selected, omitted = select(repo)
        archive.build(repo, bundle, selected=selected, status='HANDOFF_NO_EXPERIMENT_LAUNCH',
                      note='Completed SDMPC data, final models, traces, replay and referenced '
                           'checkpoints. Redundant intermediate checkpoint bytes and lock '
                           'files remain local and are catalogued separately. Export does '
                           'not pause/resume the goal or authorize any experiment.')
        archive.write_json(bundle / 'local_only_inventory.json', omitted)
    else:
        archive.inspect_or_restore(repo, bundle, args.mode)


if __name__ == '__main__':
    main()
