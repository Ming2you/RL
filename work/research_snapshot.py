"""Build, verify or restore a byte-preserving research archive; never run RL."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


ROOTS = ('data', 'results', 'models', 'checkpoints')
CHUNK = 1024 * 1024
PART_BYTES = 40 * CHUNK


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write('\n')


def safe_target(repo, name):
    part = PurePosixPath(name)
    if part.is_absolute() or not part.parts or part.parts[0] not in ROOTS:
        raise ValueError(f'archive member outside artifact roots: {name}')
    if '\\' in name or ':' in name or any(p in ('', '.', '..') for p in part.parts):
        raise ValueError(f'unsafe archive member: {name}')
    target = repo.joinpath(*part.parts)
    if not target.resolve().is_relative_to(repo.resolve()):
        raise ValueError(f'archive member escapes repository: {name}')
    current = target
    while current != repo:
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError(f'linked extraction target: {name}')
        current = current.parent
    return target


def build(repo, output, *, selected=None, status='USER_PAUSED', note=None):
    if output.exists():
        raise ValueError('snapshot output already exists; do not overwrite')
    files = (sorted(p for root in ROOTS for p in (repo / root).rglob('*') if p.is_file())
             if selected is None else sorted({safe_target(repo, name) for name in selected}))
    if any(not p.is_file() for p in files):
        raise ValueError('selected snapshot input is not a regular file')
    output.mkdir(parents=True)
    inventory, totals = [], defaultdict(lambda: {'files': 0, 'bytes': 0})
    with tempfile.TemporaryDirectory(prefix='rl-snapshot-') as tmp:
        archive = Path(tmp) / 'research.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
            for index, path in enumerate(files):
                name = path.relative_to(repo).as_posix()
                safe_target(repo, name)
                before = path.stat()
                digest, size = hashlib.sha256(), 0
                info = zipfile.ZipInfo.from_file(path, arcname=name)
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open('rb') as source, zf.open(info, 'w', force_zip64=True) as dest:
                    while block := source.read(CHUNK):
                        digest.update(block)
                        size += len(block)
                        dest.write(block)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or size != before.st_size:
                    raise ValueError(f'file changed during snapshot: {name}')
                inventory.append({'path': name, 'bytes': size, 'sha256': digest.hexdigest()})
                family = '/'.join(PurePosixPath(name).parts[:2])
                totals[family]['files'] += 1
                totals[family]['bytes'] += size
                if index % 250 == 0:
                    print(json.dumps({'archived': index + 1, 'total_files': len(files)}), flush=True)
                if path.name in {'summary.json', 'comparison.json', 'status.json', 'progress.json', 'data_summary.json'} and 'sequential_' in name:
                    dest = output / 'summaries' / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, dest)
        parts = []
        with archive.open('rb') as source:
            while block := source.read(PART_BYTES):
                path = output / f'research.zip.part{len(parts) + 1:03d}'
                with path.open('xb') as dest:
                    dest.write(block)
                parts.append({'path': path.name, 'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
        write_json(output / 'inventory.json', inventory)
        write_json(output / 'manifest.json', {
            'format': 'rl_research_split_zip_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'status': status, 'roots': list(ROOTS), 'file_count': len(inventory),
            'uncompressed_bytes': sum(x['bytes'] for x in inventory), 'archive_bytes': archive.stat().st_size,
            'archive_sha256': sha256(archive), 'parts': parts, 'inventory_sha256': sha256(output / 'inventory.json'),
            'families': dict(sorted(totals.items())),
            'excluded': ['.venv-torch', 'paper', 'personal app memory/settings/credentials', 'Git internals'],
            'note': note or 'All regular files under roots, including historical diagnostics and logs. No RL execution is authorized by build/verify/restore.',
        })
    print(json.dumps({'snapshot': str(output), 'files': len(inventory), 'parts': len(parts)}), flush=True)


@contextmanager
def verified_archive(repo, bundle):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['format'] != 'rl_research_split_zip_v1':
        raise ValueError('unknown snapshot format')
    if sha256(bundle / 'inventory.json') != manifest['inventory_sha256']:
        raise ValueError('inventory hash mismatch')
    inventory = json.loads((bundle / 'inventory.json').read_text())
    expected = {row['path']: row for row in inventory}
    if len(expected) != len(inventory) or len(inventory) != manifest['file_count']:
        raise ValueError('duplicate members or incorrect file count')
    with tempfile.TemporaryDirectory(prefix='rl-snapshot-verify-') as tmp:
        archive = Path(tmp) / 'research.zip'
        with archive.open('xb') as dest:
            for part in manifest['parts']:
                if Path(part['path']).name != part['path'] or '\\' in part['path']:
                    raise ValueError('unsafe split part name')
                source = bundle / part['path']
                if source.stat().st_size != part['bytes'] or sha256(source) != part['sha256']:
                    raise ValueError(f'split part hash mismatch: {source.name}')
                with source.open('rb') as handle:
                    shutil.copyfileobj(handle, dest, CHUNK)
        if archive.stat().st_size != manifest['archive_bytes'] or sha256(archive) != manifest['archive_sha256']:
            raise ValueError('combined archive hash mismatch')
        with zipfile.ZipFile(archive) as zf:
            if len(zf.infolist()) != len(expected) or set(zf.namelist()) != set(expected):
                raise ValueError('archive membership mismatch')
            for info in zf.infolist():
                safe_target(repo, info.filename)
                if info.file_size != expected[info.filename]['bytes']:
                    raise ValueError('uncompressed size mismatch')
                with zf.open(info) as handle:
                    if hashlib.file_digest(handle, 'sha256').hexdigest() != expected[info.filename]['sha256']:
                        raise ValueError(f'archive member hash mismatch: {info.filename}')
            yield zf, expected


def inspect_or_restore(repo, bundle, mode):
    written = 0
    with verified_archive(repo, bundle) as (zf, expected):
        if mode in {'restore', 'verify-local'}:
            # Complete conflict preflight before writing anything.
            for name, row in expected.items():
                target = safe_target(repo, name)
                if target.exists() and (not target.is_file() or sha256(target) != row['sha256']):
                    raise ValueError(f'local file differs; refusing overwrite: {name}')
                if mode == 'verify-local' and not target.exists():
                    raise ValueError(f'local file missing: {name}')
        if mode == 'restore':
            for name, row in expected.items():
                target = safe_target(repo, name)
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as source, target.open('xb') as dest:
                    shutil.copyfileobj(source, dest, CHUNK)
                if sha256(target) != row['sha256']:
                    raise ValueError(f'restored file hash mismatch: {name}')
                written += 1
        print(json.dumps({'verified_files': len(expected), 'mode': mode, 'written_files': written, 'rl_started': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build', 'verify', 'verify-local', 'restore'])
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--bundle', type=Path, default=Path('artifacts/research_snapshot_20260910'))
    args = parser.parse_args()
    repo, bundle = args.repo.resolve(), args.bundle.resolve()
    if args.mode == 'build':
        build(repo, bundle)
    else:
        inspect_or_restore(repo, bundle, args.mode)


if __name__ == '__main__':
    main()
