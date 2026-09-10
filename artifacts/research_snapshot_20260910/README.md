# Research Snapshot: 2026-09-10

**USER PAUSED. Restoring this archive does not authorize or start experiments.**
Start with [the project handoff](../../RL_HANDOFF_20260910.md).

## Contents

- 3,039 original files under `data/`, `results/`, `models/`, `checkpoints/`.
- Original size: 2,206,836,922 bytes (about 2.06 GiB).
- Compressed ZIP: 734,164,552 bytes (about 700.15 MiB), split into 18 parts.
- Each part is at most 40 MiB. All parts are required; no Git LFS client is needed.
- Full ZIP SHA256: `e2598c0580c6a07a8bff4e770c2855dc219b6cb3a7264ff37ad34ce236c67970`.

[manifest.json](manifest.json) records the part hashes, original directory-family sizes and capture exclusions.
[inventory.json](inventory.json) records every original relative path, size and SHA256.
[summaries/](summaries/) exposes sequential experiment summaries/status/progress as ordinary JSON for browsing.
[runtime_environment.json](runtime_environment.json) records the existing Python/package metadata, not a portable dependency lockfile.

This includes raw replay, trained models, simulator checkpoints, source experiment manifests, full traces,
historical results and logs. Old `process.json` files describe historical process identities, not live processes on a new machine.
The two current evaluation checkpoints and user STOP files are preserved.
Virtual environments, the Word paper draft, personal app settings/memory/credentials and Git internals are excluded.
Source code, tests, plans and project memory documents are versioned normally outside this bundle.

## Verify or Restore

From the repository root, using Python 3.11 or later:

```powershell
python -m work.research_snapshot verify
python -m work.research_snapshot restore
python -m work.research_snapshot verify-local
```

Verification joins parts in a temporary directory and checks the full archive and every member hash.
Restoration validates the entire archive and all existing-file conflicts before writing missing files.
It never overwrites a differing local file, changes STOP, enables automation, unpickles a checkpoint, or starts RL.
The original machine already has these files; no extraction is necessary there.
Allow room for the temporary compressed ZIP plus about 2.06 GiB when restoring on a fresh machine.

Exact checkpoint/source hashes and pause state are documented in the handoff and audit.
Windows absolute paths in some historical artifacts may require an explicitly versioned portability adaptation.
Do not rewrite pinned artifacts and silently present the result as the original experiment.

## Verification at Export

- All 3,039 archive members matched the local originals, byte for byte.
- Split archive hashes and complete ZIP hash passed.
- Three snapshot-tool tests passed: round-trip/STOP preservation/no overwrite, path rejection, and tampered-part rejection.
- No RL training or simulation was run for this export; the related scheduled task remains paused.
- A pattern-based scan found no common access-token/private-key signatures in the research text files; this is not a comprehensive security audit.
