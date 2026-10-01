# Scoped Windows launcher recovery

Implemented in `work/sdmpc_rl_local_20260930_v2` only. This report is the only
edited file outside that new directory. No actual migration (including dry-run),
checkpoint deserialization, boot/restore, physical collection, model training,
installation, commit, persistent OS change, or model override was performed.
Existing runtime helpers and v1 sources/results were read-only.

Final result: **236 passed in 78.46 seconds**, comprising all 200 inherited local
tests and 36 new recovery tests. The actual v2 result directory remains absent.
The 54 entries in the parent's v1 preservation manifest all retain their exact
lengths and SHA256 hashes.

## Implementation

The eight reviewed Python files were mechanically copied to v2 before editing.
`collect.py`, `exploration.py`, `frozen_inventory.py`, `validate.py`, and
`test_exploration.py` remain byte-identical to v1. The exploration formula is
unchanged. `FORMAT` remains the existing serialization format; strict source
identity and the new canonical root distinguish v2. Normal collection has no
legacy settings/checkpoint acceptance path.

`local_runtime.py` now pins `local_budget_v2`. Reservations contain separate
`owner`, `launcher`, and actual `worker` records. Only the reserving owner can
bind the launcher, once. A worker waits for binding before claiming, then must
prove live identities and matching creation times for the owner, launcher and
itself. Direct children must name the owner as parent. A different worker PID is
admissible only in Windows venv redirector mode with the bound launcher as its
immediate parent and consistent creation order. The verified actual parent PID
and launch mode are retained. Token consumption and the five-slot limit remain
inside the shared ownership lock. No process enumeration or alternate
interpreter is used.

An unbound reservation stays UNKNOWN after coordinator loss. An active actual
worker remains protected if its launcher exits or its liveness is unknown. A
coordinator cannot release such a worker. A dead/reused bound launcher can retire
an unclaimed reservation because a future claim requires that exact launcher to
be live. A self-releasing worker has already closed its numerical environment.
Coordinator cleanup waits for every returned launcher even if a release is
refused, and preserves the refusal instead of admitting overlapping work.

`launch_identity.py` holds the stdlib-only verifier and the existing single-PID
Windows creation-time probe. `identity_probe.py` exercises it with the unchanged
venv interpreter, without importing NumPy or Torch in either probe process.

Production v2 collection requires a committed prefix migration before admission.
An incomplete migration blocks both coordinator and direct worker entry points,
including the ownership critical section.

## One-Time Migration

`migrate_prefix.py` is deliberately pinned to this incident and these paths. It
has no CLI option for alternate roots, hashes, identity, or relaxed validation.
Its default is a write-free dry-run. Only `--execute` publishes results.

It authenticates the parent's manifest and every listed input, rejects changed,
extra or moved v1 files, and verifies the exact checkpoint/settings pins. It
checks old source hashes, physical/runtime/contract equality, the immutable wave
plan, failed attempt, launch commands, logs, reservations and absence of session
starts for the failed tokens. All prior numerical owners/workers must be proven
dead, including PID creation-time handling; unknown status is rejected. STOP is
checked for both cohorts and all slots. Existing v1 root, ownership and ten slot
locks are acquired through read-only handles without changing their bytes.

The physical loader is available for the parent's later migration invocation:
boot the frozen runtime, deserialize, apply the existing per-transition and
boundary validators, replay the small policy/RNG state, and restore for the
observation/physical-contract check. It does not call reset, step or PFO. The
existing restore path can evaluate its saved physical reference; that validation
cost is not new collection and is not added to prior worker timing.

Only the settings identity changes. The checkpoint retains all other settings,
the full physical object graph, trace, transitions, observation, policy/RNG,
session IDs and run ID. Equality checks compare types, dictionary keys, sequence
lengths, array dtype/shape/bytes and saved object attributes, including hidden
cache attributes; remaining serializable values are compared as protocol-5
pickle bytes. This check is repeated after staged checkpoint serialization and
fresh deserialization. Existing physical/transition validation also runs on the
new checkpoint.

The pending marker is created inside the intended v2 root before destination
slot-lock acquisition. All output publication occurs with the root, ownership
and ten slot locks held. Staging remains inside `.migration/stage`; it retains
byte-exact original v1 result files, original source files and the preservation
manifest. New settings/checkpoint, schema, timing, sessions, cohort history and
the v2 plan are staged and hashed. Publication uses exclusive file creation and
fsync, then rechecks hashes, STOP and identity before publishing the durable
commit marker last. Incomplete staging/publication is preserved and rejected on
retry; there is no deletion, automatic repair or overwrite of an existing run.

An immediate repeated migration recognizes the committed receipt, verifies all
staged/provenance and published hashes, checks all destination locks/liveness,
and returns the original receipt without boot/deserialization or writes. This
verification is intentionally strict: after ordinary collection advances the
published checkpoint/history, a migration retry refuses the changed published
outputs. Continue that run with normal `--resume`, never by remigrating it.

## Prefix And Timing

Pinned checkpoint SHA256:
`659f936702be59880760677d76a8ccc35957a03aa881a3fe85a9fb4ea1923816`

Pinned settings SHA256:
`933269729875452e776be948fc8a66b8af15423f6b63e732c6fcfb6328ee5571`

Retained run ID: `2962784e595d4ee8b1272dda7762dbc6`.
Failed attempt: `bae7788116f247999100f4ad6a2d3d19`.

The only migrated timing sessions are:

- `3e5d7caa2d974494adbb285ce0023456`, the first physical interval.
- `dffca16d89f344b0a5d9910f75077977`, the second physical interval.

Their start/end records and timing ledger are copied byte-for-byte, retaining
**66.75636630004738 seconds** of aggregate locked worker-session time. The five
failed unclaimed launch intents, including carry155's third reservation, are
retained in provenance and the migration receipt, not inserted as measured
sessions or as UNKNOWN timed sessions. They failed before session start/boot.
Coordinator and interpreter startup/teardown overhead is outside the defined
locked-session scope. Its elapsed cost is not measured here and is not zero.

The v2 carry155 ownership history contains exactly the two prior sessions. It
retains progress=2 and the same run ID, so normal resume starts at `k=7` (interval
3), not reset. The other four carry slots map to their authenticated failed old
intents and have no checkpoint/session history preventing their first physical
episode. Local behavior has no old trajectory. The total remains **750 = 2 reused
+ 748 additional**, 150 transitions per scenario; the reused prefix is never
reported as two newly collected transitions.

## Exact Test Evidence

All commands below ran from
`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL` with the existing interpreter and
dependencies. Pytest cache and bytecode writes were disabled. No old phase suite
was rerun. The inherited suite ran only from its v2 copy, with fake environments.
The copy's fixture disables mandatory migration only for fresh synthetic episodes;
two inherited launch tests add an explicit synthetic binding and parent identity.
No inherited assertions were removed or weakened.

Untouched-copy baseline command:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_collection.py","work/sdmpc_rl_local_20260930_v2/test_exploration.py","--tb=short","--show-capture=no"]))'
```

Result: **200 passed in 64.68s**. The same command after strict binding was added
initially produced **2 failed, 198 passed in 87.69s**: two inherited synthetic
launches had never bound a process. Their fixtures were updated as described.

Initial ownership regression command:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_windows_recovery.py","--tb=short","--show-capture=no"]))'
```

Result before repair: **3 failed, 5 passed in 1.45s**. Two failures reproduce
`Launch reservation process identity differs`; the new verifier module did not
yet exist. After implementation the following command produced **8 passed in
1.66s** and printed launcher 44012, worker 34336, parent 44012, creation time
134351844166065306, `numerical_imports=false`:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-s","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_windows_recovery.py","--tb=short"]))'
```

Initial migration test collection failed with **1 error in 1.43s** because the
new `migrate_prefix` module had not been implemented:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_migration.py","--tb=short"]))'
```

The executable synthetic migration suite then used:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_migration.py","--tb=short","--show-capture=no"]))'
```

It first produced **5 failed, 9 passed in 5.02s**: Windows rejected second-handle
reads of byte-locked files. Reading those bytes through the retained read-only
lock handle fixed the issue; the rerun produced **14 passed in 6.39s**.

Expanded focused command:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_migration.py","work/sdmpc_rl_local_20260930_v2/test_windows_recovery.py","--tb=short","--show-capture=no"]))'
```

Initially **1 failed, 33 passed in 11.10s**: the idempotent path had not checked a
held destination slot lock. After adding those lock checks and the mandatory
production admission test: **35 passed in 10.33s**.

The final new drain regression was demonstrated before its fix:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_windows_recovery.py","-k","drain_waits","--tb=short","--show-capture=no"]))'
```

Result: **1 failed, 13 deselected in 1.43s**. Only the first fake launcher had
been waited on after a refused release. Cleanup now finishes all launcher waits.

Final complete command:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import sys; sys.path.insert(0,".deps-budget"); import pytest; raise SystemExit(pytest.main(["-q","-p","no:cacheprovider","work/sdmpc_rl_local_20260930_v2/test_collection.py","work/sdmpc_rl_local_20260930_v2/test_exploration.py","work/sdmpc_rl_local_20260930_v2/test_windows_recovery.py","work/sdmpc_rl_local_20260930_v2/test_migration.py","--tb=short","--show-capture=no"]))'
```

Result: **236 passed in 78.46s**, exit 0. Coverage includes wrong parent,
launcher PID reuse/unknown identity, unknown/older worker, direct child support,
binding/claim once, delayed binding, unbound owner death, live actual worker after
launcher exit, five redirected workers, coordinator draining, read-only dry-run,
idempotence, old/new STOP, active/unknown processes, moved/corrupt input files,
extra sessions, semantic transition/terminal/physical/policy corruption,
restore without reset/step, normal synthetic continuation at interval 3,
held destination slot locks, and interruption before/after published output files.

Syntax-only command:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import ast,pathlib; paths=sorted(pathlib.Path("work/sdmpc_rl_local_20260930_v2").glob("*.py")); [ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p)) for p in paths]; print("AST_OK",len(paths))'
```

Result: `AST_OK 13`, exit 0.

## Actual Process-Only Evidence

Original reproduction, with no numerical imports:

```powershell
.venv-torch/Scripts/python.exe -B -c 'import os,sys,subprocess,json; p=subprocess.Popen([sys.executable,"-B","-c","import os,json; print(json.dumps(dict(actual=os.getpid(),parent=os.getppid())))"],stdout=subprocess.PIPE,text=True); out,_=p.communicate(timeout=15); print(json.dumps(dict(launcher=p.pid,child=json.loads(out),exit=p.returncode,executable=sys.executable,base=sys._base_executable)))'
```

Observed launcher **27496**, actual child **1732**, child parent **27496**, exit 0.

Final standalone verification:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/identity_probe.py
```

Observed owner PID **44340**, creation **134351851339454824**; launcher PID
**9264**, creation **134351851340215685**; actual worker PID **48380**, creation
**134351851340896799**; parent **9264**; mode `windows-venv-redirector`; exit 0.
Both parent and child report numerical imports **false**. Interpreter stayed at
`REPO/.venv-torch/Scripts/python.exe`; base executable remained
`C:/Users/alsrj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
All returned probe subprocesses were waited on and exited.

## Preservation And Limits

Manifest SHA256:
`cdbbc81d260887ccd2e74221d9ea44befd88e5f00013988aeaf09be927994262`.
Length/hash comparison of all entries before and after implementation:
**54 files, 0 mismatches**. Final direct hashing also reconfirmed the two prefix
pins above. `Test-Path results/sdmpc_rl_balanced_goal_20260930/local_budget_v2`
returned **False**. The v2-v1 comparison changes only `local_runtime.py`,
`run_wave.py`, five synthetic-fixture lines in `test_collection.py`, and five new
Python files. No v1 file was edited.

The first naive `-m pytest` invocation reported no pytest module. Adding the
existing `.deps-budget` path then hit its sandbox read ACL; approved elevated
test invocations used those same installed dependencies. Automatic approval
review rejected one migration test invocation because the fixture's dynamically
assigned `OLD_ROOT` was not clear enough to establish isolation. No test ran in
that rejected invocation. The fixture was changed to explicit assignments and
temporary-root assertions for `OLD_ROOT`, `MANIFEST`, `REPO` and `COHORT_ROOT`;
physical loading is mocked to the synthetic checkpoint. The clarified invocation
and subsequent suite runs were approved. No installation or ACL change was made.

Remaining review/validation limits:

- The real two-interval checkpoint was never deserialized, restored or migrated.
  The parent's independent review and later real migration validation are still
  required, including frozen-class deserialization, restore observation equality
  and byte/exact-state comparison. These cannot be claimed from synthetic tests.
- No numerical startup/collection smoke was performed. Only the identity
  verifier was exercised with real subprocesses. Actual solver behavior and
  collection performance remain untested in this task.
- Unbound launches and UNKNOWN liveness intentionally block recovery. The repair
  does not guess another PID or enumerate candidates. Interrupted migration
  staging similarly requires review; it is never automatically discarded.
- Publication is a guarded multi-file transaction with a final commit marker and
  fsynced files, not an atomic filesystem directory transaction or a tested
  power-loss guarantee. File-publication interruptions are covered synthetically.
- Committed migration verification refuses legitimately advanced collection
  outputs rather than overwriting them. Idempotent retry is verified before any
  additional collection; subsequent work must use normal `--resume`.

Parent-only next commands, supplied for review and **not executed here**:

```powershell
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/migrate_prefix.py
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/migrate_prefix.py --execute
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_local_20260930_v2/run_wave.py --resume
```
