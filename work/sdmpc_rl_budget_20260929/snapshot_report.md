# Immutable Budget Baseline Snapshot

Status: **DONE** for the snapshot task. No simulation, controller/RL change, or
Git commit was performed by this task. The source project and old RL runs were
not modified. The brief was read first; no parent conversation history was read.

## Delivered Paths and Layout

All paths below are relative to
`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`:

- `work/sdmpc_rl_budget_20260929/freeze_runtime.py`
- `work/sdmpc_rl_budget_20260929/test_freeze_runtime.py`
- `work/sdmpc_rl_budget_20260929/snapshot_report.md`
- `artifacts/sdmpc_budget_baseline_20260929/`

The adapter's frozen project root is:

```text
C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source
```

The layout preserves every selected original relative path:

```text
artifacts/sdmpc_budget_baseline_20260929/
  manifest.json
  source/
    scripts/...
    work/sdmpc_externality_ablation_20260923/run_scenario.py
    work/sdmpc_slide_alignment_20260922/runtime.py
    work/sdmpc_matrix_14400_20260912/historical_tree/...
    outputs/sdmpc_budget_exception_all_20260922/protocols_0/<scenario>/...
    outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/sweet_170_incident_w/upper/...
    outputs/extended_matrix_no_slsqp_20260920/attempt_1/SDMPC6/sweet_190_skew15_w/...
    outputs/sdmpc_frozen_price_20260912/matrix_C_attempt_0/interval_vsl/sweet_190_w/...
```

Manifest SHA256:
`07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`.

Snapshot completion time: `2026-09-29T04:23:51.863572+00:00`
(`2026-09-29 13:23:51.863572 Asia/Seoul`).

| Contents | Files | Bytes |
| --- | ---: | ---: |
| Byte-preserved sources and inputs | 151 | 3,975,269 |
| Generated manifest | 1 | 313,059 |
| Entire artifact directory | 152 | 4,288,328 |

The payload is approximately 3.79 MiB; the complete artifact is approximately
4.09 MiB. Payload roles: 28 runtime sources, 44 historical runtime sources,
11 bootstrap fixtures/provenance files, 60 scenario protocol files, and 8
historical probe inputs. No optional probe input was missing.

## Inspected Dependency Chain

The actual entry is `work/sdmpc_externality_ablation_20260923/run_scenario.py`
with `--scenario sweet_170_incident_w --variant upper --externality on`.
It was inspected as text and parsed where needed; it was never executed.

1. The entry imports `matrix_common` from `work/sdmpc_upper_ttt_20260922`,
   `runtime.load`, and `run_validation.plain` from the slide-alignment workdir.
   `matrix_common.environment` reads the matrix historical environment JSON.
2. `runtime.load(30)` calls `profile_current.bootstrap` with the frozen-price
   `sweet_190_w` cell. Bootstrap reads that cell's
   `protocol_snapshot/protocol.json`, imports `run_cell`, and explicitly loads
   `work/sdmpc_matrix_14400_20260912/historical_tree`.
3. `run_cell.load_runtime` imports the historical plant/controller/factory
   modules, the two source-project `scripts` helpers, and `historical_config`.
   An independent AST traversal confirmed the 44-file transitive historical
   import closure, including conditional factory imports and package initializers.
4. Bootstrap also imports `ContinuousVSLPriceSDMPC`, even though this entry
   discards that returned class. Both its module and `frozen_price_controller.py`
   are included.
5. `runtime.load(30)` reads `decision_030/input.json`, `factory_config.json`, and
   `solver_options.json` under the saved `SDMPC6/sweet_190_skew15_w` run. All three
   are included explicitly; no other decision log tree is copied.
6. The entry's own externality/fixed/proximal/anchor modules resolve to its own
   directory. Their helper imports require central KKT reuse, selected-dual
   controller/multiplier, relative-band policy, local QP, group-block engine,
   ordered blocks, and `sparse/dual.py`. Those specific modules are included.
7. `group_block_engine` reads five historical Python modules to transform their
   ASTs at runtime. All five are included in the historical closure.
8. Historical YAML configuration, environment/compatibility JSON, original
   historical manifests, and `work/launch_final_r23.sh` are included to preserve
   the historical configuration/provenance checks. The shell script is a small
   hashed provenance input; it was not executed.

All ten scenario folders are included: `sweet_155_w`, `sweet_170_w`,
`sweet_170_incident_w`, `sweet_170_skew15_w`, `sweet_190_w`,
`sweet_190_skew15_w`, `sweet_190_incident_w`, `sweet_220_w`,
`sweet_220_skew15_w`, and `sweet_220_incident_w`. Their selected JSON inputs total
1,366,416 bytes. Each includes configuration, forecast, initial state, protocol,
and scenario; historical provenance and source manifest JSON are also included
where present. The unrelated protocols-root `validation.json` is excluded.

The follow-up request added exactly these existing files beneath the original
budget-ablation `sweet_170_incident_w/upper` directory:
`plant_004.json`, `plant_009.json`, `plant_019.json`, `plant_029.json`,
`plant_049.json`, `plant_069.json`, `completion.json`, and `contract.json`.
Their combined size is 513,942 bytes.

No broad historical output tree, third-party dependency tree, archive, large
result log, image, test tree, or old RL source/run tree was copied. The original
top-level `src` was not substituted for the historical runtime. This artifact
does not replace the destination repository's existing `src`.

## Provenance and Integrity Contract

Original project: `C:/Users/alsrj/Documents/Numerical Simulation`.
Source Git revision: `8c016939a6c8a6994b60ca0836d00fdcfe096209`.
Source branch: `codex/sdmpc-20260911`. Source dirty state: **true**.
Destination branch observed: `codex/sdmpc-rl-budget-20260929`.

The manifest records the exact source Git porcelain output before copying,
its SHA256, source revision/branch, and the after-copy provenance checks.
The captured status contains 1,843 entries, including untracked material.
Git status SHA256 is
`e5927777b80edfe70cecd2367c93bf2e3826b5e210cc7774fd0d4289b3021e5c`.
Revision and observed dirty-status hash agreed before and after copying.

Git reported permission warnings while enumerating unrelated directories below
`work/head_runs/numerical_sim_head_d11cd16`. The exact warnings are retained in
the manifest. Thus the recorded status is the observed Git status, not a claim
of exhaustive access to that unrelated archived checkout. All 151 selected
source/input files were individually readable and hash-verified.

Each manifest entry includes `source_relative`, `snapshot_relative`,
`original_absolute`, role, byte count, canonical `sha256`, and four matching
hash observations: source before, bytes copied, destination after, source after.
JSON fixtures are binary copies; BOMs, line endings, whitespace, and embedded
original absolute paths remain unchanged. No provenance path in original JSON
was rebased to the snapshot.

The freezer validates portable relative names, rejects traversal, absolute or
drive-relative names, Windows device/alternate-stream names, case collisions,
symlinks/junctions/reparse points, overlapping source/destination trees, missing
required files, and unmanifested destination files. It checks the entire
existing destination before copying. Existing different bytes are rejected;
matching files are not overwritten. New files use exclusive binary creation.
The completion manifest is written only after all copy/hash and Git checks pass.

A matching unfinished copy can be completed; a failed freeze has no completed
manifest. A completed snapshot can be verified offline. A repeated matching
freeze preserves the original manifest and every existing file. This is
application-enforced immutability with an external manifest hash, not an OS
read-only ACL or a cryptographic signature. Consumers must keep caches and
new outputs outside the artifact and should verify it before and after use.

## Dependency and Adapter Handoff

Freezing and its tests require only Python's standard library. The tested
interpreter is `.venv-torch/Scripts/python.exe`, Python 3.12.14. Its base
environment exposes NumPy 2.3.5 but no SciPy, pytest, numba, or matplotlib.
SciPy is required by the baseline optimization imports; pytest is not needed
for this freezer's unittest suite. No inspected active baseline module requires
numba; matplotlib is only referenced by unused plotting helpers.

The separately installed project `.deps-budget` metadata was independently
read with elevated access and confirms NumPy 2.3.5, SciPy 1.16.3, and pytest
8.4.2. An ordinary sandbox metadata read was denied. No packages were installed
or copied by this task. The parent reports successful original-runtime imports
with these dependencies and elevated execution; runtime imports were not
repeated here. There is no missing dependency for the freezer, and the parent
has supplied the identified runtime dependency, SciPy, through `.deps-budget`.

The user reports old source dependency metadata NumPy 2.5.3 / SciPy 1.18.1.
Those old versions are recorded as user-provided provenance, not as imported
versions. The historical TTT result **5508.8644 must not be used as new timing
or acceptance evidence** under the new dependency set. This restriction is also
machine-readable in `manifest.json`. The parent's reported saved-step-5 parity
and 60.66s/59.11s timings belong to its separate work; this snapshot task makes
no independent runtime parity or timing claim.

Final parent handoff: the parent reports `SNAPSHOT_RESET_OK`, observation length
2362, and simulation time 900 through its `budget_runtime` using this frozen
root. It has started separate frozen parity at step 30 and a step-10 probe.
These are parent-reported integration results, not executions by this task.
The required `sweet_170_incident_w` protocol already contains all seven original
JSON files, including `historical_provenance.json` and `source_manifest.json`.
No snapshot payload or manifest edits occurred after publication, including
while these parent checks were running; only this report was updated.

The parent must implement isolated bootstrap separately. In particular:

- Use the frozen `source/` as the source-project root and its nested historical
  tree for `src`; protect against the existing RL project's `src` import collision.
- Preserve original JSON absolute provenance as evidence while explicitly
  selecting frozen input paths in the adapter.
- Keep `.deps-budget` on the effective interpreter import path. Original
  `matrix_common.environment` assigns a historical `PYTHONPATH` referring to a
  dependency directory that is deliberately not copied into the snapshot.
- Launch with `-B`, place any runtime caches outside the snapshot/source
  project, and write all new simulation/probe results to separate output paths.

## Exact Verification Commands and Outcomes

Commands below were run in
`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL` using PowerShell.

### Unit Tests

```powershell
& '.venv-torch/Scripts/python.exe' -B -m unittest discover -s work/sdmpc_rl_budget_20260929 -p test_freeze_runtime.py -v
```

Exit 0: **18 tests passed**, no skips, in 0.923 seconds. Coverage includes path
validation, disjoint roots, missing inputs, case/reparse rejection, raw byte and
hash preservation, source modification times, offline verification, matching
repeat and partial snapshots, differing-file refusal before any other copy,
changed source/revision refusal, changes during copying, wrong copy hashes,
tampered snapshots/manifests, unexpected files, and explicit/optional selection.

### Freeze

```powershell
& '.venv-torch/Scripts/python.exe' -B work/sdmpc_rl_budget_20260929/freeze_runtime.py
```

The first sandbox attempt exited 1 before creating the destination because Git
returned exit 128 for dubious repository ownership. A direct captured-stderr
reproduction identified the source owner versus sandbox-user mismatch. The same
command with `require_escalated` exited 0 and returned `DONE`, 151 files,
3,975,269 bytes, and no missing optional probe inputs. No Git trust configuration
was changed, and no failed/partial artifact remained from the first attempt.

### Offline Snapshot Verification

```powershell
& '.venv-torch/Scripts/python.exe' -B work/sdmpc_rl_budget_20260929/freeze_runtime.py --verify-only
```

Exit 0 under the ordinary sandbox: `VERIFIED`, all 151 payload files and
inventory totals agree. The printed manifest SHA256 matches the value above.

### Real Repeat-Freeze Check

The following exact command was run with `require_escalated` for read access to
original Git provenance:

```powershell
& '.venv-torch/Scripts/python.exe' -B -c @'
from pathlib import Path
import sys
sys.path.insert(0, 'work/sdmpc_rl_budget_20260929')
import freeze_runtime as freeze
root = freeze.DEFAULT_DESTINATION
before = {p.relative_to(root).as_posix(): (freeze.sha256(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
selected, missing = freeze.select_files(freeze.DEFAULT_SOURCE)
manifest = freeze.freeze_snapshot(freeze.DEFAULT_SOURCE, root, selected, missing)
after = {p.relative_to(root).as_posix(): (freeze.sha256(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
assert before == after, 'Existing snapshot contents or modification times changed'
print('Repeat freeze: all', len(before), 'files retain identical SHA256 and modification times')
print('Manifest SHA256:', freeze.sha256(root / 'manifest.json'))
print('Created UTC:', manifest['created_utc'])
print('Source Git warnings:', manifest['source_git_before']['warnings'])
'@
```

Exit 0: all **152 files retained identical SHA256 and modification times**.

### Dependency Metadata

```powershell
Get-Content -LiteralPath '.deps-budget/numpy-2.3.5.dist-info/METADATA','.deps-budget/scipy-1.16.3.dist-info/METADATA','.deps-budget/pytest-8.4.2.dist-info/METADATA' -TotalCount 6
```

Initial sandbox exit 1, access denied. Same read-only command with
`require_escalated` exited 0, confirming the three package names and versions
listed above. An earlier `importlib.metadata.distributions` inventory yielded
empty dictionaries because the sandbox could not read the metadata; that
result was not treated as proof of missing installed packages.

### Static Dependency and Syntax Audit

```powershell
& '.venv-torch/Scripts/python.exe' -B -c @'
import ast
from pathlib import Path
import sys
sys.path.insert(0, 'work/sdmpc_rl_budget_20260929')
import freeze_runtime as freeze
root = freeze.DEFAULT_SOURCE
selected, missing = freeze.select_files(root)
historical = root / freeze.HISTORICAL
pending = ['work/run_claude_style_five_controller.py', 'src/controllers/player_sensitivity_dmpc.py', 'src/controllers/sensitivity_dmpc.py', 'src/evaluation/metrics.py']
seen = set()
while pending:
    relative = pending.pop()
    if relative in seen:
        continue
    seen.add(relative)
    path = historical / relative
    package = Path(relative).parent.parts
    for parent in path.relative_to(historical).parents:
        init = (parent / '__init__.py').as_posix()
        if (historical / init).is_file() and init not in seen:
            pending.append(init)
    for node in ast.walk(ast.parse(path.read_bytes(), filename=relative)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = list(package[:len(package) - node.level + 1]) if node.level else []
            base += (node.module or '').split('.') if node.module else []
            names = ['.'.join(base)] + ['.'.join(base + [alias.name]) for alias in node.names]
        for name in names:
            rel = name.replace('.', '/')
            candidate = next((p for p in [rel + '.py', rel + '/__init__.py'] if (historical / p).is_file()), None)
            if candidate:
                pending.append(candidate)
assert seen == set(freeze.HISTORICAL_SOURCES), (seen - set(freeze.HISTORICAL_SOURCES), set(freeze.HISTORICAL_SOURCES) - seen)
for name in selected:
    if name.endswith('.py'):
        ast.parse((root / name).read_bytes(), filename=name)
print('Historical transitive import closure: 44/44 selected')
print('Selected files:', len(selected))
print('Selected bytes:', sum((root / name).stat().st_size for name in selected))
print('Python syntax parsed:', sum(name.endswith('.py') for name in selected))
print('Missing optional probes:', missing)
'@
```

Exit 0: historical closure 44/44, 151 selected files, 3,975,269 bytes, all 72
selected Python files parse, and no missing probes. This is static evidence;
it does not claim an executed runtime import or simulation.

### Workspace Check

```powershell
git diff --check
git --no-optional-locks status --short --branch
```

Both exited 0. `diff --check` reported no whitespace errors in tracked changes;
the new snapshot/task files were untracked and therefore outside that command's
diff coverage. Git emitted an unrelated `.gitignore` line-ending warning and
sandbox warnings for the global ignore file. Parent changes to `.gitignore`,
`docs/rl_budget_execution_20260929.md`, and other disjoint task files were left
untouched. No add, commit, checkout, reset, or cleanup of old runs occurred.

## Scope and Remaining Work

The source `AGENTS.md` was inspected. Its controller/simulation acceptance and
source-report workflow do not apply to this explicitly limited snapshot-only
task; the user prohibits simulations and writes to the original project.
All manual code/report edits used `apply_patch`. All Python commands used `-B`.
The snapshot inventory contained zero `.pyc`, `.pyo`, `.nbc`, or `.nbi` files.

There is no outstanding snapshot blocker. Runtime adapter integration,
immutable runtime tests, action-influence probes, and any new acceptance or
timing measurement belong to the parent task. This report deliberately does
not claim those tests were run here.
