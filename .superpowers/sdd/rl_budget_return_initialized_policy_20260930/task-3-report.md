# Task 3: bounded on-policy MC critic fit

Implemented and tested on 2026-09-30. Final scoped result: **46 passed, 0 failed,
0 errors, 0 skipped** in 27.637 seconds of pytest execution. Authentication time
is additional. This is an implementation handoff for independent parent review,
not self-approval or an actual fit result.

Actual optimizer updates: **0**. Actual physical calls: **0**. No simulation,
collection, install, ACL change, commit, actor proposal, canonical dispatch, or
parent-owned documentation/decision edit was performed. The fixed actual output
`results/sdmpc_rl_balanced_goal_20260930/return_mc_v1` was absent both before and
after testing. Existing dirty repository changes were left alone.

## Changed scope

All implementation and generated evidence are new, under
`work/sdmpc_rl_return_mc_20260930/`:

| File | Responsibility |
| --- | --- |
| `mc_common.py` | Fixed spec, pinned pure helper reuse, hash checks, finite atomic JSON |
| `mc_data.py` | Artifact manifest, isolated authentication caller, 375-row loader, balanced sampler |
| `authenticate.py` | Fresh isolated read-only invocation of existing wave readout |
| `mc_learner.py` | Fixed-Phi/fixed-actor MC learner, continued Adam, diagnostics, resume state |
| `run.py` | One locked job, review receipt, STOP, durable checkpoints, final publication |
| `test_mc.py` | Scoped synthetic optimizer and lifecycle tests |
| `run_tests.py` | Scoped harness, read-only actual checks, before/after preservation evidence |

The only file written outside that new directory is this report. Generated
`test-evidence/{edaa0de0,e1345be4,8cc530f0,ec7d7d03}/` and
`synthetic-fixtures/` are retained. They include synthetic checkpoints, publication
artifacts, and interrupted/orphan evidence; none are actual fit outputs.

## Implementation contract

- One shared Phi, unchanged actor, unchanged 2367/64/64 architecture, twin residual
  critics, signed NP, float64 physical projection, and existing normalized
  requested-budget encoding. No output-scale change.
- Exactly the five Task 2 training trajectories, seeds 7301..7305, 75 controlled
  rows each. Trace anchors and requests remain float64. No observation-based
  reconstruction of anchors. Native float32 actions are projected in float64
  and required to match the saved requests exactly.
- `G_t = sum(r_t..r_74)`, gamma 1, reward `-interval_ttt/100`, warmup excluded,
  and only row 74 terminal. The loader reuses the pinned sequence/return helper;
  actual checks independently reconcile every return with saved interval costs.
- Actor, Phi, both critics, and both targets load exact parent tensors using meta
  construction, which consumes no caller Torch RNG. Only critic parameters have
  gradients enabled. There is exactly one optimizer: Adam restored from
  `learner.optimizers.critics`, including moments, step 250, and learning rate
  0.0003. Successful MC completion therefore leaves Adam at step 500 while the
  new MC counter is exactly 250.
- Dedicated `numpy.random.Generator(PCG64(7201))`, with saved RNG state. Each
  update samples 40 rows: for each scenario, its one forced terminal plus seven
  uniform draws with replacement from all 75 own rows, including terminal.
  All 40 entries are shuffled with that same stream. Totals are 2000 samples and
  250 forced terminals per scenario. Incidental uniform terminal draws are
  recorded separately from the forced quota.
- Each residual target is detached `G_pi1 - Phi(s)`. Loss is the sum of the two
  heads' MSEs. No bootstrap, target noise, target-policy call, old anchor loss,
  actor optimization, adaptive stopping, extra epochs, or error threshold gate.
  Target critics remain exact parent targets throughout updates 1..249; after
  optimizer update 250 they become an exact copy of the updated critics.
- Before/after diagnostics retain both heads and minQ per row, 25 scenario/horizon
  groups (whole trajectory, 51..75, 26..50, 1..25, terminal), pooled and terminal
  MSE/MAE/signed bias/max absolute error, exact terminal reward errors, actions,
  anchors, requests, projection clipping, saturation and cap coverage. Every
  update records both head losses, sampled indices, forced-terminal flags,
  balance counts and successful finite checks.

The offline process imports the existing small offline runtime for serialization,
hashing, STOP, process identity and kernel locking. Pinned `mlp`, `Actor`,
`residuals`, `projected`, `network_budget`, `episode_arrays`, `save_once`, and
`checkpoint` definitions are reused without importing the old learner or physical
runtime. The source import-boundary test confirms no generic `runtime`, `learner`,
`data`, wave worker, `budget_env`, or `src` module in the offline learner process.
No collection/dispatch/process framework was copied or introduced.

## Authentication and durability

`authenticate()` starts a fresh `python -I -B authenticate.py` process. That child
uses the existing `readout.validate_wave()` and requires exact equality with the
retained, SHA-pinned Task 2 readout. Its source validations replay actor inference,
validate observation/transition/control/accounting contracts, reconcile summaries,
and recompute the original health screens. Acceptance is not based solely on a
saved PASS flag. Boot, environment initialization/reset/step/restore, worker run,
and Adam step are replaced with rejecting guards in this read-only child.
`torch.load` permits only the pinned parent model and the five exact
`experience.pt` paths. Physical checkpoint files are hashed as bytes only.

The data identity binds 745 files: parent completion/model/output/source/data
provenance, all five wave settings/completions/traces/experiences and referenced
outputs, latest physical checkpoint bytes, timing/session/operation evidence,
helper source, and the frozen source snapshot. The exact full path-to-hash map is
retained in final `evidence.json:preserved_before_after_sha256` and
`readonly-actual.json:authentication.manifest.files`. All 745 hashes match before
and after the final test run.

The CLI fixes the output to `return_mc_v1`, requires a parent independent-review
receipt for exact source/spec hashes, and checks STOP at repository, goal, and
output scope. The existing kernel file lock enforces one writer. Process and
session records capture the actual PID, creation identity, parent PID, executable
and original interpreter command. Parent still owns global-eight-worker admission,
launching and actual process drain; session end is not process-exit evidence.

There is an initial durable checkpoint and one after each of the 250 updates.
Immutable checkpoint names and atomic hash-bound `latest.json` support resume
from the last durable update with identical weights, Adam and PCG64 state. Sources
are checked each update; all bound data files are checked before loading, at
startup, every 25 updates, and before final publication. Resume reauthenticates
the actual wave and rejects settings/source/data/model/spec/review mismatches.
Retained sample indices, quota evidence, terminal counts and RNG state are also
reconciled against a replay of the dedicated sampler stream.

Finite JSON now uses unique exclusive temporary files, fsync before atomic
replacement, and leaves prior temporary orphans untouched. The frozen helper's
fixed JSON temporary path initially failed a focused orphan-preservation test;
only the new adapter was changed. The pinned checkpoint helper is reused with
this new writer. Old source was not edited. Tensor checkpoints already use unique
temporary names. Partial outputs/orphans are preserved, and completed runs and
mismatched retained final models are refused.

Final publication reuses the exact durable done checkpoint, including across a
STOP after metrics or model export and a new process/session on resume. It never
rewrites the done checkpoint with a new process identity. `model_final.pt` is an
atomic hard link to that immutable payload; completion binds its hash, latest,
checkpoint, settings, and metrics. Both finalization STOP boundaries are tested.

The new format is `sdmpc-on-policy-return-mc-v1`. Checkpoints retain models, target
actor hash, parent model hash, critic Adam and its parent-state hash, PCG64 state,
data-array hash, complete source/data/spec identity, losses and sample evidence.
`counts.mc_critic=250` is separate from ancestor
`phi=1000, critic=250, actor=10, polyak=125`. Metrics record unchanged actor/Phi
hashes, old/new critic/target hashes, and continued optimizer provenance. No actual
new critic hash or actual post-fit metric exists yet because fitting was not run.

## Final evidence and commands

Working directory for all commands below:
`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

```powershell
# Initial complete scoped suite: 45 passed; superseded by the final source below.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_mc_20260930/run_tests.py'

# Run once to reproduce the orphan failure, then once after the fix.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_mc_20260930/run_tests.py' --synthetic-only -k atomic_json_keeps_prior_orphan

# Final complete scoped suite, including fresh read-only actual authentication.
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_return_mc_20260930/run_tests.py'

# Read-only whitespace check of the owned scope; no tracked diff errors reported.
git diff --check -- work/sdmpc_rl_return_mc_20260930 .superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-3-report.md
```

The first import/hash check below failed in the filesystem sandbox on an existing
`.deps-budget/colorama/__init__.py` read. The same command succeeded with elevated
read access, without an install or ACL edit: 745 manifest files; physical runtime
not loaded. Scoped test commands also used elevated access to those existing
dependency files. An ancillary PowerShell hash-list command had a parser error
and was corrected; it made no writes.

```powershell
& '.venv-torch/Scripts/python.exe' -B -c "import sys; sys.path.insert(0, 'work/sdmpc_rl_return_mc_20260930'); import mc_common as c; import mc_data as d; m=d.manifest(); print('manifest_files', len(m['files'])); print('spec', c.digest(c.SPEC)); print('physical_runtime_loaded', any(x in sys.modules for x in ('runtime','budget_env','worker')))"
```

Runtime: Python 3.12.14, Torch 2.14.0+cpu, NumPy 2.3.5. Final pytest timestamp:
2026-09-30T12:05:58.832288+09:00. Only the new Task 3 suite was run; no earlier
whole learner or wave suite was rerun.

Coverage includes actual return/terminal/warmup reconciliation and float64 request
parity; synthetic signed-NP/cap projection; exact initialization/continued Adam;
critic-only gradients and bit-exact frozen weights; no global RNG consumption;
MC targets/summed-head MSE/no bootstrap; full fixed quotas and target-copy timing;
bit-exact 73+177 deterministic resume including optimizer/RNG/history; invalid
parent and resume provenance; source/model/data hash failures; readout equality
despite unchanged PASS flags; all STOP scopes; kernel lock exclusion; finite JSON;
orphan preservation; interrupted runner at durable step 7; settings/checkpoint
hash mismatch; both final publication STOP boundaries; and completed/mismatched
final refusal. Synthetic arrays use independent seeds 9901/9902. Their fabricated
ancestor fixture has nonzero moments from one synthetic step with its test-only
counter set to 250; it is not an actual trained ancestor or a traffic result.
Actual arrays were never passed to `Learner` in this test harness.

Evidence paths below are under `work/sdmpc_rl_return_mc_20260930/test-evidence/`.

| Evidence | Result | SHA256 of evidence.json |
| --- | --- | --- |
| `edaa0de0` | Initial 45 pass, 29.282 s | `20f2713df726cb252fbac4cf60a4d51b32c1c9a9c82b514fc3ba1c9ebdb1b443` |
| `e1345be4` | Expected orphan regression: 1 fail, 45 deselected | `8ef4024cc56f29949880e8e8afe90b9b3c91039ca028dcbaa32475eb0e33007d` |
| `8cc530f0` | Fixed orphan regression: 1 pass, 45 deselected | `f2d60c986ffcdf205fff2767c2261e26ef19c426ca75eea2b9a67cc01751cd60` |
| `ec7d7d03` | **Final 46 pass, 27.637 s** | `818f48a146982eea43fa44105073a3bbdf12a7b0adb36e9e57a5af23aef2483e` |

Final `ec7d7d03/tests.xml` SHA256:
`99f0b499586b5235760866074cec1daaa7aeeb9ffb901fbf5346a0410067dc46`.
Final `ec7d7d03/readonly-actual.json` SHA256:
`d19730c57cdc89ab09e6dad2615193309eec675bb373071857bb4011d4e756c7`.
The full recomputed readout and all 745 hashes are included there.

Regression XML hashes: failing `e1345be4/tests.xml`:
`acf68ba72dafc4617598aadac58501019a2db7c9833866509deeb72167294666`;
passing `8cc530f0/tests.xml`:
`5cbcac472474769349757f3b96ef1151b130ca2d2b969840aa093b0a1c4b1d79`.
Initial XML: `ee323d8aae7c0dda058802424c8c690dc1e6d99e033eb427252b4d7752503daa`.
Initial read-only evidence: `97afe7fc37d3bc5ff1717f8823b59dbd5f98ebf922006006e5c8dff63e098e2a`.

## Actual read-only findings

All 375 loaded observations have width 2367. Each scenario has exactly one true
terminal and 75 rows. Every NUF request is exactly 6000. First returns reconcile
to `-(total_ttt - warmup_ttt)/100`; terminal returns equal terminal rewards.

| Scenario | G0, warmup excluded | Terminal reward | Excluded warmup TTT |
| --- | ---: | ---: | ---: |
| sweet_155_w | -38.77601173655539 | -0.2091181801491666 | 110.88484315684288 |
| sweet_170_w | -41.522577309703685 | -0.2066690199972345 | 110.14137724659363 |
| sweet_170_incident_w | -55.424174565477664 | -0.2069491428286983 | 109.84285378493286 |
| sweet_170_skew15_w | -38.842091352940734 | -0.2049083140218363 | 108.33685500334093 |
| sweet_190_w | -66.8144825705665 | -0.20611440157049402 | 109.57538969878446 |

| Identity | SHA256 |
| --- | --- |
| Fixed MC spec | `3610bb43c22430a3fb9ba95f7f14175b2b29b68a2000d6bfe1af04ec4c8ff811` |
| Retained Task 2 readout | `e8d2795c3828069d88f83797d1b644d3cba9f87f58d428b84ee387601777f1db` |
| Parent model file | `7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904` |
| Parent completion | `34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757` |
| Loaded 375-row tensor arrays | `e0e4b59f7cd13b90ac94a3906bdcc3fdaa0864f96a57305db34a73da5ed41d02` |
| Parent actor tensor state | `9956ba4f6f5d64c7a65bf05bdab47984efe041e15358dafebbb9bfd731d44877` |
| Parent Phi tensor state | `944ac15b10df6375c728128f09f5d78cdc69e1c914cdc4a84326f4d05b102a50` |
| Parent twin critic tensor state | `d72c5c28604843129d596ff4d7cb6597047cd305aa842ddd26472f209317a644` |
| Parent target critic tensor state | `6f36554d4cf16fc7383f8ce9be3348e56c2da653d406fe38f23449193eb67ba6` |
| Parent critic optimizer structured state | `448ab779287a1b00ad6de4b803c7078ce9e01531e80843addf6cf6c2fd69be86` |

The actual parent Adam group was inspected read-only: lr 0.0003 and all parameter
state steps 250. Tensor hashes use the frozen tensor-hash helper; optimizer hashes
use `mc_learner.state_hash`, not the nondeterministic Torch archive byte format.

Wave artifact hashes, with paths relative to
`results/sdmpc_rl_balanced_goal_20260930/return_policy_wave_v1/`:

| Scenario | File | SHA256 |
| --- | --- | --- |
| sweet_155_w | settings.json | `017f6044ecacb07863f5e5bdc3e5eb471a6739d8b23e16cf22daec3c6bf89860` |
| sweet_155_w | completion.json | `8e7c3901f6b187f555dd1a671983f96713785f45bbd47c3904da78d9bdadfb8a` |
| sweet_155_w | trace.json | `206a84dd0a9ec7a0c1354700216d23fe9953b93b24f2a908627ce0055fb464eb` |
| sweet_155_w | experience.pt | `a3ccf679aefa63f6b8c5c4f6dea3659d168006c683e72efcf25ac4d8e24ce931` |
| sweet_170_w | settings.json | `c4409ebf9a8effdf7c8f4257326ceaf87c29cfae75c7c627a15ccc5561542e80` |
| sweet_170_w | completion.json | `2cb4438ccf694cbdb62157ef52315304d1e62d6fc2dc6218ed6030a74030b568` |
| sweet_170_w | trace.json | `22d0b6e1d8e031b39c51d86d502a47c3cfa3a98fcc8ff14069e734a14c63a48d` |
| sweet_170_w | experience.pt | `b2f63d1159f36cdba24b6d2c3c17e5b4b415379888a14094908e8b917f77f2d1` |
| sweet_170_incident_w | settings.json | `f61d56f9b761cc4a033d1a3a7c05d61ee4c17e242e15c9e26d9e64c4c2006f50` |
| sweet_170_incident_w | completion.json | `8747d0a9aa80097655eaf28e1ddf0713e5710bdef6ade8ea36f53b68142c71b4` |
| sweet_170_incident_w | trace.json | `8b42e1f5c0c0fa893f0090f183d72ca7c0bd7d9bf892c53712b3fca85e2a3325` |
| sweet_170_incident_w | experience.pt | `bdab164cb0c075d6e106b97487c390e2191bf168503b05eb357efb4dd2533c1c` |
| sweet_170_skew15_w | settings.json | `0280eaeeb7dd20be97149ba973201d83c5be8876a81ee2ee10b1d99c8dced750` |
| sweet_170_skew15_w | completion.json | `bc558a1ae5fef246c837aee055a56a3262e8e3df6a42df221c5b3be5d00914e5` |
| sweet_170_skew15_w | trace.json | `583b022e667a00f27ca84f24ce6b5c1fa4f1dd6ada59c616c6e84348d0a677da` |
| sweet_170_skew15_w | experience.pt | `80bfde68983f368f1cb9b3a40b1dcf6ca8b12427aa10d3a215039aa2fe86057f` |
| sweet_190_w | settings.json | `e8c9487cb81e9db8f7694f524c93ca7864ba1857ffa42c9eda7c91187e7b62c2` |
| sweet_190_w | completion.json | `556718d1de681d07f7aa1446f999a7d08d53ef03acafd6b602a76d86f93236bb` |
| sweet_190_w | trace.json | `408ed8b107c9fe856d26c077b31538c4cc5ebe8a146e3a880d7694ae67fb6938` |
| sweet_190_w | experience.pt | `5d393cb3e147c27bd1f68f8e0d4270dcec988d0c6f996e5bb3fc68cc47ee74d4` |

## Final source hashes

New files, relative to `work/sdmpc_rl_return_mc_20260930/`:

| File | SHA256 |
| --- | --- |
| authenticate.py | `c3c99bd02566d4a540cb091a2b58692901d0097348ece3a1011f154de21437e4` |
| mc_common.py | `8a4f76d37a7f6682a61d5c2224fac948bd68a35fc670e0b64f617ca0eb1219bd` |
| mc_data.py | `755e79a2d7288b10fa2c51a7005913b141384fd79b288ccd6acc98327bdeca92` |
| mc_learner.py | `d4d8038163f7a1d52fdbdb20c0d98280fc745ec4e9719eab2a541766e055752d` |
| run.py | `3469ed66acec337fce43202dc5e26d03eb39bc08b9af6cdec3450f0c0829d39a` |
| run_tests.py | `a4a35467c77834ce9d515e96831d7f9d5816664e88e6e8ec6d9e6b2df4a5f6cd` |
| test_mc.py | `2507b7c98d311bdd077a35f01ac61a5d8c58dc2150b592a5d1cf2be90d5edcc8` |

Unchanged predecessor/helper sources included in the review source identity:

| Repository-relative file | SHA256 |
| --- | --- |
| work/sdmpc_rl_multi_20260929/budget_runtime.py | `6ae552bad2c092af9d2ab55b0c81e34558754861217fe427788820689800e9a6` |
| work/sdmpc_rl_nuf_retention_20260930/launch_identity.py | `5cc1a94037b349072ff7ff37e9ceb756c3205b84cb8c1940b4305a5f1514e160` |
| work/sdmpc_rl_return_init_20260930/authenticate.py | `747ca1b8e1e2f8787e3fe368f0a190f2c9874cba9c04e85df073269b336063a5` |
| work/sdmpc_rl_return_init_20260930/data.py | `10a8e894cd6c54a77fe26ed96bd314026e9390a6674b7ad64960fb82c9102cdb` |
| work/sdmpc_rl_return_init_20260930/learner.py | `b667bcf8accbcc5ed105a4c2069017e1d29f73274826d5052590fb804a912ebe` |
| work/sdmpc_rl_return_init_20260930/run.py | `f1019cc867c074810493024e40716cf801cde4987c172dbbf85bde1c966ef5d6` |
| work/sdmpc_rl_return_init_20260930/run_tests.py | `7b395f1352bf0a6070c73ce8bcd5605c385f5b400b8565583940524d280949c2` |
| work/sdmpc_rl_return_init_20260930/runtime.py | `8663b77b76abeb3777c827470491bfb14ab332f2523ee044e7348ecdd9af4975` |
| work/sdmpc_rl_return_init_20260930/test_return_init.py | `13a1689aeaa708c1538b29f9555bd6cd209f98ef3d959a7570391c2b671ec016` |

The final evidence's `source_sha256` object is the complete 16-entry review object.
Its original parent source set is retained for provenance; only the small runtime
and named pure definitions are executed in the offline process. Wave sources and
their transitive frozen inputs are additionally bound in the 745-file manifest.

## Parent-only actual run

Not executed. Parent must independently review this exact source and create its
own `task-3-review.json` with `status: "approved"`, a nonempty `reviewer`, the exact
16-entry `source_sha256` object from final evidence, and `spec_sha256` equal to
`3610bb43c22430a3fb9ba95f7f14175b2b29b68a2000d6bfe1af04ec4c8ff811`.
No approval receipt was fabricated. Parent owns the current global budget check,
including the unrelated numerical worker, hidden launch/redirection, and drain.

```powershell
& '.venv-torch/Scripts/python.exe' -B -u 'work/sdmpc_rl_return_mc_20260930/run.py' --review '.superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-3-review.json' --output 'results/sdmpc_rl_balanced_goal_20260930/return_mc_v1'
```

For an interrupted admitted run, use the same command plus `--resume` after the
parent resolves any STOP. Never delete partial evidence, force retraining of a
completed phase, or choose a different output directory. Exit 0 means completed;
exit 3 means STOP; uncaught validation/runtime failures exit nonzero. Resume after
finalization STOP does not add optimizer steps or rewrite the exported checkpoint.
There are 251 durable checkpoints in an uninterrupted run; preserve them and any
orphans. Exactly one final model is exported. No automatic downstream action runs.

## Interpretation and remaining boundary

These returns are observed on-policy Q^pi1 samples at `(s,b_pi1)`, not Q-star or
labels for other actions. Before-fit Q-carry versus G-pi1 is a continuation
discrepancy, not independent Q-pi1 calibration. After-fit errors are TRAINING fit
on the same 375 rows, not heldout evaluation. The all-6000 NUF requests provide no
NUF action variation and cannot identify that axis. Existing local exploration
data remain untouched; they are not included in this fit.

Actor and Phi remain bit-exact, so this critic-only fit leaves the physical policy
unchanged and supplies no traffic-TTT improvement claim. Parent must review the
actual bounded fit before deciding any policy update or canonical evaluation.
Actual fit quality, actual final critic hashes and actual job timing remain
unmeasured by design.
