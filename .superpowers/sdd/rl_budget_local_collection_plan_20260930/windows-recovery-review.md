# Scoped Independent Windows Recovery Review

SPEC: **PASS** for the submitted v2 delta.

QUALITY: **PASS** for the submitted v2 delta, with the validation limits below.

Concrete blockers: **None found.** These are source-review verdicts, not claims that the real migration or numerical restart has succeeded.

Reviewed the brief first, then `windows-recovery-report.md` and `windows-recovery.diff`, and checked the corresponding source and tests. Review was limited to launcher/worker identity, startup and owner-loss nonoverlap, and the explicit two-transition migration followed by ordinary v2 resume. Unchanged helpers were read only where needed to trace those interfaces; their admitted design was not reopened.

No tests, process probes, collector subprocesses, migration commands (including dry-run), checkpoint deserialization, boot/restore, physics, training or evaluation were run. No model override was used. This review file is the only file written.

## Independently Checked Evidence

- Read-only length/SHA256 comparison against `v1-frozen-inputs.json`: **54 entries, 0 mismatches**. This includes all 46 listed v1 result files and all eight v1 Python files.
- Manifest SHA256: `cdbbc81d260887ccd2e74221d9ea44befd88e5f00013988aeaf09be927994262`.
- Checkpoint SHA256: `659f936702be59880760677d76a8ccc35957a03aa881a3fe85a9fb4ea1923816`.
- Settings SHA256: `933269729875452e776be948fc8a66b8af15423f6b63e732c6fcfb6328ee5571`.
- `collect.py`, `exploration.py`, `frozen_inventory.py`, `validate.py` and `test_exploration.py` in v2 are byte-identical to v1. The directory comparison shows three modified files and five new files, matching the submitted diff.
- The v2 result root `results/sdmpc_rl_balanced_goal_20260930/local_budget_v2` was absent at review time.
- Read-only JSON inspection confirms the retained run ID `2962784e595d4ee8b1272dda7762dbc6`, the two prior session tokens, progress 1 then 2, and five failed launch intents. The pinned settings retain carry/sweet_155_w, capacity 6000, CPU mask 1, seeds 6801/6901, evaluation=false and model_sha256=null.

## SPEC Assessment

| Requirement | Evidence and assessment |
| --- | --- |
| Windows redirector and direct-process identity | [launch_identity.py:34](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/launch_identity.py:34) requires live owner, launcher and worker identities with exact creation times and ordered creation. Direct workers must name the owner as parent. Distinct workers require Windows venv mode and the bound launcher as immediate parent. [local_runtime.py:171](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/local_runtime.py:171) restricts binding to the owner, once, and records actual worker identity and parent during the locked claim. No alternate interpreter or PID search was added. |
| Five-worker nonoverlap and owner loss | [local_runtime.py:98](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/local_runtime.py:98) maintains the shared reservation lock and five-slot cap. An unbound intent cannot be reclaimed merely because its owner dies. A bound unclaimed intent can retire only after launcher death/reuse; claim still requires that launcher to be live. Active workers remain protected until worker and launcher death are established, or the actual worker releases after environment close. [local_runtime.py:223](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/local_runtime.py:223) rejects a coordinator's release of a live/UNKNOWN actual worker. [run_wave.py:96](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/run_wave.py:96) continues waiting on returned launchers when release raises the expected liveness refusal. Existing one-thread configuration is retained. |
| Narrow explicit migration | [migrate_prefix.py:62](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/migrate_prefix.py:62) authenticates the fixed manifest, inventory and exact prefix hashes. [migrate_prefix.py:87](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/migrate_prefix.py:87) checks physical/runtime/contract identity, source pins, failed attempt, commands, claim-failure logs, old process death and absence of sessions for failed tokens. There are no CLI root/pin overrides or normal-collection legacy exceptions. |
| Preserve the two transitions and full boundary | [migrate_prefix.py:166](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/migrate_prefix.py:166) loads only after boot registers frozen classes, invokes existing checkpoint/policy/boundary validators, restores and compares observation/schema. It does not call reset, step or PFO. Restore works on copied environment state; the new migration code copies the checkpoint and changes only settings.identity. [migrate_prefix.py:183](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/migrate_prefix.py:183) checks recursive payload equality, including saved object attributes and array bytes; publication repeats the comparison after fresh deserialization. Actual frozen-object compatibility remains unexecuted. |
| Transactional publication, STOP and preservation | [migrate_prefix.py:244](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/migrate_prefix.py:244) holds old root/ownership/slot locks through read-only handles, rejects an existing uncommitted destination, and acquires destination locks before staging/publication. It retains original bytes and hashes, uses exclusive publication, checks STOP/source/input hashes again, and writes the commit marker last. [local_runtime.py:69](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/local_runtime.py:69) blocks missing/incomplete migration at both entry points and inside ownership. Partial publication remains diagnostic evidence and is not silently repaired. This is a guarded multi-file transaction, not a demonstrated power-loss guarantee. |
| Ordinary v2 resume without prefix recollection | [migrate_prefix.py:335](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/migrate_prefix.py:335) seeds only the two prior session reservations with the retained run ID/progress and adds the v2 plan. Failed intents stay in provenance/receipt with destination mappings. [run_wave.py:23](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/run_wave.py:23) selects --resume for carry155 from checkpoint existence; the other four carry slots have no blocking failed-intent history in the new cohort. Existing strict settings, run-ID and progress checks apply. The synthetic migration continuation asserts the next step starts at k=7 and produces transition 3 with the same run ID. |
| Experimental design and accounting | The exploration and collection/validation files are byte-identical. The migrated plan remains ten episodes, 750 transitions, 150 per scenario, maximum five workers and one thread each. The receipt distinguishes two reused transitions from 748 additional transitions. Existing session/timing files are copied byte-exact, retaining 66.75636630004738 seconds. Failed coordinator/startup costs are explicitly outside that timing scope and are not represented as zero. No new model, evaluation data or performance admission gate was added. |

## QUALITY Assessment And Actionable Limits

No new actionable correctness defect was established by this scoped review. The tests and the report support the intended division between process-only verification and synthetic collector/migration verification. The reported **236 passed in 78.46 seconds** and real redirector probe results are implementer evidence; they were not independently rerun here.

1. **Parent validation remains necessary for the real checkpoint.** The physical loader, frozen-class deserialization, saved solver caches, staged equality and restored observation must succeed on the pinned inputs during the parent's planned migration validation. Synthetic coverage cannot establish those results. Any failure must preserve v1 and incomplete v2 evidence; do not reset, drop or recollect the prefix.
2. **Nonblocking integration coverage gap.** [test_migration.py:95](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930_v2/test_migration.py:95) proves migration followed by direct `collect.run(..., resume=True)` at k=7. It does not run the migrated fixture through `run_wave.run(..., resume=True)` and the launch/claim path together. A focused synthetic integration test could assert one resumed carry command, four fresh carry commands, unchanged prefix/run ID and no sixth reservation. No suite rerun is requested by this review.
3. **Preserve the documented recovery limits.** UNKNOWN/unbound launches and incomplete staging intentionally block. Do not infer that every abandoned startup is automatically recoverable. Immediate committed-migration retry is hash-verified and write-free; after normal collection advances checkpoint/history, the migration command intentionally refuses the changed outputs. Continue with ordinary v2 --resume, not remigration.

The production handoff remains two explicit operations: `migrate_prefix.py --execute`, then `run_wave.py --resume`, using the existing `.venv-torch/Scripts/python.exe -B` and the v2 script paths. The migration command without --execute is an optional preflight that can boot/restore but does not publish. None of these commands was executed by this reviewer.

## Reviewed Production Source Hashes

| File in work/sdmpc_rl_local_20260930_v2 | SHA256 |
| --- | --- |
| launch_identity.py | 5cc1a94037b349072ff7ff37e9ceb756c3205b84cb8c1940b4305a5f1514e160 |
| local_runtime.py | 6312136f5fb5a8acfcf24483b57cd23165f9ea191c7ca364ba928270d50c87dd |
| migrate_prefix.py | 3934e15d8664603a1eb302b8a2e23581602aacba4e86fae37e08908562a66eb7 |
| run_wave.py | 7304669f4a5c65cb42237f8e79e0e6fa7bab766d5827e99d2169e33465dd6952 |

Later changes to these sources require a scoped follow-up review before relying on these verdicts.
