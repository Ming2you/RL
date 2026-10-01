# Scoped Fix2 Re-review

Reviewed 2026-09-30 against `fix2-brief.md`, the previous F1/F2 findings,
the round-2 section of `fix1-report.md`, and `fix2.diff`. Scope is only F1,
F2, and concrete Important regressions introduced by this diff. This is a
static source review; reported tests were not rerun.

**SPEC: PASS.** Both F1 and F2 are addressed within the requested scope.

**QUALITY: PASS.** No concrete new Important regression or remaining scoped
code blocker was identified in `fix2.diff`.

## Findings

### F1: Complete-config Serialization - ADDRESSED

[frozen_inventory.py:22](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/frozen_inventory.py:22)
supplies the dependencies for the exact frozen `to_plain_dict` definition,
selects its unchanged AST body from `historical_config.py`, and exposes it through
`complete_config` at line 36. Recursion resolves to that same definition in its
shared namespace. This preserves declared dataclass fields, non-private runtime
attributes, mappings, sequences and sets without importing the physical runtime.

[validate.py:44](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/validate.py:44)
now compares the simulator configuration using this complete representation.
The existing configuration hash check at line 35 remains unchanged, as does
settings serialization through the frozen runtime at
[collect.py:88](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:88).
The previous loss of `network.terminal_zero_gradient` is therefore removed.

The focused nested-dataclass regression at
[test_collection.py:668](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:668)
accepts the intact runtime extra, rejects a changed extra, and rejects changing
the settings representation without changing the pinned hash. Its assertions
cover the specific round-1 failure rather than substituting a dictionary config.

### F2: Retained Cohort Progress and Finished Identity - ADDRESSED

[local_runtime.py:153](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:153)
rejects a fresh episode whenever the slot has retained history, independently of
whether its output directory remains present. Resume requires a checkpoint and
the retained run ID. New reservations inherit that run ID, the furthest cohort
phase and the greatest recorded step count at lines 169-171. Thus releasing
process occupancy no longer erases started/checkpointed/finished identity.

[collect.py:95](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:95)
validates the saved checkpoint and reconciles its retained progress before
restore or another step. Each successful checkpoint write is followed by the
progress update at line 117. The checks at
[local_runtime.py:207](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:207)
reject run-ID changes and rewind, allow a valid newer checkpoint to catch up
after interruption between the two writes, and require step 75 for a finished
slot.

[collect.py:185](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/collect.py:185)
records `finished` in the worker's own finalization, after environment close and
before the elapsed-time sample. The implementation at
[local_runtime.py:219](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/local_runtime.py:219)
does not downgrade cohort phase on later process release. Standalone completion
and coordinator loss before post-exit release therefore retain finished identity
without an intervening ownership scan.

An interrupted final publication leaves the phase `finished` but occupancy
`exited`, permitting resume from the same validated terminal checkpoint.
The `env.k < 80` loop then performs no further interval. Session and timing
publication still precede `completion.json`, which remains the last write at
`collect.py:195`. Finished-ledger serialization is included in measured time;
the existing final timing-publication exclusions and UNKNOWN handling remain
coherent. Coordinator reservations pass the same resume mode at
[run_wave.py:58](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/run_wave.py:58).

Relevant regression evidence in the reported suite:

- [test_collection.py:690](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:690): immediate relocation after started, checkpointed, standalone-finished and child-finished states rejects fresh collection and preserves moved files. The former extra idle scan is also removed from the earlier regression.
- [test_collection.py:715](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:715): interrupted session-end or completion publication resumes without stepping, preserves run ID/session files, and distinguishes UNKNOWN from KNOWN timing.
- [test_collection.py:743](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:743): restoring a relocated prefix permits continuation; the following cases at line 761 reject rewind and unrelated identity before a step or checkpoint overwrite.
- [test_collection.py:779](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_local_20260930/test_collection.py:779): a nonzero finished-ledger write cost is measured and completion remains last.

## Evidence and Remaining Limits

Reported evidence is acknowledged: **79 collector PASS in 64.58 s, exit 0**;
**200 combined PASS in 66.13 s, exit 0**. No test was rerun for this review.
All six reviewed source files match the destination Git blob IDs in `fix2.diff`.
Diff SHA-256:
`084503f5a7251da22d30f9a580dbf15c6fbb8f473594fdc7b3c62a3315d2f025`.
Exploration and exploration-test hashes still match the previously reviewed
values.

No remaining actionable concern was found within F1/F2 or the new diff.
Physical-runtime compatibility and real concurrent process behavior remain
unexecuted; the reported evidence uses synthetic fixtures. Missing checkpoints
after a started attempt and unresolved launch identities intentionally remain
fail-closed. Pre-fix ledger migration is not provided, consistent with the
reported absence of production collection under this version.

The diff does not change the frozen physical contracts, exploration formula,
fixed cohort root, five one-thread numerical-worker cap, sequential episode
loop, carry-before-local ordering, STOP/ABORT checks, or training-only data
isolation. It adds no performance gate, model use, or evaluation-data use.
This scoped pass closes the two prior code findings; it is not evidence of a
physical smoke result, policy improvement, or collection completion.

Only this report was written. No source edits, production/collector/simulator
launches, model or trajectory loads, installs, commits, model-setting changes,
or changes to completed/checkpointed collection were performed.
