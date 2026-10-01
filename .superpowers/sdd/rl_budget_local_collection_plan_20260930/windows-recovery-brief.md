# Windows launcher recovery implementation

Read this first. This is a NEW bounded runtime repair after real smoke admission,
not a rerun of R1-R5. Preserve work/sdmpc_rl_local_20260930 and all existing result
files byte-for-byte. No production collection or migration is authorized for the
subagent; parent will execute after independent review. No commits/install/OS
changes or model overrides. Use apply_patch for edits; mechanical copies allowed.

## Reproduced root cause

Two actual standalone carry155 intervals succeeded (separate sessions, same
episode) under the reviewed v1 code. PID28800 then45436, exit0. Checkpoint has
2 transitions, no terminal. Hash before failed wave:
659f936702be59880760677d76a8ccc35957a03aa881a3fe85a9fb4ea1923816.
settings hash933269729875452e776be948fc8a66b8af15423f6b63e732c6fcfb6328ee5571.
Run ID2962784e595d4ee8b1272dda7762dbc6. Measured locked worker sessions66.7563663s.

The first coordinator launched at2026-09-30T04:31:19KST, launcherPID37600,
real coordinatorPID27044, attemptbae7788116f247999100f4ad6a2d3d19. All5children
failed at claim_worker with 'Launch reservation process identity differs',
BEFORE session creation/boot/restore/reset/step. All Python processes exited.
Logs, reservations and immutable plan are under
results/sdmpc_rl_balanced_goal_20260930/local_budget_v1.

Minimal nonnumerical reproduction using .venv-torch/Scripts/python.exe:
Popen([sys.executable,'-B','-c','print(os.getpid(),os.getppid())']) returns
launcherPID35620 but childactualPID3048,parent35620. Windows venv redirector.
sys.executable=REPO/.venv-torch/Scripts/python.exe;
sys._base_executable=C:/Users/alsrj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe.
Do not change interpreter/dependencies to bypass identity checks.

## Versioned repair

Mechanically copy reviewed local implementation/tests to
work/sdmpc_rl_local_20260930_v2. Change fixed COHORT_ROOT to
results/sdmpc_rl_balanced_goal_20260930/local_budget_v2. Preserve the exploration
formula exactly. Fix launcher/actual worker ownership: distinct identity records,
token claim-once, verify expected direct parent/launcher and PID creation time
for the Windows redirector, retain direct-process support, unknown/live fail-
closed, no extra numerical slots/duplicate work. Avoid broad process discovery
or treating an arbitrary different PID as valid. Record actual worker identity
as well as launcher so startup/coordinator-loss recovery is coherent. Include a
real tiny subprocess test (no NumPy/physics) for venv redirector identity and
synthetic wrong-parent/PID-reuse/startup-window/worker-cap tests.

## Explicit one-time carry155 migration

Implement a narrow, dry-run-able migration into the v2 cohort; do not execute it
on actual results. It must verify no old/new active numerical run, STOP absent,
old pinned source/runtime/contract, failed attempt/error evidence, exact prefix
checkpoint/settings hashes above, per-transition and physical checkpoint
validation. Boot/restore MAY be required only when the parent later executes
migration to deserialize frozen classes; never reset or step and never call PFO.
All v1 files stay unchanged. No broad legacy-format acceptance in normal runs.

Copy the two durable transitions, full physical boundary, policy/RNG and prior
session records into the new canonical carry155 slot, retaining its run ID.
Use the v2 identity in a NEW settings/checkpoint and retain byte-exact original
inputs in provenance with hashes; record old->new identity/outputs and a precise
physical-contract/transition equality check. Seed a compatible v2 cohort history
so normal --resume works without repeating those intervals. Preserve timing
sessions and disclose failed coordinator/startup costs outside that defined
scope. Do not rewrite old settings/plan or claim these are 2 new transitions.

Other4carry slots have only failed unclaimed launch intents, no session start,
settings or checkpoint. New v2 slots may start their first physical episodes
after authenticating this evidence, retaining a mapping to those old intents.
There is no completed training trajectory to repeat. Local behavior has not run.
Total data goal stays750transitions (2 reused +748 additional),150per scenario.

Migration must be transactional/fail-closed and idempotent (recognize and verify
already committed migration; no overwrite). Stage only inside new intended root;
preserve incomplete staging for diagnosis. Do not invent a general migration
framework. Test small synthetic old roots and moved/corrupted/active/stopped/
partially published migration cases; parent will validate actual migration after
review. If any part cannot be safely implemented, report the concrete issue
instead of weakening checks or dropping the prefix.

## Report and review

Own only new v2 modules/tests and windows-recovery-report.md in this ledger
directory. Use existing reviewed helpers read-only. Run inherited local200tests
against v2 plus covering new tests with existing Python/deps. No old phase suite
reruns. Report exact commands/results, actual process-only reproduction, source
preservation evidence and limitations. Parent will package v2-v1 diff and request
scoped independent review before executing migration or numerical collection.
