# Multi-scenario runner integration tests

Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL.
Own only new work/sdmpc_rl_multi_20260929/test_multi_contracts.py,
test_multi_pilot.py, test_multi_runner.py as needed, plus task-2-test-report.md
in this brief's directory. Do not edit production or other tests. No commits,
agents, traffic simulation, package installs or old experiment writes.

Read new run_budget.py, train_round.py, compare_runs.py, run_pilot.py and
build_preflight.py. The coordinator is implementing these while you write
synthetic focused tests. Existing fake-runtime examples live in old
work/sdmpc_rl_carry_20260929/test_carry_{environment,runner,run_contracts,pilot}.py;
read them for patterns, do not edit old code. Do not import same-named old
production modules into the new test process.

Binding requirements:
- Exactly five scenarios as exported from td3.SCENARIOS. Single shared learner;
  collectors never update a policy. Canonical center/RL full runs use no
  exploration/training seed. Collection tagged by scenario/round/seed/model.
- Two rounds:6301..6305 then6401..6405;75transitions each; each central round
  gets five distinct completed collectors with identical shared contract/schema,
  hashes and correct predecessor model. Replay retains true terminal only.
- Actor collector round0first32steps uniform thenGaussian.3; round1 frozen
  shared actor +Gaussian.3. No local updates. Requested action/reward/next_obs
  persisted exactly; adjacent observations must connect, episodes must not join.
- Completed trace75steps/14400sec, correct terminal timeline and TTTsum, areaTTT,
  PFO/reference/guard/decision time, physical+budgetvalidity. H3guarddisabled.
- Inference/collection checkpoint resumes environment+exploration RNG and reloads
  immutable frozen policy; source/runtime/scenario/profile/model drift rejected.
- Completed collection is skipped only after full validation; reject malformed
  completion BEFORE launching a sibling. Existing active output locks prevent
  duplicate/orphan overlap. At most five numerical children; all-one-core, learner
  alone. STOP at parent/stage/child must prevent new work and drain active workers.
- Failures/partial collection must not be promoted to training/evaluation success;
  paused child must stop stages. Exact settings/provenance/counters on trainer
  resume. No evaluation data admitted to replay.
- Gate needs five distinct actual smoke scenario results, matching schema/source,
  physical carry evidence and passing tests/review hashes.

Write meaningful tests of the new behavior and failure paths, including actual
shared learner update from five synthetic experience groups where inexpensive.
Do not merely assert constants. Include controlled fake subprocess lifecycle
tests rather than starting traffic children. If production fails an intended
requirement, retain failing regression and report exact failure; do not weaken
test or patch production. Run only your synthetic tests in one process with
.venv-torch/Scripts/python.exe and .deps-budget; may require sandbox escalation
for dependency ACL. Record command/output and concise exact defects in report.
