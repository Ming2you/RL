# Terminal quota ablation

## Purpose and scope

The terminal-only and projection audits have finished. Replay accounting is
consistent; fitting only terminal targets substantially reduces train error,
but ordinary fixed-policy TD updates made terminal error grow. Test ONE factor:
increase exposure to exact boundary conditions, without changing the Bellman
equation, representation, optimizer, actor, reward, or physical environment.

Implement ONLY `terminal_quota.py` and `test_terminal_quota.py` in this directory.
Write the implementation report to
`.superpowers/sdd/rl_budget_value_audit_20260930/terminal-quota-report.md`.
Use apply_patch; preserve every old file. No commits or production runs.
Synthetic tests may use one CPU thread with the local runtime. No new simulator
episodes, evaluation inputs, policy exports, installations or unrelated changes.

## Fixed production experiment

- Load the known authenticated base using terminal_fit/common.load_base.
- Consume only its existing 750 training transitions. Validate through the
  existing terminal extractor and critic_diagnostic.replay_tensors, then additionally
  require the successful projection audit completion, its exact base hash and
  its 73-file input manifest to match before and after. Pin that completion
  and all imported diagnostic source files before load and verify after.
- Seeds 6529 and 6530, run sequentially, one CPU numerical worker; 3750 updates
  per arm per seed, log at 0, 375, 750, 1500, 3750. CLI accepts only --output;
  shorter configurable budgets may exist solely in internal synthetic tests.
- Two arms `uniform` and `terminal_quota`. Both start as independent full
  copies of the final original TD3 learner (including critic Adam and targets).
  Actor AND actor_target are frozen copies of the final online actor. Do not
  update either, or its optimizer. Critic target tau=.005 every second update.
- Every minibatch has 8 samples from each of five scenarios, total 40.
  Draw the uniform arm's 8 indices per scenario from all 150 with replacement.
  For the quota arm, copy these indices but replace the first entry in each
  scenario by a uniform draw from that scenario's two terminal indices 74/149.
  The remaining 7 entries stay identical, including incidental terminals.
  Thus quota means at least one terminal per scenario, NOT exactly one.
  Use a separate local RNG for replacement so uniform draws are not perturbed.
  Share the exact smoothing-noise tensor between the two arms at each update.
- Reuse read-only helpers from recovery/critic_diagnostic.py where appropriate
  (target_values, critic_step, diagnostics). No code duplication of the old TD3
  trainer or physical simulator. Make finite parameter/gradient checks around
  steps so nonfinite runs cannot complete. Retain gamma=1 and true-terminal mask.
- Save JSON diagnostic curves and metadata only, no learned weights. Metadata:
  checkpoint/source/runtime/input hashes, seed, initial state identity, sample
  counts, actual terminal counts per scenario, index/noise stream hashes,
  initial/final frozen actor+target+actor optimizer and original learner/model
  fingerprints, counts of online/target critic updates, elapsed time and PID.
- Each log includes original diagnostics (terminal absolute errors, observed
  behavior-return discrepancy and TD residual by scenario and time slice).
  Add terminal-only per-episode/per-scenario values sufficient to inspect each
  of the ten exact targets. Evaluation noise must use a separate fixed stream
  and be identical for both arms. Never use metrics to stop/select/export.
- Input and original learner/model/global RNG state must remain unchanged;
  use terminal_fit helpers for fingerprints and RNG preservation if helpful.
- STOP at repo, goal root and output: check before any load/write except
  acquiring the lock, inside the exclusive output lock, between steps, and
  before completion. A stopped run is not completed. Refuse nonempty output
  both before and inside the lock (allow only the lock file). Use existing
  atomic JSON writer with allow_nan=False validation. Failure must not publish
  a completion; no resume/overwrite mechanism is needed for this short run.

## Falsifiable interpretation, predeclared

Hypothesis: sparse boundary samples contribute to terminal drift during fixed
policy TD. Evidence supporting the sampling change requires final min-Q terminal
MAE lower than uniform in BOTH seeds, at least 50% lower pooled MAE than uniform,
and final MAE below the shared initial 5.701837813854217 reward units in BOTH
seeds. This is a diagnostic criterion, NOT policy admission or a traffic claim.
If unmet, do not keep increasing this same update budget. Full curves, each
critic and each terminal must remain available even if the criterion passes.

Observed behavior returns are NOT current fixed-policy Q ground truth. Exact
terminal rewards are ground truth only for those stored state-actions. The
actor remains the old saturated actor; these outcomes cannot justify deploying
it. No holdout or generalization claim is possible from this ablation.

## Tests

Use tiny synthetic CPU replay and stubbed I/O, never actual production data.
Verify equal scenario counts, terminal replacement and paired other seven,
independent RNGs/reproducibility, identical noise, original gamma/terminal mask,
critic-only update/target cadence, actor/optimizer/model/replay immutability,
per-terminal metrics, STOP including lock races, nonempty output, source/input
change rejection, nonfinite rejection, no weights or completion on failure,
and fixed CLI budget. Run focused tests, report exact command and results.

Runtime: `.venv-torch/Scripts/python.exe` plus `.deps-budget` for pytest.
ACL escalation may be necessary. Parent handles production execution only after
independent review. Shared working tree is dirty: do not stage or revert files.
