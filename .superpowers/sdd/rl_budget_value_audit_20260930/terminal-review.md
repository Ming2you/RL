# Bounded Terminal-Fit Review

Date: 2026-09-30

SPEC: PASS

QUALITY: PASS

Actionable scoped findings: none. No correctness or specification defect was
identified in the supplied terminal-fit diff. These verdicts cover the bounded
diagnostic implementation, not a measured fit outcome or policy admission.

## Scope and Evidence

Reviewed the supplied `terminal-review.diff` against
`work/sdmpc_rl_value_audit_20260930/terminal-brief.md` and `terminal-report.md`.
Both added files match the supplied diff line for line: `terminal_fit.py`
(433 lines) and `test_terminal_fit.py` (479 lines).

Accepted the implementer's evidence of **46 synthetic tests passing in 4.66s**,
exit code 0, as instructed. The test source was inspected; tests were not rerun.
Read the existing recovery `common.py` and the relevant TD3, atomic-write,
locking, provenance, and snapshot-verification helpers to check their contracts.
No production model or evaluation data was loaded, no diagnostic or training
was executed, and no source files or commits were changed. This review writes
only this ledger file. Projection auditing remains with the parent.

## Specification Assessment

Source: [terminal_fit.py](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_value_audit_20260930/terminal_fit.py)

- **Terminal extraction and fold separation, lines 111-148:** validates the
  exact scenario set, 150-row CPU float32 replay shapes, finite values, boolean
  termination flags, and terminal positions exactly 74 and 149. Copies only
  observation/action inputs and their stored immediate rewards. The two folds
  reverse collection rounds and each split contains exactly one row per scenario.
- **Arms and objective, lines 151-189 and 254-300:** independently copies each
  original/fresh critic template into each fold, enables all critic parameters,
  clears copied gradients, and constructs a fresh Adam at 0.0003. Fresh critics
  come from the same TD3 dimensions and hidden size with seed 8100. The fixed
  full batch has five rows; each critic's mean squared error gives every scenario
  20% weight, and the two losses are summed. No bootstrap, actor/target update,
  noise, reward transformation, or augmentation enters this path.
- **Budget and reporting, lines 32-37, 192-212, 254-302, and 422-429:** the CLI
  exposes only the output argument and uses 1000 updates per arm per fold, with
  metrics at 0, 50, 250, and 1000. Original-critic errors on all ten terminals are
  recorded before fitting. Checkpoints include Q1, Q2, min-Q, targets, signed and
  absolute errors, MSE/MAE/max error, scenario/round/row identifiers, and training
  draws. Final per-critic training thresholds are descriptive and do not gate
  completion. Holdout values do not select arms, change updates, or stop fitting
  based on accuracy; nonfinite results correctly fail the run.
- **Interpretation, lines 39-50 and 277-280:** explicitly discloses original
  critics' prior exposure to both folds during TD3 training. Fresh holdout is
  excluded from that fold's updates, and the one-sample-per-scenario limitation
  and absence of a generalization or policy-admission claim are stated.

## Quality and Lifecycle Assessment

- **Preservation, lines 59-100, 151-161, and 348-400:** fitting uses detached data
  copies and independent critic/optimizer objects. The imported TD3 architecture
  is CPU float32 Linear/ReLU, so diagnostic forward passes have no running-state
  or stochastic-layer side effects. Fingerprints cover original networks,
  optimizers, replay, counters, learner RNG, gradients, trainability, and modes;
  the deserialized learner state is also checked. Global Python, NumPy, and CPU
  Torch RNG restoration encloses loading and fitting, including exception paths.
- **Provenance, lines 225-251 and 354-387:** checks the fixed model hash, baseline
  completion identity, sidecar/common/brief sources, baseline source pins and
  frozen snapshot, runtime versions, and runtime entrypoint hashes before and
  after execution. The existing loader additionally binds the model's saved
  source/runtime contract. Verification failures prevent completion. Runtime
  entrypoint hashes are appropriately described as partial runtime identities.
- **Output and STOP lifecycle, lines 305-419:** exclusively creates a new output
  directory and holds the existing OS lock through final verification and writes.
  Existing outputs are rejected, including empty directories. Entry STOP returns
  without output writes; output, goal-root, and repository STOP locations are
  checked before loading, at fit boundaries, around every update, and before
  completion. Observed STOP preserves the marker, records stopped status and
  actual partial progress, and prevents a completion artifact.
- **Artifacts and finite guards, lines 53-56, 164-211, and 337-416:** all diagnostic
  data artifacts use the existing temporary-file/replace JSON writer, preceded
  by strict finite JSON validation. Predictions, loss, gradients, updated
  parameters, and Adam state have finite checks. No fitted model, optimizer,
  pickle, or policy export is written; `runner.lock` is the lock marker.

The inspected synthetic suite addresses extraction, malformed inputs, exact
rewards and metrics, fold balance/separation, fresh initialization, all-layer
training, preservation, holdout independence, STOP timing, overwrite refusal,
lock contention, source/model drift, finite guards, and the fixed CLI contract.
No additional test execution is requested by this review.

## Limits

Production loading compatibility, the actual 1000-update numerical trajectory,
whether each fit reaches 0.01 reward units, and measured production immutability
remain unverified by design. The accepted synthetic evidence does not establish
those outcomes. The existing lock helper is Windows-specific. No conclusion is
made about the parent's projection audit or traffic-control performance.

## Reviewed Identities

SHA-256 of the supplied diff:
`e2f3d5269f73218d373c171426eaee2b3ae4ce676844e4daec2bd5151ccece6b`

SHA-256 of `terminal_fit.py`:
`07251732bd66b27c5315f580ab4de1f79bdd4f1943a1ec5c33bdd5e821443ba2`

SHA-256 of `test_terminal_fit.py`:
`01acf86d0b09ccd91bbb2017b76b2d1767e45b72f4d9663005f6361667d634b3`
