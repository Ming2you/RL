# Task 2 Independent SPEC + QUALITY Review

SPEC verdict: **NOT APPROVED**. The required nonterminal resume conflicts with the brief's explicit no-extra-preview constraint (finding 1).

QUALITY verdict: **CHANGES REQUESTED**. The same integration behavior is omitted by the synthetic restore and is not covered by the retained read-only parity evidence. No other actionable defect was established within this scope.

## Ordered Findings

1. **[P1] Resolve the extra reference preview on the required prefix resume before dispatch.**

   Changed call site: [worker.py:100](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_wave_20260930/worker.py:100). The planned one-interval 155 prefix checkpoints at `k=6`, after the next reference has already been prepared. On `--resume`, the worker invokes the frozen `BudgetEnv.restore`. That method creates a new `BudgetController` and calls `prepare_reference` for every nonterminal checkpoint ([budget_env.py:314](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/budget_env.py:314)). `prepare_reference` calls `lower.begin`, `lower.evaluate`, and `lower.execution_check` ([budget_controller.py:47](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/budget_controller.py:47)). The frozen lower implementation clears its cache in `begin` and evaluates the control on a cache miss ([prox_controller.py:145](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_externality_ablation_20260923/prox_controller.py:145)). Thus the already-prepared reference is previewed again on this mandatory resume path.

   This is a Task 2 integration-contract conflict, not a claim that Task 2 edited or introduced a bug in the frozen environment. Plant intervals and PFO solves are not repeated by this call. However, the brief also prohibits extra previews, with no stated restore exception. Preserving the raw checkpoint and checking exact observation equality do not establish compliance with that separate requirement.

   The restore also copies the checkpoint's `reference_timing` and `observation_timing`, then performs the reconstruction without updating those fields. The extra work is within the declared worker-session scope, so session totals are not shown to be false; it is not separately identified in the decision timing or preview accounting. The report's claim that the prepared reference is retained needs this qualification.

   **Targeted evidence:** executed only the unchanged `BudgetEnv.restore` and `BudgetController.__init__/prepare_reference` method bodies extracted with `ast`, using an in-memory `k=6` checkpoint and spy dependencies. Output was `calls=["lower.begin", "lower.evaluate", "lower.execution_check"]`, with saved `reference_wall_seconds=7` still unchanged after restore. The command used `.venv-torch/Scripts/python.exe -B -c`; no physical runtime was imported and no real reset, plant step, preview, PFO or optimization ran. The fresh real-preview implication follows from the pinned lower implementation cited above, not from running physics.

   **Why existing evidence misses it:** [conftest.py:126](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_return_wave_20260930/conftest.py:126) restores synthetic fields and returns an observation without executing the real restore/reference path. The retained physical import check constructs environments but never restores one.

   **Required resolution:** reconcile the explicit no-extra-preview requirement with the equally explicit unchanged-`BudgetEnv.restore` requirement before the actual prefix/wave. If inherited reference reconstruction is an intended exception, make that exception and its accounting explicit and add focused restore-effect coverage. Otherwise, a separately reviewed compliant restoration design is needed. This review does not authorize changes to frozen sources.

## Scope And Evidence

Read `task-2-brief.md`, `task-2-integration-notes.md`, `task-2-report.md`, and `task-2.diff`. All nine added Python files match the supplied diff and the final `a64504a6/evidence.json` source hashes. The three retained evidence-file hashes match the report. Existing suite and actor-parity results were consumed as evidence, not rerun.

The scoped review found no additional actionable issue in fixed actor/model admission, learner/physical import separation, seeds 7301..7305 and profile identity, exact retained actor actions, reward/terminal/inventory checks, STOP boundaries, kernel slot locking, completed-slot refusal, diagnostic tagging, or read-only completed-wave validation. Global budget-eight admission and actual-worker draining are explicitly parent responsibilities. Q remains absent and critic continuation remains labelled `carry`; unknown interrupted sessions and parallel wall time are not misrepresented.

The operation journal deliberately refuses uncertain in-flight reset/step recovery and preserves orphans. This documented conservative limit was not treated as permission to reset, repeat physics, or automatically promote an orphan.

Actual physical checkpoint serialization/resumption and traffic outcomes remain unverified, as the implementation report states. The actual wave root was absent when checked. Only this review file was written; no implementation fixes, suite/parity reruns, actual dispatch, or commit were performed.
