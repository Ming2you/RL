# Frozen S-DMPC Budget Wrapper Review

Reviewed 2026-09-29. Scope: the supplied wrapper diff, both design/execution documents, and the frozen baseline's physical-budget, solver, PFO, bootstrap, and state contracts. Learner implementation and the parent's unfinished runner are outside this review.

**SPEC: NEEDS_CHANGES. QUALITY: NEEDS_CHANGES.** Three P2 findings below prevent complete signoff. No P0/P1 execution blocker was identified: the original physical/budget/PFO guard and selected-price behavior are preserved by the inspected paths.

**Pilot blocker verdict:** no confirmed critical wrapper issue requires holding a bounded exploratory pilot once the action-influence gate completes, provided it uses the verified snapshot, unchanged environment/reward contract, and the declared partial-observability limitation. The findings concern acceptance evidence and reproducibility; they do not demonstrate unsafe control execution. This is not end-to-end runner/learner approval or full-run acceptance.

Parent update received before finalization: `run_budget.py` and `compare_runs.py` now exist; the runner labels Markov sufficiency unproven, and the parent explicitly acknowledges the POMDP interpretation and further validation requirement. This addresses the observation-disclosure concern as a standalone blocker. Those new files were not in the supplied diff and were not reviewed; whether they already enforce findings 2/3 externally remains outside this verdict. Parent-reported probes at steps 10/30/50 show noncentral control and tail differences; the last step-5 candidate was still running. No executing wrapper/env files were edited.

## Findings

### 1. [P2] Preserve candidate failures and solver diagnostics before preparing the next interval

Location: [budget_controller.py:146](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_controller.py:146), audit construction through line 154; [budget_env.py:187](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:187), next-reference preparation before returning.

The returned audit exports the selected execution check and budget list, but no per-candidate status/reason, achieved budget, residual, predicted TTT, stationarity, local-QP diagnostics, or inner price history. On fallback, the selected check describes the successful PFO budget, not the failed lower response. `step()` then calls `prepare()`, whose `prepare_reference()` resets `self.results` at `budget_controller.py:63`; the lower `begin()` also resets diagnostic caches/archives. A caller cannot recover the completed decision's missing evidence after a normal nonterminal `step()` returns.

This loses required evidence for distinguishing physical invalidity from algorithmic failure and for evaluating budget influence and convergence separately. The baseline explicitly persists `candidates`, derivatives, and execution/anchor audits in `artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_externality_ablation_20260923/run_scenario.py:107`. The design requires these diagnostics in section 8.

An in-memory synthetic fallback check confirmed that requested `[-50,1000]` and executed `[0,1000]` remain separate and incoming prices are retained, but the candidate's failure status, stationarity, and local rows are absent from the audit. The existing probe JSON trace schema independently shows the omission. This is a reporting defect, not evidence that fallback executes the failed request.

Required before accepting action-influence/solver-diagnostic results: export a detached, compact candidate audit before the next preparation clears it, including rejected-response checks and reasons, relevant solver diagnostics, and the identity of an archive-selected point. Preserve the executed-control audit separately. Full rollout objects need not be serialized.

### 2. [P2] Reject checkpoints from a different environment/reward contract

Location: [budget_env.py:193](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:193), checkpoint contents through line 203; [budget_env.py:205](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:205), restore validation and object reconstruction through line 219.

Restore checks only the forecast hash. The saved simulator, warm controller, and observer retain their saved configuration, while the lower controller is rebuilt from the receiving environment's runtime. `reward_scale` is neither saved nor validated. Thus restoring an otherwise identical checkpoint into `BudgetEnv(..., reward_scale=200)` after collection with scale 100 silently changes subsequent rewards. Changed lower options/configuration can also produce a saved plant paired with a different controller.

A synthetic terminal-checkpoint restore, using the real checkpoint/restore methods without a plant or lower solve, accepted a matching profile hash with a saved simulator configuration, a different receiving lower configuration, and reward scale 200. The profile-only admission check is the same for nonterminal checkpoints. The reported same-environment observation parity does not cover this case.

Required before durable resume across environment construction/processes: save and validate the reward scale and a contract identity covering configuration, lower options, source snapshot, and observation names/scales before replacing live state. An outer runner can enforce the same contract explicitly. This is not a demonstrated defect in the parent's current same-object branch restores.

### 3. [P2] Connect frozen-source verification to the execution path

Location: [budget_runtime.py:43](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_runtime.py:43), bootstrap through line 78; [budget_runtime.py:87](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_runtime.py:87), protocol-input verification.

`boot()` imports code from the supplied root and checks selected module locations and H3/six-iteration options, but never verifies the snapshot manifest or source hashes. `protocol()` verifies five protocol input files; those checks do not cover controller/plant source, bootstrap solver options, or the manifest itself. A source file edited in place still passes the module-location checks. Neither supplied entry point, `check_wrapper.py:26` nor `probe_budget.py:40`, invokes the available offline snapshot verifier.

The artifact itself is intact: independent read-only verification passed for all 151 files. This finding concerns enforcement during future use, not an observed mutation. The original runner hashes source/input files and checks them during and after execution (`run_scenario.py:90`, 127, 150), and the snapshot report explicitly requires consumers to verify before/after use.

Required for a frozen-run acceptance claim: verify the manifest payload before importing/using it, retain its expected identity in run metadata, and verify it again on completion/failure. The parent's runner may own this once per process/run; it need not be repeated for every candidate. Preserve the snapshot payload unchanged.

## Declared Partial Observability: Nonblocking

Location: [budget_env.py:69](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:69), controls and memory features through line 84; [budget_env.py:146](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_budget_20260929/budget_env.py:146), reuse of the persistent warm controller.

The observation includes traffic state, forecasts, mapped PFO controls/budget/TTT, and lower prices, but never reads `env.warm` state. The frozen PFO follower carries `_prev_coupling`: `artifacts/sdmpc_budget_baseline_20260929/source/work/sdmpc_matrix_14400_20260912/historical_tree/src/controllers/wu_faithful_follower.py:3756` seeds the next solve from it, and line 4340 updates it. The finite-iteration follower and subsequent control mapping do not establish that exposed PFO output uniquely determines that memory.

An in-memory encoding check changed only `_prev_coupling` and obtained an identical observation. The dependency of the next PFO solve on that memory is established by source inspection; no traffic rollout was run to quantify its effect. Saving the full warm object correctly preserves continuation state, but does not make it visible to the actor.

The parent's explicit disclosure is consistent with the design's allowed POMDP path (`docs/rl_budget_leader_design_20260929.md:81`), so this is not a standalone implementation blocker. Keep the limitation attached to pilot results and validate whether history/recurrent input is needed before claiming observation sufficiency or broader policy generalization. Exposing normalized coupling state is another possible remedy, not a required change during the running probes. This review does not establish that the current MLP will fail.

## Contracts That Passed Inspection

| Area | Assessment and evidence |
| --- | --- |
| Physical budget | The wrapper uses the frozen evaluator's budget vector. `player_sensitivity_dmpc.py:417` sums inbound minus outbound service across H3; line 445 uses the metering-command sum. NP remains signed. These are veh and veh/h respectively, not stock and realized ramp throughput. |
| Residual action | `budget_controller.py:7` validates finite shape/range, adds `[50,1000]`, and clips only NUF. Zero action requests the mapped PFO budget. The parent parity checks exercise this. |
| Original guard | Reference and selected lower points use the inherited original execution check. Upper policy has zero band margin (`fixed_policy.py:11`, 23, 38). `budget_controller.py:124` requires feasibility/physical validity and rejects worse H3 TTT. Fallback preserves requested budgets separately and is uncertified. |
| Candidate prices | `budget_controller.py:75` copies request, seed, and incoming prices; the `finally` block restores persistent dual/budget on exceptions. Only selection commits prices. PFO/archive recovery retains incoming prices. No candidate-price contamination was found by inspection. |
| Shared caches | Per-interval evaluation/derivative caches are keyed by control point under fixed state/forecast/previous control. Candidate-dependent width/model references are reset by the inherited solve. Shared caches are not themselves evidence of price contamination. |
| Lower semantics | The wrapper calls the inherited solver; nine players, four freeway groups, common Jacobi prices, k-to-k+1 updates, quantization, and original rollout gates remain in the frozen implementation. `converged=False` is retained. Native candidate generation matches the original selected-dual policy. |
| Reward/replay boundary | `budget_env.py:159` takes the actual plant TTT difference and checks regional accounting. Line 175 preserves the submitted RL action independently of fallback. Warmup TTT is retained; no horizon-overlap reward, extra queue penalty, clipping, or terminal-clearance cost is introduced. Actual replay insertion belongs to the learner/runner. |
| Episode/failure | Five warmup intervals precede 75 decisions; only k=80 is terminal and 14,400 seconds is checked. Invalid PFO raises rather than returning a zero-cost successful terminal. Chunk truncation and structured failure persistence remain runner responsibilities. |
| Same-contract checkpoint | Simulator counters/logs, plant buffers/time, previous control, dual/last budget, observation memory, fixed profile identity, full PFO object, and prepared forecast/reference are retained. Restore avoids solving PFO twice. FrozenProfile has no live random generator to advance; training perturbations are generated once and hashed. |
| Traffic observation | All numeric TrafficState leaves are included, both delayed buffers have fixed physical time bins, all DemandStep fields are represented, and names/finite values are checked. Directional demand skew is available from per-link demands; storage occupancy/receiving space is derivable from storage and fixed capacities. No additional future information beyond the controller's H3 forecast is introduced. Hidden PFO memory remains the explicitly declared limitation above. |

Baseline filenames in this table resolve under `artifacts/sdmpc_budget_baseline_20260929/source/`: `player_sensitivity_dmpc.py` is in `work/sdmpc_matrix_14400_20260912/historical_tree/src/controllers/`; `fixed_policy.py` is in `work/sdmpc_externality_ablation_20260923/`.

## Timing and Nonblocking Follow-Ups

- Timing scopes for PFO, reference evaluation, lower solve, and guard are disjoint; `budget_env.py:182` does not sum nested rollout counters into decision time. The next prepared reference's timings are associated with its next decision. No double-counting defect was found here.
- `decision_wall_seconds` is a sum of measured decision components, not measured end-to-end environment/runner time. Forecast retrieval and observation preprocessing are outside those scopes; actor CPU and total decision CPU are not exported. The parent should measure these when publishing the required wall/CPU and p50/p95/max comparisons. Plant execution and training/checkpoint overhead should have separately identified scopes. This is an acceptance-reporting obligation, not a reason to rerun expensive parity simulations now.
- The saved step-30 parity times (37.33s baseline, 59.35s wrapper) and step-5 times (60.66s, 59.11s) are one-shot parity measurements. `check_wrapper.py:34` computes PFO before both timing blocks. They do not establish end-to-end speedup or regression, particularly with concurrent parent probes.
- Useful inexpensive regression additions after fixes: real audit preservation using fake lower responses; successful/rejected/failing candidate order isolation; nonzero-dual fallback; same-forecast/different-reward checkpoint refusal; observation schema identity; terminal versus runner truncation. The three controller tests currently cover mapping, invalid action, and exception rollback only. Missing test cases are a coverage limitation, not proof of another implementation defect.
- A published observation schema/version and less brittle runtime import names would improve maintainability. They are optional beyond the contract validation needed in finding 2. No general refactor or baseline edit is recommended.

## Evidence and Limits

- Read both requested documents and the supplied diff. Current `budget_controller.py`, `budget_env.py`, `budget_runtime.py`, `freeze_runtime.py`, and their two test files matched the added-file contents in that diff. Diff SHA256: `5df4e3cb8d75f1646b541cd1e2c2150f798e3fb88f1c0741e1e4fca5a36dfa22`.
- Inspected the saved step-5/30 parity PASS artifacts. Accepted the parent's reset/checkpoint observation parity, three controller tests, and 18 snapshot tests as reported evidence; did not rerun traffic simulations or those suites. Read their test/check implementations to establish coverage.
- Independently ran `.venv-torch/Scripts/python.exe -B work/sdmpc_rl_budget_20260929/freeze_runtime.py --verify-only`: PASS, 151 payload files, 3,975,269 bytes, no unmanifested files. Manifest SHA256: `07836ff139e2e2f9bf8db549d30b5dfba0e4f8c7478165315feef84ab5ed0005`. Dirty source provenance is recorded, not reduced to a Git revision alone.
- Ran only short in-memory synthetic checks of audit export, observation memory visibility, and checkpoint admission. These imported the new wrapper with `-B`, did not boot the historical runtime, did not run a traffic simulation, and created no test/output files.
- Read-only review except this report. No implementation, baseline, learner, existing result, or Git state was modified; no commits were requested or performed. Full-run parity, resumed-transition parity, action-effect acceptance, and training performance are not established by this review.
