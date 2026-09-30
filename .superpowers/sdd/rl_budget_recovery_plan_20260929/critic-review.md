# Independent critic diagnostic review

Date: 2026-09-29

## Findings and verdict

No actionable defects found in the requested scope. **SPEC: PASS. QUALITY: PASS.**
No implementation changes are requested by this review. These verdicts concern
the paired critic-only diagnostic implementation, not its numerical outcome or
policy performance.

## Scope and evidence

Read `critic-brief.md`, `critic-report.md`, and `critic-review.diff`. Both current
files match the supplied diff line for line: 293 diagnostic lines and 272 test
lines. References below to `critic_diagnostic.py` and `test_critic_diagnostic.py`
are under `work/sdmpc_rl_recovery_20260929/`.

Read only the relevant constructor, constants, and checkpoint-loading contract
in `work/sdmpc_rl_multi_20260929/td3.py` to establish the diagnostic's dependency
behavior. Unrelated actor/evaluator work and the prior implementation as a whole
were outside this review.

## SPEC checks

| Requirement | Assessment and concrete evidence |
| --- | --- |
| Frozen input and clone isolation | PASS. `critic_diagnostic.py:48` verifies the pinned hash before deserializing those same bytes on CPU. `:57` constructs both learners from the same state; `:191` copies the caller's state. The dependency's `td3.py:371` copies network weights and `:374` independently deep-copies both optimizer states. Clone equality and independent critic/Adam storage are checked by `test_critic_diagnostic.py:53`. |
| Fixed final actor in both arms | PASS. `critic_diagnostic.py:62` replaces both target actors with the final online actor and freezes both actor modules. Only the critic optimizer steps at `:125`; actors are checked against the original final actor at `:258`. The actor, target actor, actor optimizer, replay, and supplied state are checked for immutability by `test_critic_diagnostic.py:128`. |
| True terminals and replay boundaries | PASS. `critic_diagnostic.py:79` requires five scenarios, 150 transitions each, terminals at 74 and 149, and within-episode continuity. `:115` masks continuation only at true terminals; `:69` resets behavior reward-to-go at those boundaries. Tests at `test_critic_diagnostic.py:71`, `:158`, and `:164` cover target values, both return boundaries, and invalid replay layouts. |
| Equal samples and shared smoothing noise | PASS. `critic_diagnostic.py:100` draws exactly eight samples with replacement per scenario, forming a 40-transition batch. `:105` draws clipped float32 CPU noise once; `:247` passes the same batch/noise to both arms. The inherited constants are gamma 1, noise standard deviation 0.2, and noise clip 0.5; actions are clipped to [-1, 1] at `:112`. Tests at `test_critic_diagnostic.py:86`, `:98`, and `:181` cover balance, smoothing, and identical paired inputs. |
| Target rate is the single changed factor | PASS. `critic_diagnostic.py:21` defines tau 0.005 and 1.0. Both arms retain identical initial online critics, target critics, and optimizer history. `:128` updates critic targets after every second additional diagnostic update in both arms, without an update-zero synchronization. No actor or actor-target training update is called. Exact target timing/rates are checked by `test_critic_diagnostic.py:146`. |
| Budget and logging | PASS. `critic_diagnostic.py:19` fixes 3750 additional updates and logs at 0, 375, 750, 1500, and 3750. `:136` reports total twin-critic TD MSE, per-scenario terminal absolute errors and behavior-return gaps, episode metrics, and three time slices. `:228` adds counts, target-update counts, paired-draw digest, and training losses. Counts are per arm: completion means 150,000 sampled transitions, 30,000 per scenario, and 1,875 target updates for each arm. |
| Reproducibility and runtime | PASS. `critic_diagnostic.py:186` sets one intra-op and inter-op CPU thread. `:194` uses explicit training RNGs; `:197` gives fixed evaluation noise a separate stream. `:200` records source hashes, runtime/package paths and versions, seeds, parameters, and the original specification. `test_critic_diagnostic.py:244` checks that extra logging leaves training draws and final diagnostics unchanged. |
| Overwrite refusal and STOP | PASS. `critic_diagnostic.py:190` refuses an existing output directory and `:223`/`:272` exclusively create files. `:226` flushes every record. `:216` defines STOP locations, including an optional explicit path; `:243` checks before each paired batch. Stopping writes the current partial checkpoint when needed and a stopped summary before returning normally. Tests at `test_critic_diagnostic.py:215` and `:237` cover partial and preexisting STOP plus overwrite refusal. |
| Input/source integrity and restricted effects | PASS. `critic_diagnostic.py:262` rechecks source and input hashes before successful or stopped completion. The entry point at `:277` loads existing replay and invokes only this diagnostic. Persistent outputs are diagnostic JSONL and summary JSON; there is no rollout, collection, policy export, traffic interaction, or scheduling path. `test_critic_diagnostic.py:253` covers the input hash pin and unchanged file bytes. |
| No policy overclaim | PASS. `critic_diagnostic.py:22` explicitly says behavior returns are not current-policy Q truth and that this diagnostic does not establish causal policy superiority. The caveat is recorded in metadata at `:215` and the summary at `:270`. The implementer report also distinguishes fixed-noise TD MSE from an expectation over smoothing noise and makes no traffic-benefit claim. |

## QUALITY checks

The standalone loop is small and directly expresses the requested experiment.
It reuses the existing networks/loading/optimizers without introducing a new
framework. Optimizer history is preserved without shared mutable state, and
logging cannot perturb the training RNG stream. The focused tests exercise the
important experimental invariants and STOP/output behavior. No correctness or
maintainability blocker was identified within this scope.

The diagnostic computes Q and terminal/behavior gaps at stored replay actions
(`critic_diagnostic.py:137`); continuation uses the frozen final actor with TD3
smoothing (`:112`). Thus these metrics support critic calibration and target-rate
comparison on this replay. They do not measure current-policy traffic return.

## Verification and limits

Accepted the supplied result: **19 synthetic tests passed in 3.80 seconds**.
Did not rerun unchanged suites or start another full diagnostic. The source and
test hashes match the implementer report, as do the dependency and frozen-model
hashes, checked read-only during this review:

| File | SHA-256 |
| --- | --- |
| `work/sdmpc_rl_recovery_20260929/critic_diagnostic.py` | `134d6f95d4c3ea863d15102e03ac62d6d30d4790eb7f5ba97f4478d354367076` |
| `work/sdmpc_rl_recovery_20260929/test_critic_diagnostic.py` | `56f7780fe4ba3d42b612783a752f74a89d9056f9f563217351b5c58df4ff76d8` |
| `work/sdmpc_rl_multi_20260929/td3.py` | `e27665133ccb3fac2b1769da1c93df73a60c73572fa4ccede24a1296292e2e8e` |
| `results/sdmpc_rl_multi_20260929/pilot_v1/train_round1/model_final.pt` | `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650` |

The synthetic runs reach at most four updates. Production loading, full-budget
numerical stability, and arm-comparison results were not independently executed
or certified here. The coordinator's full run is bounded existing-replay
computation with no traffic effect or policy export; this review does not
introduce an additional approval requirement for it.

Only this `critic-review.md` report was created by the reviewer.
