# Return-initialized shared policy iteration

Status: specification admitted after all five NUF-retention screens passed.
Implementation and independent review required before training. This is the
next bounded stage of the ACTIVE five-scenario goal, not a replacement goal.

## Hypothesis

The old small one-step TD budget with slowly moving targets did not establish
the full75-step return scale before actor optimization. Direct carry-return
initialization can supply that numerical scale; a limited residual TD phase can
then propose an action whose real sequential outcomes inform subsequent learning.
It cannot establish action ranking from two behavior trajectories/scenario.

## Global constraints

- New source work/sdmpc_rl_return_init_20260930; new outputs below
  results/sdmpc_rl_balanced_goal_20260930/return_init_v1.
- Preserve all frozen source, results, physics, demand definitions, reward
  -intervalTTT/100, gamma1, true terminal, previous-executed anchor, physical
  guard and initial/recovery PFO. No canonical data enters training.
- Every optimizer minibatch has exactly20% from each of the five scenarios.
  One state-conditioned actor/twinQ; no scenario-ID network input or five models.
- No automatic arbitrary recollection, model sweep, extra PFO, performance guard,
  dependency installation, commit/push, ACL/power change. Honor STOP/checkpoints.
  At most8 numerical workers overall. Offline training is one single-thread job.
- Freeze stages before execution. Never change an active/source-hashed run.
  No claim of improvement from training loss or exploration data. Apply the
  unchanged canonical five-scenario/reproduction acceptance contract only later.

## Task 1: Offline proposal

Authenticate the completed NUF-retention root (completion/comparison hashes in
its results document) and its five carry references with the existing read-only
validators in isolated subprocesses. Do not boot/reset/step any environment.
Consume only the375carry and375newlocal transitions,150/scenario. Preserve exact
float64 current/next anchors from trace; never reconstruct from rounded float32
observation. Terminal next anchors can be zero and must not be evaluated.

Keep existing2367 physically scaled observations. Actor is memoryless incremental
u=[0.2,0.1]*tanh(MLP(s)), two64-ReLU hidden layers, final layer zero-initialized.
No new policy base memory: zero actor is exactly previous-executed carry even
after fallback. Small bounds are not an episode-wide drift guarantee. Critic
action is projected REQUEST b=P(anchor+[50,1000]*float32(u)), normalized by
[1000,10000]. NP signed; NUF capacity6000. Use float64 transform before network
float32 conversion, same replay/actor/target semantics; execution is not action.

One seed7200, independent saved NumPy/Torch RNG. Shared Phi(s), twin residual
A_i(s,b), all width64/twoReLU, Q_i=Phi+A_i. Fit Phi to each carry trajectory's own
remaining reward sum (exclude warmup, true terminal reset) for1000Adam updates,
lr3e-4, batch40: eight/scenario (one terminal + seven uniform with replacement).
These are training-only sampled V-carry labels, never Q*, other-action labels,
or independent heldout values. Freeze Phi and hash it. Initialize residual final
heads to zero; exact-copy targets after initialization. Actor remains zero.

Then250 critic-only Adam updates lr3e-4, batch40 eight/scenario: fourcarry and
fourlocal, each subgroup one terminal plus three uniform. For carry rows use
observed Gcarry-Phi(s) targets, valid only while continuation remains carry. For
local rows use actual sequential residual TD:
  yA=r-Phi(s) if terminal;
  yA=r+Phi(s')+min A_target(s',P(next_anchor))-Phi(s) otherwise.
No target noise. Stop gradients through full target. Mask true terminals before
next-network evaluation. Target Polyak tau.005 after everysecond critic update.

Before actor proposal, record full per-scenario/horizon Phi and Q training errors,
terminal exact errors, local behavior-return discrepancy with continuation caveat,
own-target residuals, both heads/minQ, and projected-action sensitivities. Finite
parameters/outputs/losses and provenance are mandatory. Phi pooled MSE must fall
at least50% relative to its own initialization, and per-scenario MSE must decrease;
failure is a numerical-fit diagnostic, not a traffic comparison. Freeze these
resource/gate choices now; do not extend1000updates automatically after failure.

If numerical-fit/integrity gates pass, take exactly10actor-only Adam steps lr3e-4
maximizing Q1 through projected requests, batch40 eight/scenario with the same
carry/local quota. Critic/Phi weights frozen; target actor not used/updated during
this proposal. No further critic TD targeting a changing actor. Export one
checkpoint with actor, frozenPhi, critics/targets, optimizer/RNG/counters, full
data/source identity, metrics and a declared critic continuation of carry.
Do not describe these critics as Q of the new actor. Record zero/action/output
hashes, saturation/projection aliases. Only the final fixed-step candidate may
be considered; no selecting an intermediate checkpoint or favorable seed.

Implement resumable phase checkpoints, process identity, STOP handling, atomic
files, finite strictJSON, hashes and completion; preserve partial work. Reuse
existing local runtime/serialization helpers where they fit, avoid environment
imports and broad copied process frameworks. Test algebra/projection/terminal
masking, exact balance, no evaluation leakage, data/hash failures, phase freezing,
deterministic resumed updates, gate failure, STOP and no-overwrite. Independent
task review before one actual training job. Tests may use small synthetic arrays.

## Task 2: Bounded on-policy training wave

Restore clarification after review: unchanged BudgetEnv.restore reconstructs
one saved reference preview on nonterminal resume, without repeating a plant
interval, candidate solve or PFO. Retain this inherited reconstruction; measure
the entire restore wall/CPU separately and include it in session totals. No
additional per-decision RL previews are allowed. Do not claim reference-only
timing or preview-free execution. Parent chose this conservative default under
the ongoing authorization after asking the optional user preference; no reply
or explicit approval is claimed. A later user choice overrides this default.

Only after Task1 actual completion/readout admit implementation of a new versioned
runner using the same frozen BudgetEnv/physical snapshot. Fix one actor checkpoint
before outcomes. Five NEW training profiles, seeds7301..7305 in scenario order,
one75-step/14400s trajectory each, no within-wave learning or added exploration.
At most5single-threadworkers, account for other jobs within8. Resume interrupted
episodes in place; do not rerun completed collection. Keep source/model identity,
raw diagnostics plus admitted taggedInf exports, physical action traces, requested
and executed budgets, real sequential replay, full state checkpoints and timing.

This wave is a provisional policy intervention, not canonical evaluation. No
matched carry for these new profiles is available, so report raw TTT/inventory,
not improvement vs older-profile carries. Prospective completion-health screen:
finite/valid fulltrajectory, <=5zeroNUFrequests, final inventory<=550vehicles in
every row. Preserve bad results even if screen fails; never truncate simulator
physics or relabel failed episodes. A health failure triggers diagnosis, not
automatic additional episodes or actor updates.

## Task 3: Learn from actual continuation

After valid healthy wave, its realized G^pi1 labels refer to that SAME frozen
actor's actual continuation. Use exactly250critic-only residual MC updates on
its375transitions (8/scenario,1terminal+7uniform), fixedPhi and unchanged actor.
This is training fit, not independent calibration; prior carry-target Q error
against pi1 returns is a continuation discrepancy, not calibrated Q^pi1 error.
Predeclare implementation/test/review and new output version before this stage.
Then decide the next limited policy update or canonical full-run evaluation from
actual coverage, terminal and interval losses, action use and fit evidence. Do
not stall solely on an arbitrary tight all-horizon holdout threshold. Goal remains
unmet until all five canonical cases and separate reproduction improve.

## Task 4: Canonical evaluation of the frozen proposal

Tasks1..3 are complete; see rl_budget_return_mc_results_20260930.md. Current
critics retain large long-horizon errors and no observed NUF-action variation.
Do not extend optimization before measuring the existing actor's actual policy.
Implement a compact evaluator in NEW work/sdmpc_rl_return_eval_20260930, output
return_canonical_v1 under the goal root. Use the exact completed MC model
820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6 for all5;
its actor is bit-identical to Task2. Authenticate canonical carry centers and
reuse their results. No old sources/results change, no baseline recollection.

Canonical profiles use training_seed=None, original reset/5warmup/75controlled
steps/14400seconds. Validate exact profile, source, runtime, demand, physical
options, guard and reward/accounting contract against completed canonicalcenters.
No exploration, actor/critic updates, Q selection, performance gate, forcedaction,
or episode substitution. Keep initial/recoveryPFO and physicalfeasibilityguard.
Measure all decision components/restore/workersession scopes honestly.

Parent dispatches only after tests and independent review; maximum concurrent
evaluations depends on current OTHER numericaljobs within global8workers. No
automatic scheduler/coordinator or extra physicalstartup tests needed; leverage
the proven Task2 checkpoint/restore protocol and test adapted contracts synthetically.
Readout reconciles all fulltrajectories/constraints/TTT, interval losses, terminal
inventory, controls, cap/NP/NUF coverage and timing, using one modelhash. All5must
beat their own matched centers beyond max(1e-6,1e-8*baselineTTT), not justmean.
Inventory health is diagnostic, not permissiontochange physics/dropfailedrows.
All5passadmits separatefrozenreproduction, notimmediategoalcompletion.
