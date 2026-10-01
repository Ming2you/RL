# Task1 offline learner requirements

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


Write only work/sdmpc_rl_return_init_20260930/ and this task-1-report.md. No real training or simulation during implementation. No commit. Keep code compact and use existing dependency paths. Report exact tests, hashes, unchanged old sources/artifacts, limitations and command for the admitted actual job. Parent owns docs and dispatch.

