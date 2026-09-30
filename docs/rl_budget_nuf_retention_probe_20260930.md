# NUF center-retention causal probe

Status:170pilot COMPLETED and passedallpredeclaredintegrity/sufficiencygates.
TTT4553.111048961477,terminalinventory418.71776352836196; noD-rampclosure.
All75RNGvectorsmatcholdlocal; physical/nativeprefixthrough9matches, firstaction
difference10. OldlocalTTT11945.939292655728, pairedtrainingcarry4602.4114708736415.
This is a same-seed trainingprofile explorationprobe, NOT a learnedpolicy or
canonicalacceptance. Samecode/source nowadmittedforremaining4; reuse170once.
See SDDledger forcurrentprocess/logstatus. Oldcollectionstayscompleted/frozen.
Evidence: rl_budget_local_collection_results_20260930.md and the completed
rebase-diagnosis-report.md in that plan's SDD directory. This is the next
falsifiable stage, not a new goal or a successful learned policy.

## Hypothesis and treatment

Retaining the initial NUF exploration center, rather than adopting the lower
achieved reference budget after physical fallback, reduces the observed170
long-horizon loss sufficiently to restore the predeclared inventory screen.
The earlier near-zero D-ramp response may persist; the hypothesis may fail.

Change only the local policy's NUF center update. Keep base[1] equal to its
initial actual action anchor. After every commit set realized_offset[1] to
actual B_executed[1]-base[1], even on fallback. NP rebases/resets exactly as
before. Preserve mean reversion0.8, noise[10,100], offsetbounds[50,500], PCG64
seeds/draw ordering, rate/actionscales[50,1000], float32 conversion and capacity
projection. Actual previous-executed-budget anchor and all physical execution
rules remain unchanged. A desired NUF in[5500,6000] does not mean an actual
rate-limited request or executed flow is forced into that band.

## Global constraints

- New code: work/sdmpc_rl_nuf_retention_20260930; new root:
  results/sdmpc_rl_balanced_goal_20260930/nuf_retention_v1.
- Freeze all old sources/results, including local_budget_v2 and export helper.
  No in-place patch, remigration, old collection restart or old data deletion.
- Unchanged physical/runtime/config/profile/reward/gamma/terminal/accounting
  contracts; one candidate budget solve per interval, no new PFO or preview.
- This is training-profile exploration, not canonical evaluation. No actor,
  Q training, action optimality labels, performance fallback or retuned follower.
- At most8 numerical workers total; stage1 one single-thread170worker, stage2
  at most4remaining single-threadworkers. Respect STOP, ownership and creation
  identities; no overlapping runner and no repeated completed collection.
- Every future learner minibatch remains exactly20percent per scenario, one
  shared state-conditioned policy/value. No training until all5new local runs
  complete, all screens pass and a learner specification is separately admitted.
- No commit/push/install/ACL/power changes. Preserve unrelated dirty work.

## Stage 1: Implement, verify, run170

Reuse frozen collection machinery in a new version with explicit source
provenance. Fix the already-understood diagnostic-Inf JSON boundary in the NEW
collector directly, retaining raw checkpoint values and explicit tagged paths;
do not reintroduce repeated failed terminal exports. Unknown/physical nonfinite
values still fail. Preserve all original checkpoint/sequence/inventory, timing,
RNG-replay, STOP and process checks. No general process framework redesign.

Authenticate the completed v2 carry/local references read-only; avoid mixed
same-name versioned Python modules. A fresh read-only verification subprocess
using the admitted v2/export validators is acceptable. Use pinned artifact hashes
to reuse those references, rather than copying/relabelling or recollecting them.

After focused tests and independent review, run only local sweet_170_w with
training seed6802/explorationseed6902 from normal reset to75steps/14400s. Preserve
its checkpoints, complete output and resume identity in the final five-slot root.
An interruption resumes that episode; it never starts another seed or resets it.

Integrity: compare original profile/config/physical/runtime/warmup, all75RNG
vectors, actual anchor chain, feasible execution, full reward/end/accounting.
Require matching actions/physical control/observation/reward/TTT through step9
(zero-based), excluding timings/policy metadata; existing1e-8accounting
tolerance applies. First action difference should be10, not15. New policy
memory differs at9; environment observations/physical outcomes through9 do not.
Violation invalidates attribution and requires diagnosis, not a performance label.

Predeclared sufficiency: zeroNUFrequests<=5 and terminalinventory<=
517.470147382039, plus fullTTT<11945.939292655728 (old local170).
Report the comparison to pairedcarry4602.4114708736415 too, without claiming
canonical/shared-RL improvement. If integrity passes but inventory fails, the
proposed treatment is insufficient. No lower TTT falsifies its loss-reduction
prediction for this seed. Preserve negative results; do not auto-expand.

## Stage 2: Conditional balanced extension

Only after stage1integrity/sufficiency pass, use the EXACT same frozen candidate
to run local155/incident/skew/190 with seeds6801/6901,6803/6903,6804/6904,6805/6905.
Skip and reuse the completed170 slot. Reuse all5v2carryreferences. This is75
newtransitions first,300more only if justified; no repeated carry collection.
All5must pass the unchanged zeroNUF/inventoryscreen relative to their own
same-profile carries and all physical/integrity checks. Do not require identical
post-intervention actions to old traces or assign their old returns to new actions.

## Stage 3: Readout and next decision

Report per-scenario training TTT, interval losses/inventories, budget/headroom
drift, fallback events, closure/control diversity, Q absence and measured timing.
Passing data screens only supports coverage; it does not prove policy optimality.
Keep old750transitions versioned and distinguish bad behavior from invaliddata.
Choose/test the smallest shared learner revision only after this readout; if the
probe fails, diagnose initiation vs persistence instead of blindly recollecting.
