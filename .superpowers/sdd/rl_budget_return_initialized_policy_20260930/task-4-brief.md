# Task 4: Fixed-policy canonical evaluator

Read this first: exact requirements for one scoped implementation, no actual
simulation/learning until independent review and parent admission. Workspace
C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL, dedicated dirtybranch at c40eb9d.
Preserve all existing work. Do not commit/push/install/change OS or old files.

## Purpose and Scope

Task3 MC fit completed; actor unchanged, no canonical performance known for it.
Evaluate the current fixed policy before another actor gradient step. Only own
NEW work/sdmpc_rl_return_eval_20260930 and task-4-report.md in this SDDdirectory.
Numerical output (parentonly): results/sdmpc_rl_balanced_goal_20260930/return_canonical_v1.
No automated next-stage queue, broad framework, physical collection test, actual
fit, or actual fullrun in this implementation. Read docs/rl_budget_return_mc_results_20260930.md
for evidence/context, not other planworkspaces. One evaluatorworker/scenario and
readonlyfullreadout; parent controls global8workerbudget and dispatch.

Reuse directly relevant proven helpers from work/sdmpc_rl_return_wave_20260930
(actor inference, checkpoints, operations, restore accounting, timing, validator
primitives) and sdmpc_rl_multi_20260929 where fit. Do not rewrite the full older
runner architecture or modify completed frozen sources. Prefer small explicit
adapters/functions to broad copying or global monkeypatching. If fixed existing
training-only checks prevent direct reuse, adapt only those checks explicitly,
with tests showing canonical inputs rejected by old but admitted by new contract.
Do not falsely mark canonical data training-only to bypass a validator.

## Exact Model and Physical Contract

Model: return_mc_v1/model_final.pt, SHA
820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6.
Completion SHA a815f65f315ce1a73cce8a3754b9edb3fac9dbed6a8fd0aece9b9238ea532675.
Spec3610bb43c22430a3fb9ba95f7f14175b2b29b68a2000d6bfe1af04ec4c8ff811.
Actor tensor9956ba4f6f5d64c7a65bf05bdab47984efe041e15358dafebbb9bfd731d44877.
Explicitformat sdmpc-on-policy-return-mc-v1, phase done, counts mc_critic250,
ancestor counts phi1000/critic250/actor10/polyak125, continuation pi1_recorded_on_policy_MC.
Authenticate completion/model/output/settings/source/data identities and fixed
actor. Tiny strict meta-loaded inference adapter only; never construct a generic
learner in the physical process or import Task1 genericruntime/learner/data.
Prove bitexact parity on all375 retained Task2 observations against recorded
actions. No evaluating physical canonical profiles in tests. Q inference absent;
policy_q=None explicitly, policy is actor only, critics held in model for learning.

One modelhash across scenarios, no scenario-ID-input or per-scenario policy:
sweet_155_w, sweet_170_w, sweet_170_incident_w, sweet_170_skew15_w, sweet_190_w.
Use canonical BudgetEnv(training_seed=None, guard_mode='physical') from exact
unchanged physical snapshot artifacts/sdmpc_budget_baseline_20260929/source.
Original reset,5warmup+75controlledsteps,14400s, dt/control180s, true terminal,
gamma1/reward -intervalTTT/100. NP signed/NUF0..6000. Native float32 action,
float64 anchor+[50,1000]*action projection, previous-executed carry anchor;
NO new policy base memory. Actor bounds[.2,.1], network2367/64ReLU/64ReLU/2tanh.
Initial/recovery PFO and existing physical feasibility guard remain, counted and
timed. No exploration, learning, Q-based choice, performanceTTTguard, gating,
forcedfirstaction, or extra action-preview/response search. Retain inherited
nonterminal restore reference reconstruction, separately measure ENTIRE restore
wall/CPU and include within session once, no repeated plantinterval/PFO.

Frozen physical contract digest
d6d14c4c18a7f8cfc88969bf0e9b8231ba5c452600593db49b125c1e5730eb70
from value_audit_v1/projection/completion.json SHA
b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3.
Compare exact scenario canonical profilehash, sourcepins/runtime, cfg/options,
guard/accounting with existing completedcarry centers under
results/sdmpc_rl_multi_20260929/pilot_v1/center/<scenario>.
Reuse compare_runs.load_completed_run(folder,'center',[None]) in an isolated
readonly process if its td3 import would collide with inference boundary. No
physical boot/reset/step/restore/optimizer in readonly authentication/readout.
Hash-bind their current actual evidence in preflight. Reject incompatible centers;
never recollect center or use training-wave returns as matchedcanonicalbaseline.

Canonical baselineTTT respectively:
3103.0110715680044,3935.903236508048,5546.224352256691,
4250.876599300032,6604.2970168093225.
Reconcile from actual retained outputs, not just hardcoded numbers. Warmup+all
intervals, area totals, profiles, full timeline/terminals, no learning/exploration,
one candidate/previous reference/PFO accounting, executed physical controls and
fallbackvalidity. Baseline source/runtime/env contracts must equal candidate's.
All5must individually improve beyond max(1e-6,1e-8*baselineTTT). No average-only
success. Passing all5means eligible_for_separate_reproduction, NOT goalachieved.
Any failure remains fully reported; no inventory/healthscreen filter dropping
rows, alteredphysics, seed selection, or trained-on-evaluation data.

## Runtime and Readout

Existing STOP scopes REPO/GOAL/newroot/scenario, processidentity PID+creation+
command, 1Torch/BLASthread/worker, preserve kernelsinglewriter lock, actual
source/modelsettings hashes, full simulator/controller/observer/profile state,
immutableintervalcheckpoints+atomicpointer, exactobs/boundaryresume. Preserve
orphans and refuse uncertain inflight operations/repeatedphysics and completed
duplicates. Add explicit parentreviewreceipt gate for exactnewsource/spec before
physical startup. No hidden dependency installs. Parentlaunch hidden logs outside
freshslot. Tests synthetic, source/data auth may read actual artifacts only.

No automatic prefixsimulation needed: Task2 proven actualrestore and tests already
cover it. New wrapper behavior and canonicalmodel/profile changes still need
focusedtests. Keep finalization identity stable across STOP/resume after75steps;
do not repeat physics. Reject mismatched retained output rather than overwrite.

Preserve trace/actualtransitionevidence and allowlisteddiagnosticInf tags/rawck.
Canonical evidence is explicitly evalonly, never fed to trainingbythis runner.
Report raw/percentmatchedTTT changes, perintervaldifference windows (1..25,
26..50,51..75), terminal/peakinventory, physicalcontrol/NP/NUF/requestcoverage,
fallback/PFOcounts and all measured decisioncomponents/plant/sessiontime scopes.
Actor-onlytime is NOT totalcomputationtime; includePFO/references/guards/solves.
Queueexposure remains endpointestimate, not exactsubstep. Parallel sum !=wall.
No previewfree claim or nonlinearprice causation. Deterministic all5repeat alone
does not establish stochastic generalization. No automatic reproductiondispatch.

## Tests, Handoff

Focused synthetic/newcontract tests, exactall375actorparityreadonly, baseline
auth/accounting/profile/source/modelmismatch failures, evalflags/noQ/nooptimizer,
physicalimportboundary, canonicalnottraining metadata, STOP/lock/resume/nooverwrite,
operationorphan refusal, partial/fullreadout and strictall5acceptance. Reuse prior
successful suite evidence, no whole-history reruns. Before/after hash snapshot
for all actual inputs touched. No actual optimizer/simulation called bytests.

Write fulltask-4-report.md: changedfiles, rationale, tests/evidence/hashes,
exactsource/spec/commands forparent, actualoutputsABSENT, residualconcerns.
Return onlystatus,one-line testresult,changedpaths,concerns; no commit. Parent
will generate diff and independentreview; do not fabricateapprovalreceipt.
