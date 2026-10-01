# Task2 bounded frozen-actor training wave

R1 clarification, parent resolution before any actualrun: preserve unchanged
BudgetEnv.restore, including its one nonterminal reference reconstruction.
The no-extra-preview restriction below means no new per-decision RL previews;
this inherited restore-only work is an explicit exception. Measure entire
restore wall/CPU and declared branch reconstructioncount separately, retain it
in session totals, do not pretend it is isolated reference-only cost. No plant
interval/candidate solve/PFO repetition is permitted. Parent asked optionaluser
preference and defaults to minimum physicalchange underongoingauthorization;
no useranswer/approvalisclaimed. Lateruserinstructiontakespriority.

Implement only NEW work/sdmpc_rl_return_wave_20260930/ and task-2-report.md here.
No real simulation or optimization during implementation; synthetic tests and
read-only model/experience/contract checks allowed. No commits. Read integration
notes alongside this brief for exact frozen helper boundaries/importcollision.

Task1 actually completed, alloutputsreconciled. Model:
results/sdmpc_rl_balanced_goal_20260930/return_init_v1/model_final.pt
SHA7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904.
Task1completionSHA34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757.
No edits to completedsource/model/checkpoint/data. Dataformat/modelsettings contain
the real observation/source contract. Critic continuation is carry, NOT newactor.

## Global constraints

- One frozen shared actor for155/170/170incident/170skew/190, in SCENARIOS order.
  New TRAINING seeds7301/7302/7303/7304/7305; never canonical evaluation. Fixedroot
  results/sdmpc_rl_balanced_goal_20260930/return_policy_wave_v1/<scenario>.
- Original BudgetEnv, physicalsnapshot, config, options, followers, demandprofile
  construction, reward=-intervalTTT/100,gamma1,true75terminal,14400s,warmup5,
  previous-executedanchor, physicalguard and initial/recoveryPFO unchanged.
- No per-worker learning, target updates, exploration, extra previews/PFO, forced
  actions, performance fallback, dependency/install/ACL/power/commit/push.
- At most5single-threadworkers, globalbudget8 includingunrelatedjobs. Parent owns
  dispatch/budget/CIM/launcherrecord and waitsallactualworkers before nextwave.
  Implement a compact single-worker CLI and completed-wave readonly validator,
  not a new coordinator/processframework. Each slot must hold kernel lock, record
  actualPID+creation+command, source/model/settings, STOP and sessions/checkpoints.
  Honor repoSTOP, goalSTOP, waverootSTOP, slotSTOP; never remove/ignore markers.
  Refuse completedslotrerun and mismatch/duplicate. Interruptions resume same
  fullenvironment checkpoint without reset/step/PFOrepetition. Preserveorphans.

## Model and episode interfaces

Avoid generic learner runtime.py collision with frozen physicalruntime. A compact
local actoradapter must exactly reproduce admitted Actor:2367-64ReLU-64ReLU-2,
then torch.tanh * storedfloat32bounds[.2,.1]. Loadstrict completeactorweights from
dictcheckpoint learner/models/actor; do not load/execute learner optimizer/data
modules in theenvironmentprocess. ExactoutputparitytestwithoriginalActor on
saved observations in isolatedreadonlysubprocess; no randomnewmodel substitution.
Validate completion/source/spec/phase=done/counters1000/250/10/125/gate/modelhash
before any environmentrun. One identicalmodelhash in all5settings and summaries.

After boot physicalruntime, create BudgetEnv(training_seed=7301+index,guardphysical),
verify runtime/config/options/observationcontract against authenticatedpredecessor.
Log profilehash, seed, runid, physicalconfig usingto_plain_dict, warmupTTT. Explicit
mode training_wave, evaluationFalse, explorationFalse, noimprovementclaim.
Actor inference underno_grad consumesnativeobsfloat32, outputfloat32nominalaction;
actualunchangedenvironmentperformsprojection. Log actorwall/CPU in decisiontime.
policy_q=None duringphysicalrollout: noextraQcalls, criticsnotQpi1. Preserve
completeobs/action/reward/nextobs/true-terminalexperience andphysicalrawtrace.

Checkpointeveryactualinterval: fullBudgetEnv.checkpoint, obs, trace, transitions,
modelhash/settings/sessiontiming and progress. Validate boundary/experience/action
chain before restore, exactrestoredobs, and frozenactor.act(obs)==loggedaction for
everyretainedrow. Atterminal save fulltrace, experience, schema, summary, bound
hashcompletion and timing. Trueclock/reward/areaTTT/inventory/controlconstraints,
onecandidate/step, PFOcounts, source/runtime/modelunchanged, full75required.
Known diagnostic+Inf only via pinnedtagger, preserve rawcheckpoint with auditreceipt;
NaN/-Inf/physicalorlearningnonfinite mustfail. Do not deserializerawcheckpoints in
readonlycompletedwavevalidator unless necessary; experiencedata+tagreceipt+hashes
supportsequencevalidation. Rawphysicalcheckpointis stillrequired forresumption.

Reuse small existingfrozen accounting/inventory/observation/sequence helpers where
compatible, but newformat/settings/model/profileauth must notuse oldseed/policy
assumptions. Read task-2-integration-notes.md. No falsemodelQ or nonlinearpriceclaim.
Summaries reportTTT, intervalTTT/inventory arrays, terminal/peakinventory, budgets/
controlsdiversity/drift/saturation/cap/zeroNUF, fallbacks/closures/PFO/solves, actor
and fulldecisiontiming, lockedworkersessions with knownscope and unknowninterrupted
sessions. Do not sumparallelworkerseconds aselapsedwall or claimactortimeaslatency.

Completed-wavecheck requires exactly5uniquecomplete75step14400strajectories,
onefixedmodel, expectedprofiles/scenarios, no canonicaldata. Predeclaredhealth:
eachzeroNUFrequests<=5 and terminalinventory<=550, plusallintegrityfinitechecks.
Preservebadresults; healthfailure is diagnostic, notautomaticretryorstopmidphysics.
No pairedcarryexistsforthesenewprofiles, so NOcomparisonpercentagevsoldcarries.
Do notfit critic ordispatchanotherwave. ParentdecidesTask3afteractualreadout.

CLI must support --scenario, --resume and --max-new-steps for a checkpointed real
integrationprefix. Parentwillrun155oneinterval thenresume it alongsideother4;
prefixnotdiscardedorrecollected. Tests: parity/importboundary, actorhash/spec/source
failures, exactaction/experience/physicalaccounting, STOPalllevels, source/model
mutation, duplicate/completedrefusal, interruptedresume/timeunknown, diagnosticInf
boundaries, all5healthgates/missing/duplicate/mixedpolicy/wrongprofile rejection.
No need to rerun earlierlearner/probe/fullsource suites; scopedtests plusreadonly
auth/parity evidence suffice. Fullreportwithcommands/hashmanifest/testlimits,
changedpaths andactualprefix/wavereadoutcommands. Parentreview beforeactualrun.
