# Task1: NUF-retention candidate implementation

Read this first, then docs/rl_budget_nuf_retention_probe_20260930.md: that short
plan is the exact numerical/protocol specification for this single task.
Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL. You own only new
work/sdmpc_rl_nuf_retention_20260930 and task-1-report.md in this SDDdirectory.
No actual simulator reset/step/run, checkpoint migration, oldsource/results
edits, modeltraining, install, commit/push or arbitrary subsystem refactor.

Implement the smallest complete collector/probe workflow so parent can review
and run170first, then optionally4remaining with unchanged code. Preserve the
original v2patterns rather than inventing another scheduler/framework.

Interface/context:
- Old work/sdmpc_rl_local_20260930_v2 modules implement sound frozenphysical
  checks, fullenvironmentcheckpointresume, RNG-policyreplay, 75stepvalidation,
  sessiontiming and Windowsredirector ownership. Read needed source first.
  A mechanical copy of necessary collector modules into the new version is
  allowed, manual edits via apply_patch. Do not copy oldmigration/resultdata.
  New canonicalroot nuf_retention_v1 needs no migration and only5localslots;
  carries are immutable external comparisons, never newly dispatched.
- Change only NUF base retention and corresponding realizedoffset bookkeeping
  in exploration; new policy/collectionformat prevents oldcheckpoint confusion.
  Keep NPupdate rule, noise sequence and oldprojector/rates unchanged.
- New diagnostic JSON serialization may reuse the pure tag_diagnostics function
  from immutable work/sdmpc_rl_export_recovery_20260930/export_recovery.py with
  explicit sourcehashpin (do NOT use its actual export/supervisor entrypoints
  for newroot). Rawcheckpoint retains inf; export trace tags only exact old
  +Infstationaritypaths, receipt lists paths and hashes; unknown nonfinite fails.
  Finaloutput validators must accept/reconcile this additional diagnostic audit
  and coreexperience unchanged. No terminalfail/retry workaround in newversion.
- V2 root completionSHA5b6e645f6f6ca64c4f593c52d19ee3ab5bda7ae3c3d31f423d6fae2d2f3ef4a3,
  comparisonSHAc294c1f5b2100daeaccb4c927357a8bc73d44391f34641fd37b4ba5b2555c4cd.
  Prior report77hashchecks and fullsupervisorvalidation already passed. Reference
  authentication should reuse original validators in a fresh subprocess/read-only
  entrypoint ifneeded, avoiding same-name old/newmodule collisions. Verify linked
  repairedreceipts as well as original outputhashes. No need deserializeraw
  environmentcheckpoints just to read completed v2experience (ordinaryarrays).
- settings/manifest must explicitly bind newtreatment, source/runtime/physical
  contract, pairedoldcarry/localreference hashes/profile/seed/runID, realnewrunID.
  Oldreferences may differ in collection-codeidentity, not physicalcontract.
- Newcollector retainsactualprevious-executedanchor and passesonlyone requested
  budget to unchangedBudgetEnv. NoextraPFO, previews, h3performanceguard, lost
  reward, fakeend, canonicalfit, branchfromoldterminal or reusedhypotheticalrows.
- Runner stages: pilot170only; remaining4onlyaftercompleted170 passesallplan
  criteria, unchangedsamecandidateidentity. Neverrelaunch170. Do not autoretry
  arbitraryfailed/unknownwork. Processrecordmustincludeactualcreationidentity;
  existingv2workerredirectorclaimcheck reused. All STOP and cohort/slotlocks,
  retainedrunID/monotonicprogress/draining rules survive. No active oldrunner.
- Exact comparisonthroughstep9 uses oldactual experience/trace under sameprofile;
  compareactions/physicalcontrol/nativenextobs/reward/TTT, excludingtimerfields
  and newpolicymemory. At10old/newaction differs; all75noisevectors mustmatch.
  Iffloatingdifferencesoccur useexisting1e-8accountingtolerance, do not silently
  weakenprefix/RNG/physicalchecks. Archiveintegrity failures, no causalclaim.
- Timing: retain measuredoriginalsession-scope definitions and provenance;
  include diagnosticserialization in newcollector scope, separatecoordinator/
  verification cost. Don't claim millisecond control orfullendtoend totals.

Tests: focus on amendedsurface. Cover rebaseNUFretained/NPunchanged; realized
offsetconsistency, lowexecutedbudget+rate-limited recovery, bothcapboundaries,
zero-noisemeanreversion, resume/RNGroundtrip; pure saved170prefix/arithmetic
regression (newstep10NUF5552.771855 approx vsold5504.421625); independent5slot
ownership/resume/skip; pilotfailblocksremaining and successfulpilotnotrepeated;
pairedsource/profilemismatch rejection; terminal+diagnostictags/noaltered
experience; realunchangedphysicalconfigextra serialization. Use smallsynthetic
fixtures for workflows and existingread-onlytraces for pure regression. Do not
run frozen236/133suites orsimulations. ExistingPython/deps need sandboxread
escalation; do not install dependencies. Report exact tests/evidence/hashes,
mocklimits and executableparentcommands. Parentindependentreviewprecedesrealrun.

Keep report concise with sourcecopy/changeinventory, tests, correctnessrisks and
nextcommands; returnshortstatus/testsummary. No unrelatedcodecleanup.
