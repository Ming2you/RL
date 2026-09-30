# SDD ledger - plan: docs/rl_budget_local_collection_plan_20260930.md

Active parent goal: docs/rl_budget_balanced_goal_20260930.md, NOT achieved.
Previous goal turn: progress (3 completed diagnostics, scoped hypothesis tested,
158 passing tests, final independent integration review, next collection chosen).
Current preflight: no Python workers, no current STOP, local_budget_v1 absent.
Branch codex/sdmpc-rl-budget-20260929. Preserve dirty staged/untracked work.

Tasks:
1. Pure resumable local budget exploration rule and synthetic tests (sidecar).
2. Versioned collector, validation, paired-wave runner and tests (parent).
3. Independent reviews, combined tests and bounded physical smoke admission.
4. Execute 5 carry + 5 local training episodes once; reconcile paired coverage.
5. Report results and choose next shared learner revision. Goal remains active.

No modifications to completed old scripts/configuration. Maximum five numerical
workers this phase, within user's total8. Repo/goal/experiment/child STOP honored.
No commit, push, dependencies or OS changes. No production run launched yet.

- Task1 Rawls01a0ee61-94a6-71e2-bdfa-9bc3ee21cec1 DONE,closed;121testsPASS.29s.
  exploration-brief.md/exploration-report.md are task requirements/evidence.
- Task2 parent implemented5newfiles;22testsPASS7.86s. Readonlyrealidentity
  preflightPASS7sourcepins, correctoldphysical/runtime/gate. No model orenvboot.
  collector-brief.md/collector-report.md ready. Independent reviews pending.
- Reviewer Zeno01a0ee6e-58e1-79a3-9864-9fb6e7bc55d7 active, report
  implementation-review.md. Combined143testsPASS8.47s exit0. No numerical
  worker/process active. Production waits for review, then smoke firstcarry155
  --max-new-steps1 and resume sameepisode into wave (no repeated collection).
- Zeno review: explorationSPEC/QUALITYPASS; collectorFAIL R1cohortpath/ownership,
  R2prefixobs/budgetvalidation, R3inventory/terminalstate, R4slotbinding,
  R5sessiontiming. All five accepted as scope-relevant. NOphysicalsmoke launched.
  before-fix1 snapshots saved; fix1-brief.md scopes a fresh fixer. Source is not
  yet admitted/frozen for production. Current goal turn remains progress.
- Fix round1 implementer Ramanujan01a0ee78-aec6-7622-a296-7fd73e99b46c active;
  owns collector modules/tests and fix1-report.md. Parent must not edit those
  modules or launch collection until scoped re-review passes. No production
  simulation or learner has run in this stage yet.
- Parent wrote docs/rl_budget_next_learner_notes_20260930.md, conditional design
  notes only. They distinguish carry-policy returns from optimal action values
  and record representation/actor-initialization decisions still to be tested.
  They do not admit a learner or supersede the collection coverage screen.
- Fix round1/5: 189testsPASS33.54s, independent Curie review fix1-review.md:
  R2/R4/R5 addressed, R1/R3 open. F1 complete config serializer drops runtime
  dataclass extras; F2 finalized/checkpointed slot identity must survive output
  relocation without a later ownership scan. Both accepted. No production
  launched. Real read-only identity passed (8 sources); no Python worker/STOP.
  before-fix2 contains exact reviewed sources; resume original fixer for F1/F2.
- Fix round2/5: 5 pre-fix failures reproduced, 79collectorPASS64.58s,
  combined200PASS66.13s. Curie fix2-review.md SPEC/QUALITYPASS, both open
  findings addressed; no new Important issue. All subagents closed.
- Task1 complete: exploration reviewed/tested. Task2 complete: collector,
  paired runner and validation reviewed/tested (uncommitted by authorization).
  Task3 code admission complete; physical smoke next. Source must now remain
  frozen. Run carry155 --max-new-steps1, resume that same episode for one more
  interval to exercise actual restoration, then resume remaining intervals via
  run_wave. This adds no duplicate collection or episode and changes no reward
  or terminal flag. At most five numerical workers during the full wave.
- Task3 physical smoke PASS: carry155 first interval PID28800 exit0, then actual
  resume second interval PID45436 exit0. Total66.7563663s defined session scope.
  2 actual transitions saved; original checkpoint/settings hashes and runID are
  recorded in windows-recovery-brief.md. No completed episode/TTT improvement.
- Task4 first wave FAILED before any child boot. Windows venv redirector makes
  Popen launcher PID differ from actual os.getpid. All5children failed exact
  identity claim; all Python processes confirmed dead. Original source/results
  now frozen. Minimal nonnumerical reproduction confirmed parent/child mismatch.
  Next task: bounded new v2 launcher repair and explicit preservation/migration
  of the same two-interval prefix. See windows-recovery-brief.md. Do not retry
  v1 wave or mutate its settings/ledger. This is actionable progress, not an
  external blocker or goal completion; original authorizations remain active.
- Recovery implementer Archimedes01a0eea9-20f7-7640-b6f2-90a4c00988ae active;
  owns new v2 only, report windows-recovery-report.md. No numerical worker active.
- Recovery implementation complete: 236testsPASS78.46s, actual stdlib-only
  Windows redirected subprocess identity verifier PASS, all54v1files preserved.
  windows-recovery-report.md/diff contain evidence and exact next commands.
  Newv2 outputs still absent. Independent review pending; do not run actual
  migration or any numerical collector before it passes. Implementer closed.
- Recovery independent Arendt review SPEC/QUALITYPASS; reviewer closed. Real
  migration dry-run PASS, --execute PASS, immediate idempotent verification PASS.
  Exact physical/state/obs/2transition/policy/RNG/session preservation verified;
  original54files still byte-exact. Receipt SHAf1b895d659c930708782e3b4fd3e99b21f3ca5ff6540e315880de8ab1dbf2a43.
- Task4 ACTIVE v2 wave launched05:11:46KST, launcher3100, coordinator46348,
  attemptb96cfc7cd0164c15a0c26e02ebdff860. All5 actual workers advanced, no errors;
  carry155 log begins3/75 (no prefix repeat). See durable parent report
  docs/rl_budget_local_collection_status_20260930.md for commands/PIDs/budget.
  A separate verified serial single-core project was left untouched; observed
  numerical total6 incl our5, below8. Do NOT launch overlapping coordinator,
  remigrate advanced output or modify either frozen source version.
- Task3 physical integration complete. Current goal turn made actual progress
  (2preservedsteps, diagnosed/repaired launcher defect, live5-scenario collection),
  not blocked. Goal not achieved; no new policy evaluated. Next: await/reconcile
  both waves, then decide the shared learner from the predeclared screen.
- New continuation: previous turn classified PROGRESS. Live coordinator/5worker
  identities were verified, then waited without duplicate launch. At05:28KST
  authoritative wave FAILED and all handles absent after successful drain.
  Carry155 completed75physical intervals but failed strict JSON export of four
  positive-inf candidate stationarity diagnostics at control_step14. Read-only
  checkpoint scan proves exact paths; no other trace nonfinite values. Other
  prefixes safely stopped:170=34,incident=48,skew=51,190=42; total250preserved.
- Next scoped task export-recovery-brief.md: a companion terminal exporter and
  bounded continuation wrapper outside immutable v1/v2. No remigration, source
  edit, discarded data or repeated interval. Tagged diagnostic inf remains
  explicit; unknown/physical nonfinite values still fail. Goal ACTIVE, no policy
  improvement claimed. This is a diagnosed local output defect, not external
  blockage. V2 runner is NOT currently running despite earlier execution entry.
- Export companion implementer Descartes01a0eede-d3e4-7d52-936d-852e811faf3a
  ACTIVE, owns only new work/sdmpc_rl_export_recovery_20260930 and report.
  No original code/result modifications or production export by subagent.
  Additional actual checkpoint validation PASS; offending candidate was rejected
  in favor of physically feasible reference fallback. Finite training summary
  recorded in docs/rl_budget_local_collection_status_20260930.md.
- Export companion implemented100focusedtestsPASS19.81s;214oldsource/result
  files unchanged. Parent additionally validated all250actual saved transitions:
  only allowed positive-inf diagnostics (155:4,170:7,190:5;incident/skew:0).
  export-inputs-before.json pins35current inputs before production export.
- Independent Russell export-recovery-review.md SPEC/QUALITYFAIL, accepted
  F1no-worker final coordinator identity, F2retained receipt token/runID,
  F3all-completion-path receipt authentication. before-export-fix1 snapshots
  preserved. Resume original Descartes with export-fix1-brief.md; NOactualexport
  or numerical resume yet. This is repair round1 for the new companion only.
- Export fix round1 implemented:133focusedtestsPASS63.48s,214frozenfiles
  unchanged. Implementer closed. Same Russell resumed for scoped F1-F3/new-diff
  re-review using export-fix1.diff, export-fix1-review.md expected. No actual
  export or collector started yet. Read-only process check06:12KST found no RL
  numerical process; one unrelated project diagnostic left untouched.
- Export fix round1/5:3addressed,0open. Russell export-fix1-review.md SPEC and
  QUALITY PASS; reviewer closed. Companion admitted/frozen at hashes in report
  (export f7951394,supervise a119ee8a,bootstrap9aead7b5). Parent will execute
  terminal155 output-only export, verify receipt/unchanged inputs, then launch
  bounded supervisor. Last STOP/process recheck found neither STOP nor Python.
- Actual terminal155 export exit0, repeat verification exit0. All35preexport
  pinned inputs unchanged except current155status; original failedstatus is
  archived byte-exact. Raw75transition checkpoint/settings/schema/timing/logs
  preserved. Fourallowed diagnostic tags; TTT3210.2043413811693 unchanged.
  Receipt SHA8e54beb7fcfb8e7ec322aed45b18ef1096ae7497e6adc83f00e8f6228ee5a83e;
  exportadditional5.575445799971931s (definedscope, not endtoend). No interval
  replay/reset/PFO performed. This is training carry output, not RL acceptance.
- Bounded supervisor ACTIVE06:29:31KST, launcher2856,actual25164 creation
  FILETIME134351909711276110. Its attempt314820db745b49d0b5608d0fac51f954
  bootstrapped actualcoordinator44524 (launcher36416), originalv2waveattempt
  5575201be550434999246c6137ecdd98. Binding handshake succeeded. Fourcarryworkers
  actual50604/50756/23760/5872 resume170/incident/skew/190;155skipped. Max5
  numericalworkers, no other numericalprocess at launch. Parent launch record
  in local_budget_v2/export-supervisor-parent-20260930_062931.json.
  Allhelpersnowfrozen: do NOT run independent run_wave or another supervisor.
- Carry wave COMPLETE375transitions: incident/skew original completion;155manual
  exporter;190/170 same known diagnostic exporter via supervisor. Both automatic
  repairs succeeded without recollection. Supervisor now on waveattempt
  2cea588974b04a01b1964beab8faeb18, fiveLOCALworkers active (6084,26608,36252,
  36384,37472) about4-5/75. No actionable new failure, no duplicate dispatch.
  Paired screen and shared learner admission still await complete local wave.
- Task4 collection COMPLETE750transitions, supervisorCOMPLETED(all15attempts
  exited,8wave/7export;155manualexportadditional). NoPythonprocessafterfinal
  check. Completed hashes/readout in docs/rl_budget_local_collection_results_20260930.md.
  ScreenFAILED170:TTT11945.939292655728 vs4602.4114708736415; inventory2966.2218
  vs413.9761(7.1652times). Other4inventoryscreensPASS; allzeroNUFcounts0.
  NOlearner/canonicalevaladmitted. GoalACTIVE, thisturnPROGRESSnotblocked.
- Parent trace investigation: movinglocalbase causes noepisodebound.170NUFbase
  6000->5758.25(step9)->5361.15(step12)->2840.80(step15)->3249.98(step26),
  persists. But twoDrampsalreadyclosedat14 feasiblelowerresponse before15rebase.
  Need distinguish initiation from persistence. Nextrebase-diagnosis-brief.md
  scopesread-onlyindependent decomposition and falsifiableone-variableprobe;
  do not auto-launchnewcollection or train from failedscreen.
- Planck01a0ef46-c24d-70c0-8559-7d5471330f58 ACTIVEread-onlyrebase diagnosis;
  owns rebase-diagnosis-report.md (optional evidenceJSON in same SDDdir). No
  sourcechange/model/physicalrun authorized for this subtask. Allnumerical
  workersdead. Parent updatedmain goal'scontinuation pointer tocompletedfailed
  screen, preventingoldaudits/collection frombeingrestarted.
- Rebase diagnosis COMPLETE:Planckreport116lines,77hashchecksPASS,all750JSON
  transitions/inventoryreconciled. Agentclosed. Mechanismvsinitiation distinguished;
  NP-dependenttwoDclosuresprecedebigNUFreset,96.90%net170excessTTTafterstep26.
- NEXTPLAN docs/rl_budget_nuf_retention_probe_20260930.md and its OWN SDDledger.
  One-variableNUFretention, first170thenremaining4onlyifvalidated/sufficient;
  no newlearner yet. Thiscollectionplan's code/data stayfrozen/completed.
