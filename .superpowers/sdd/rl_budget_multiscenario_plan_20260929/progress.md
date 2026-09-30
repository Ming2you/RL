# SDD ledger - plan: docs/rl_budget_multiscenario_plan_20260929.md

Windows adaptation: plan-local workspace created with apply_patch. Existing
feature branch has staged/untracked prior implementations; no reset, branch
switch, commits or scratch deletion. Review the new versioned files directly.
Prior plan ledger and pinned experiment source remain untouched.

- Task1: pending balanced learner and contract verification.
- Tasks2-6: pending. No numerical experiment launched.
- Task1: implemented balanced learner; focused122tests PASS. Independent review
  found3P2strict-restore issues (effective Adam settings, complete Adam step/history,
  impossible sampling totals). Fix round1 assigned to implementerLeibniz
  01a0ece5-a23a-7490-b787-92f9aef8a665; reviewerEinstein
  01a0ecf1-b2ae-7722-9d02-fab861d6b2e9. No traffic admission yet.
- Task1 physical compatibility: allfive frozen protocols verified; BudgetEnv
  contracts identical before reset; initial-state hashes identical, forecasts
  separately pinned. `FIVE_COMMON_PHYSICAL_CONTRACTS_PASS` (no plant interval).
- Task2: coordinator implemented frozen collectors, shared round trainer,
  five-worker lifecycle/gates and read-only full-run analysis outside pinned source.
  Synthetic integration tests owned byMcClintock01a0ecf2-6006-7522-9980-9963529fa0a6;
  production code still under review, no simulator worker launched.
- Task1 fix-round1 implemented211testsPASS4.87s; scoped re-review byEinstein pending.
  Leibniz is closed (resume same ID if further fixes needed). Original and fix
  diffs preserved. Do not start traffic until both learner and lifecycle reviews pass.
- Integrated Task2 review active: Peirce01a0ecfd-4980-73a2-9bd6-651dc2132820,
  package integration-version.diff; code-only verdict first, then final evidence.
- New commands prepared: run_tests.py --output results/.../admission_v1 binds
  test XML/current source/test hashes; smoke_budget.py --scenario <each of5>
  --cpu-mask <1,4,16,64,256> --output <smoke_SCENARIO.json>. Run only after fixes
  settle; gate requires allfive actual serialized resume smokes plus final review.
- Read-only final analysis: work/analyze_sdmpc_multi_20260929.py (outside pins),
  syntaxPASS. Pilot outputs not created/launched yet. Legacy jobs/automations untouched.
- Task1: COMPLETE, fix round1 all3findingsADDRESSED; independent SPEC/QUALITY PASS,
  211focused tests. Implementer and reviewer now closed. No commit; versioned
  working-tree files and review artifacts retained. Task2 tests/review still active.
- Task2 synthetic suite COMPLETE:140tests,133PASS7FAIL in33.99s. Report
  task-2-test-report.md retains failing regressions for trainer resume seed/sample/
  replay drift, exported provenance not matching collectors, skipped experience
  payload not validated before siblings, and stage/child STOP appearing mid-launch.
  Test authorMcClintock is completed/open; may assign consolidated scoped fixes
  after Peirce's integration-review.md is finalized. No production fix yet.
- Peirce received test failure report and coordinator concern that verify_gate
  trusts an empty/partial evidence map. Await consolidated code review, then one
  fix dispatch followed by focused re-review and fresh full source-bound tests.
  Still no traffic worker or pilot output; no admitted experiment is active.
- Integrated review COMPLETE:6findings I1-I6 in integration-review.md (cached gate
  incomplete evidence P1; stale approval binding, trainer replay/provenance,
  mid-launch STOP, completed-payload validation P2). Single consolidated fix
  round1 assigned toMcClintock01a0ecf2-6006-7522-9980-9963529fa0a6 via
  integration-fix-brief.md. Same agent owns scoped production+test fixes now.
  Peirce01a0ecfd-4980-73a2-9bd6-651dc2132820 remains available for scoped review.
- Coordinator must NOT edit those active production/test files. Independent final
  analysis return-boundary checkPASS. Await integration-fix-report.md, then scoped
  re-review, fresh source-bound full suite, five actual smokes, and final machine
  attestation tied to exact source/evidence before creating gate/launching.
- Integration fix-round1 returned:199focusedtestsPASS56.82s, all7originalfailures
  resolved. Source-bound tests/gate now require typed9-role evidence plus a separate
  final reviewer JSON attestation; exact schema in integration-fix-report.md.
  Before scoped review, same fixer adding runtime_versions to actual smoke JSON
  and validator/tests, so final reviewer can bind observed per-smoke runtimes rather
  than infer them. No real smoke or learner episode launched.
- Runtime amendment complete, focused205testsPASS; fixerMcClintock now closed.
  Peirce scoped re-review pending. Fresh FULL source-bound evidence generated:
  results/sdmpc_rl_multi_20260929/admission_v1/{tests.xml,test_source_pins.json};
  486PASS52.69s, six test files,486collected/passed node IDs, exit0. Test exec
  session68798 finished0. Current source unchanged. RuntimePython3.12.14,
  NumPy2.3.5,SciPy1.16.3,Torch2.14.0+cpu,PyYAML6.0.3.
  Await CODE_REVIEW pass then actual five smokes (no numerical process active now).
- Scoped integration re-review COMPLETE: CODE_REVIEW/SPEC/QUALITY PASS, I1-I6
  addressed,486test identities independently reconciled. No source changed.
  No pre-existing numerical worker found. Five one-core actual serialized smokes
  launched with masks1,4,16,64,256; exec sessions82763,28256,43942,88772,5615
  correspond to155,170,170-incident,170-skew,190. Final admission still pending.
- All five smoke sessions exited0, SMOKE_RESUME_REPLAY_PASS2367. Canonical
  profiles match; each has physical/budget validity, PFO1initial/0carried and
  serialized parity. No numerical worker remains. Final admission evidence review
  assigned to same independent reviewerPeirce; awaiting finalized human report
  and exact sdmpc-multi-final-review-v1 JSON attestation. No pilot launched yet.
- Final independent admission APPROVED. Peirce closed after finalizing report and
  hash-bound attestation. build_preflight.py returned MULTI_PREFLIGHT_PASS486 5,
  gate admission_v1/preflight.json; no production/test changes after evidence.
- Task5 ACTIVE: one finite pilot launched via run_pilot.py --output
  results/sdmpc_rl_multi_20260929/pilot_v1 --gate
  results/sdmpc_rl_multi_20260929/admission_v1/preflight.json. Exec session21637,
  parentPID18160, startedUnix1790685737.5356002. Initial five numerical childPIDs
  15236,44592,8176,3032,26236 in scenario order. Stage0collect_round0 running.
  Do not duplicate runner/edit pinned sources or configuration. Per-step checkpoint
  and STOP support active. Next: await two rounds and both evaluation waves, then
  read-only work/analyze_sdmpc_multi_20260929.py and per-scenario outcome report.
- Round0 complete:5episodes/375transitions,75per scenario,375central updates,
  exactly3000draws/scenario. Training wall6.3425207s; model SHA256
  97eb5679948f9dbc4afb5f04bd77ddb5e7b31c3e36a684726560cc2e0b35a190.
  On the round0 replay, actor saturation(abs>=.95) is100% in both dimensions
  across allfive scenarios, approximately[+.9997,-.9997]. Exploratory behavior
  return gaps are NOT current-policy calibration. Stage2collect_round1 now uses
  this frozen common model plus exploration; do not alter it mid-run. Continue
  finite plan and diagnose canonical fixed-policy outcomes before conclusions.
- Tasks5/6 COMPLETE: exec21637 exited0 MULTI_PILOT_COMPLETE. All22jobs completed,
  10collection and10evaluation full episodes, about69.39minutes. Final model
  3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650;
  150transitions/scenario,2true terminals/scenario,750criticupdates/375actorupdates,
  6000cumulative draws/scenario. No evaluations in replay. Source unchanged.
- Read-only full22job reconciliation exec88197 exited0. Allfive RL TTTs are
  5.50-9.89times carry-center; all375canonical actions saturated,69final intervals
  of zero NUF/allramps per scenario,0RLfallbacks. Same-policy Q values severely
  optimistic; not an improvement/generalization/nonlinear-price claim.
- Additional read-only saved-model probes found actor/critic mismatch: finalQ1
  grid winner[+1,+1] and positive local NUFgradient on750/750replaystates, while
  actorremains[+1,-1]. NUFtanh slope~1e-4; strong gradientattenuation measured,
  but no causal rescue proved. Probe artifacts/scripts outside frozen source.
- Final result/limitations/next falsifiable diagnostics documented in
  docs/rl_budget_multiscenario_results_20260929.md. All six bounded plan tasks
  complete; performance rejected. No multi-pilot process remains. Unrelated
  calibration process left untouched; no legacy loop, scheduler, commit or push.
