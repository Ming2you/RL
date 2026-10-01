# Task3:250on-policy MC critic updates

Implement in NEW work/sdmpc_rl_return_mc_20260930/ and task-3-report.md only.
Output fixed results/sdmpc_rl_balanced_goal_20260930/return_mc_v1. Parent dispatch
after independentreview. Noactualoptimizationorphysics duringimplementation;
synthetictests and readonlyactualdata/modelauth allowed. Nocommit/install/ACL.

Task2actualwave COMPLETE/PASS,375transitions75/scenario,newtrainingseeds7301..7305.
Read docs/rl_budget_return_wave_results_20260930.md forlimitations. Readout stored
at .superpowers/sdd/rl_budget_return_initialized_policy_20260930/task2-readout.json,
SHAe8d2795c3828069d88f83797d1b644d3cba9f87f58d428b84ee387601777f1db.
Parentmodel results/sdmpc_rl_balanced_goal_20260930/return_init_v1/model_final.pt
SHA7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904,
completionSHA34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757.
Alltheseartifacts/source remainimmutable. Priorcriticcontinuationcarry is NOTpi1.

## Algorithm and data

One sharedPhi, fixedactor, twincritics. Same2367obs/signedNP/projectedrequested
budgetencoding; gamma1,r=-intervalTTT/100,true75terminal. No canonicaldata,
trainingprofilechanges,recollection,actorupdate,exploration,newPFOorpreview.
Authenticate actual5wave via existingreadonlyreadout in freshisolatedsubprocess
(noenvboot/reset/step/rawphysicalckunpickle), compare exactretainedreadout, bind
all5settings/completion/trace/experience/source/modelhashes. Parentrecorded
healthPASS mustbereconfirmed bysourcevalidations, nottrustedflagalone. Reuse
small existingreadonlyhelpers; do notcloneprocessorcollectionframework.

Loadonlythese375on-policytransitions,with exactfloat64anchors/requestfromtrace.
RemainingreturnG_t=sum_{k=t}^{74}r_k, excludeswarmup; truefinitehorizonterminal.
They are actual Q^pi1 samples atobserved(s,b_pi1), NOTQstar orotheractionlabels.
All375NUFrequests6000, noNUFactionvariation: document identifiabilitylimit.

InitializePhi/actor/critics EXACTLYfromparent. Reuse architecture/projectedbudget
purehelpers inofflineprocess ifcompatible; avoidimporting physicalruntime.
FreezePhiandactorbitexact, nooptimizersforthem. CriticAdam continues its matching
parentoptimizerstate (includinglr3e-4) ratherthanclaimingfreshresume. New dedicated
NumPyPCG64seed7201 forsamples, checkpointRNG. Do notconsumeunknownglobalRNG.
Exactly250criticoptimizersteps,batch40:8/scenario (1terminal+7uniform with
replacement) fromall75ownrows. Bothresidualheads optimized againstG_pi1-Phi(s),
MSEsum overheads, nobootstrapping/targetnoise, nooldcarryanchorloss. Thisis actual
on-policyMonteCarloevaluation/fitting, not unchangedTD3orpolicyimprovement.
Nooutputscale/architecturechange inthisphase. Noextraepochs/seedsearch/adaptive
stopping. After250steps settargetcritics exactcopyofupdatedcritics forfuturework;
therewerenotargetupdatesduringMCfit. Noactorproposalorcanonicaldispatchhere.

Capturebefore/afterbothheads+minQperrowandscenario/horizon, terminalrewarderrors,
MSE/MAE/signedbias/maxerror, action/projectioncapcoverage. BeforemodelQ-carryvs
G-pi1 is continuationdiscrepancy, not independentQ-pi1calibration. Afterfiterror
isTRAININGfit, noheldoutclaim. RecordPhi/actorhashes unchanged, parentandnewcritic
hashes, exactsamplecounts2000/scenario andforcedterminalcounts250/scenario,
per-updateloss/balance and finitechecks. Rejectnonfinite/provenanceviolations;
don'tgateorretunebasedonnewarbitraryerrorcutoffs. Parentreviewactualfitbefore
anynextaction. Exportonefinalcheckpointwithnewexplicitformat, own250MCcounter
separatefromancestor1000/250/10, models/tiedtargetpolicyhash, optimizer/RNG/data/
source/specidentity, continuedcriticoptimizerprovenance. ActorandPhiunchanged
meansphysicalpolicyunchanged; neverclaimcritic-onlyfit improves trafficTTT.

## Execution and tests

One singlethreadjob; parentchecks8globalbudget (lastoneunrelatednumericalworker).
STOP repo/goal/output checked, kernel lock, actualprocessidentity, atomic
checkpoints/metrics/settings/finiteJSON/completion/hash. Resume fromlastdurable
updatewithsameRNG/optimizer, preserveorphans, refusecompletedduplicate/mismatch.
No forcedretraincompletedphases, nostalehashafterfinalizationSTOP (Task1R1lesson).
Mayreusesmallfrozenruntimehash/serialization/lockinghelperwherefit; don'tcopya
largegenericframework. Finalcheckpointmustbindsameactorasall5collectedepisodes.

Testactualreturnconstruction/terminal/warmupexclusion, float64projectionparity,
equalbalance/quota, critic-onlygradients/frozenexactweights, continuation/provenance
labels, MC(no bootstrap)target, deterministicresume incloptimizer, source/model/
datahashfailures, STOPandfinalpublication/completedrefusal. Syntheticarraysfor
optimizer tests; readonlyrealvalidation doesnotauthorizeoptimization. Do not
rerun earlierwholelearner/wavesuites. Reportexactcommands/tests/hashes/limitations,
changedpaths andactualruncommand. Parentindependentreviewbeforeactualjob.
