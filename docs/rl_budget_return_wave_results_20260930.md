# Shared-policy on-policy training wave

Completed all five75-step/14400s trajectories, same frozen actor, no exploration
or within-wave learning. Integrity and prospectivehealthscreens PASS. No canonical
evaluation or matchedcarry comparison: newtrainingprofiles have seeds7301..7305.

| Scenario | Actual TTT | Terminal inventory | Peak inventory | Distinct controls | Decision seconds | Worker-session seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 155 | 3988.486017 | 420.094125 | 2946.020780 | 71 | 893.017440 | 1013.473183 |
| 170 | 4262.399108 | 414.943501 | 3335.288981 | 42 | 992.279401 | 1111.235445 |
| 170 incident | 5652.260310 | 414.477843 | 4545.974326 | 61 | 874.118568 | 986.914695 |
| 170 skew | 3992.545990 | 414.792208 | 3092.796305 | 59 | 908.762180 | 1023.602125 |
| 190 | 6791.023647 | 413.375862 | 5217.874577 | 69 | 1073.917022 | 1186.277451 |

One initialPFO, zero recoveryPFO,75candidate solves per scenario. Fallbacks2/2/3/2/2;
nozeroNUFrequests orsimultaneousD-rampclosures. Actor saturationabsent, but ALL375
NUFrequests are projected to6000; requestedNUFdriftzero. NPmaximumrequestdrift
88.584/92.200/133.130/87.892/90.183, including physicalfallbackanchor changes.
Distinctphysicalcontrols do not by themselves prove a causalactor effect versus
carry. The actual newpolicy samples lack NUFrequestvariation; preserve previous
localexploration data rather than pretending on-policy data identifies this axis.

Total actor inference per75decisions is0.02957/0.03049/0.02574/0.02455/0.02789seconds;
decisionmedians8.77/11.83/10.20/9.11/14.49seconds. Inference is not end-to-end
latency. WindowsprocessCPU quantization can reportzero for shortactorcalls.
Worker-session excludesfinalpublication/interpreter/parent/readout, includes
checkpoint/validation/bootandrestore. Parallelworker seconds are not wavewalltime.
155prefix/resume retainsfirstinterval; totalrestorecall.2310865s/.203125CPU is
alreadyinitsworkersession, notanaddedchargeorisolatedreference-onlymeasurement.
Inheritednonterminalreference reconstruction is explicitlydisclosed in theplan.

## Provenance and decision

Source: work/sdmpc_rl_return_wave_20260930 (frozen after95tests and reviewPASS).
Output: results/sdmpc_rl_balanced_goal_20260930/return_policy_wave_v1.
ModelSHA7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904.
Readonly fullreadout saved to
.superpowers/sdd/rl_budget_return_initialized_policy_20260930/task2-readout.json,
SHAe8d2795c3828069d88f83797d1b644d3cba9f87f58d428b84ee387601777f1db.
No physicalcheckpointdeserialization inreadout. Allourworkersdrainedbeforeit.
FreshCIMfoundonlyunrelatedNumericalSimulationlauncher51192/worker43940; untouched.

Admit Task3's predeclared250critic-only updates from these375actual transitions,
equal20%scenario. Realremainingreturns label Q for THIS actor's recordedcontinuation,
notQ-star and not alternativeactions. FixedPhi/actor preserved; previouscritics
were Q-carry, so their discrepancy is notindependentQ-pi1calibration. The250updates
are fitting, not a freshheldouttest. NUFcoverage limitation remains explicit.
Afterreadout decide nextsmallpolicyupdate orcanonicaltest; goalACTIVE/unachieved.
