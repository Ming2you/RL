# Task1 fix round1

Read task-1-review.md R1 verbatim; parent accepts this scoped P2 finding.
Fix only STOP/failure propagation during actual-worker drain in the new runner
and covering tests. No oldsource/results changes or productionrun. Before-fix
snapshots are in before-fix1. Use apply_patch; do not change treatment/scientific
gates or add generalprocess infrastructure. No commits.

Known nonzero launcher exit must trigger attemptABORT before waiting on its
still-live numericalchild. During normal polling of a successful exitedlauncher
whose actualworkerisstillalive, remain responsive to all-slotSTOP and otherchild
failure. Onceaborting, drain all knownlaunchers/actualworkers, retainlive/UNKNOWN
ownership, never releaseearly or enableduplicate/retry. A nonblocking identity
probe in the ordinary polling loop may be simpler than blocking there; choose
the smallestcorrectchange. Preserve finaldrainingchecks andfailclosedUNKNOWN.

Reproduce the reviewedcase then add regressions for nonzeroexit, later sibling
STOP, and anotherchildfailure while the firstactualworkerremainslive; assert
earlyABORT, allsiblingdrain and noreleasebeforeactualdeath. Cover changed
run_wave.py and test_workflow.py, then rerun69+focusedcandidate suite to refresh
sourceidentity evidence (no frozen suites). Append commands/results/limits to
task-1-report.md under FixRound1. Returnbrief; parent scoped re-review before
anyreal170pilot. Current candidate root remainsabsent andallRLworkersdead.
