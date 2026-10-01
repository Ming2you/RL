# First return-initialized shared proposal

Task1 completed once, no canonical evaluation yet. Immutable output:
results/sdmpc_rl_balanced_goal_20260930/return_init_v1.
ModelSHA7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904.
CompletionSHA34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757.
All completion output hashes verified; actual process39768 exited. Launcher33304
at10:08:57KST, creationFILETIME134352041372938852 foractualprocess.

Updates: Phi1000, residualcritics250, actor10, Polyak125; one seed7200 and one
shared actor/twinQ. Each optimizer batch contains exactly8samples/scenario.
Input pool750trainingtransitions,150/scenario, no canonical observations/outcomes.
Independent reviewPASS after one scoped STOP/finalization fix;39testsPASS.
All old353source/artifact hashes remained unchanged during testing/authentication.

## Numerical evidence

Pooled carry-Phi training MSE916.554503 ->202.306715, satisfying the prospective
50%reduction/per-scenario-decrease gate. Per-scenario final MSE:
97.214459/135.926780/233.614560/142.493708/402.284070.
This is return-scale initialization on training carry trajectories, not precise
calibration, heldout generalization, action-ranking proof or a traffic gain.

Residual critic errors remain material: carry full-horizon Q1 MSE
69.866104/135.628476/251.183862/142.128657/454.607706. Last-action Q1 absolute
errors on carry are approximately6.26/6.54/11.06/6.91/5.04scaledrewardunits.
Local behavior-return discrepancies are not Q-carry calibration targets; true
terminal rewards are exact. This candidate is admitted only as a bounded
training intervention, not as a validated value estimator or successful controller.

On750savedtrainingstates, actor nominal NP ranges0.003603127..0.004211580 and
NUF0.001734511..0.002029082; no actor saturation. This corresponds to requested
increments approximately0.180..0.211NP and1.735..2.029NUF before capacity projection.
433requests hit the NUFcap. The output is small and nearly uniform; actual control
diversity and long-horizon consequences must be observed, not inferred from loss.
Critics still estimate carry continuation after their initial action, not the new
actor's continuation. There are no target-policy-Q accuracy claims.

Locked training session55.395505seconds, including training/checkpoint/publication
inside execute; excludes prior authentication/data loading/interpreter startup.
Do not call this end-to-end experiment runtime. Parent logs/sessionidentity are
in .superpowers/sdd/rl_budget_return_initialized_policy_20260930/ and output/sessions.

## Next

Implement/review Task2 of rl_budget_return_initialized_policy_20260930.md in
work/sdmpc_rl_return_wave_20260930, leaving Task1sourceandmodel frozen. Run only
one new-profile episode/scenario (seeds7301..7305), no learning/exploration during
the wave. Health checks remain those predeclared in the plan. Real G-pi1 can later
train the shared critic for that continuation; old Q-carry disagreement is not a
reason to relabel these outcomes as optimal or compare unmatched profiles.
