# Completed local-budget collection

Status: collection complete; predeclared coverage screen FAILED for170.
The shared-policy improvement goal remains ACTIVE. No actor training or
canonical learned-policy evaluation is admitted from this batch yet.

## Evidence and scope

Root: `results/sdmpc_rl_balanced_goal_20260930/local_budget_v2`.
All10trajectories completed75actual control intervals/14400s;750transitions,
150per scenario. Paired carry/local use training seeds6801..6805 and local
exploration seeds6901..6905. These are perturbed TRAINING profiles, not canonical
evaluation or deployment results. Carry here means previous-executed-budget
zero-action behavior, not P-Stack or an independently run PFO leader.

Original validators and the export companion reconciled all outputs. Root and
supervisor status are completed; all15supervisor attempts exited (8wave,7export),
plus the earlier manual155export. Independent process check found no remaining
Python worker. Saved250prefix transitions were resumed, never recollected.
All8diagnostic-only exports retain original raw checkpoints and explicit inf
tags/receipts. Physical feasibility, reward, terminal and TTT contracts unchanged.

Completion SHA256:
`5b6e645f6f6ca64c4f593c52d19ee3ab5bda7ae3c3d31f423d6fae2d2f3ef4a3`.
Comparison SHA256:
`c294c1f5b2100daeaccb4c927357a8bc73d44391f34641fd37b4ba5b2555c4cd`.

## Paired training comparison

Positive TTT change is worse. Inventory is the frozen physical inventory
accounting. Percentages do not compare to canonical goal thresholds.

| Scenario | Carry TTT | Local TTT | TTT change | Carry inventory | Local inventory | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
|155|3210.204341|3262.269009|+1.62185%|412.509438|415.839694|PASS|
|170|4602.411471|11945.939293|+159.55826%|413.976118|2966.221803|FAIL|
|170-incident|5570.093987|5497.152024|-1.30953%|417.623289|419.321172|PASS|
|170-skew|4618.714187|4624.132560|+0.11731%|423.151374|417.501756|PASS|
|190|6589.493773|6861.870165|+4.13349%|426.668592|423.755549|PASS|

All10runs had zero zero-NUF-request intervals. The170terminal inventory is
7.16520times its paired center, above the predeclared1.25limit. Eliminating
zero-request collapse did not establish bounded state/response coverage.
Local physical fallback counts8/7/6/4/7 versus carry3/2/4/2/3. Local unique
physical-control vectors33/68/69/55/65 versus carry35/52/51/53/69.

Known locked-worker-session elapsed sums:13644.690749600064seconds across all10
trajectories (parallel worker sums, NOT elapsed campaign wall time). Additional
successful export-defined scope53.19415919983294seconds. Both scopes exclude
specified startup/finalization/coordinator costs; do not present their sum as
end-to-end runtime or RL inference speed. Per-run/session and supervisor attempt
timestamps remain in the output. No timing observations were silently zeroed.

## Trace-supported mechanism, not a causal ablation

All following indices are zero-based control steps on local170.

1. Initial NUF base6000. Local policy clips offset to plus/minus500 around its
   CURRENT base, then rebases BOTH coordinates to the executed budget whenever
   physical fallback occurs. A moving base has no episode-wide bound.
2. Fallback9 changes NUF base to5758.248869; fallback12 to5361.154284.
3. At14 the requested NUF ceiling is4961.061339, but the feasible follower
   response achieves2840.800684. Both D ramps are effectively zero, while the
   F ramps remain about1418.32/1422.48. Several urban greens are at20/92.
   Thus physical-control degradation precedes the largest base change.
4. At15 the request is5081.775497. NP feasibility fails; physical fallback
   carries the previous controls. Its freshly evaluated reference budget is
  2840.800684, which becomes both executed budget and new exploration base.
5. The local base remains2840.800684 until fallback26, then3249.983841 through
   the end. Requested NUF drift reaches3349.333418 from the initial6000 even
   though the local offset rule remains inside its moving500limit. Total TTT
   divergence and residual inventory accumulate through the rest of the run.

Source: frozen `work/sdmpc_rl_local_20260930_v2/exploration.py`, `_choice` and
`commit`; `work/sdmpc_rl_multi_20260929/budget_controller.py`,
`prepare_reference` and `select_and_commit`; the paired170trace files.
NUF is an upper budget constraint (`G <= B`), not a command guaranteeing that
achieved flow equals B. Do not conflate B_requested, B_executed and G_achieved.

Confirmed: the proposed local exploration is not episode-locally bounded once
fallback rebasing is included. Not yet confirmed: how much loss is caused by
that rebasing versus the earlier NP-sensitive, discontinuous follower response.
No claim that a new collector or more learning will necessarily improve TTT.

## Next bounded investigation

Keep all750transitions, but do not start actor training on the failed screen.
Finish a read-only, source-backed drift/response decomposition across allfive.
Form a one-variable counterfactual test of NUF base retention, keeping NP
rebasing, noise draws, physical follower and all accounting unchanged. A stable
NUF desired center would still be converted through the ACTUAL carried anchor
and existing rate limits; it must not overwrite physical execution or bypass
feasibility. No such experiment has been admitted or launched yet.

Check whether removing NUF base ratcheting is sufficient; it may not prevent
the earlier near-zero ramp response. If a small comparison batch is justified,
reuse the completed paired carry references rather than collect them again.
Never reinterpret frozen-trace hypothetical actions as actual network rollouts.
The equal20percent scenario rule applies to every future learner minibatch;
there is still one shared state-conditioned policy/value model, not5models.
