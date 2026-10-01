# Review actor recovery and evaluator

Read docs/rl_budget_recovery_plan_20260929.md tasks1/3/4 and the supplied diff.
Repo C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL. Own only
actor-review.md in this brief's directory; no production edits, no simulations.
Review bounded new common.py, actor_repair.py, evaluate.py and their two tests.
Prior code/read-only helper contracts may be inspected as needed. Reviewer must
return explicit SPEC and QUALITY verdicts with actionable file/line findings.
Do not restart a review of immutable old implementations or rewrite the plan.

Evidence:12focused tests PASS2.57s. Actor fit already completed,7.36s process,
results/sdmpc_rl_recovery_20260929/actor_fit_v1. Three375update arms with frozen
critic and identical balanced minibatches. All recover NUF positive; original
actor+optimizer continuation wins fixed pre-evaluation Q1 selection. Export hash
bf771c0dd08a61ec0a29f5ccd78beb52a11c23430b9f039c07c9030eca3545d7.
Old model remains immutable. No traffic claim. Full evaluation not launched;
coordinator will run two short serialized-resume parity smokes then full five
canonical runs only after load-bearing review issues resolved.

Review concrete risks: truly frozen critic/common sampling, honest export versus
resumable TD3 checkpoint, runtime/source/model/scenario identity, STOP/checkpoint
and duplicate behavior, same physical/reward/accounting contract, no evaluation
training, inclusion of inference/follower costs. Numerical performance is not
guaranteed. Sidecar critic_diagnostic.py is a different task and not in this diff.
Report findings once; fix rounds will be scoped to findings. No code changes or
test reruns if supplied evidence already covers current files.
