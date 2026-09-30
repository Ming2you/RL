# Task 1 re-review: fix round 1

Reviewed: 2026-09-29. Scope: R1-R3 from the initial Task 1 review and new breakage introduced by their fixes.

## Verdicts

- **SPEC: PASS** for this scoped re-review. R1, R2, and R3 are addressed.
- **QUALITY: PASS** for this scoped re-review. No actionable remaining finding or newly introduced regression was found in the fixes.

## Finding disposition

### R1: ADDRESSED

Effective optimizer settings are validated at [td3.py:241](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:241). A fresh Adam supplies the required parameter-group schema and values, so validation does not trust either checkpoint spec metadata or potentially altered recipient optimizer settings. Both saved groups must match the required option keys, value types, values, and ordered parameter IDs. A changed learning rate such as zero is rejected before either optimizer is installed.

Both optimizer validators are called at [td3.py:364](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:364), before network loads, optimizer loads, or replacement of learner RNGs and counters. Parameterized tests at [test_td3.py:624](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:624) cover both optimizers and changes to learning rate, betas, epsilon, weight decay, AMSGrad, maximization, and execution options. Their shared rejection helper verifies the recipient remains unchanged.

### R2: ADDRESSED

Optimizer history validation at [td3.py:266](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:266) requires the exact parameter-state set after scheduled training begins, and an empty history before the first scheduled step. Each populated entry must have the required Adam fields, matching parameter/moment shapes, finite CPU floating-point scalar steps, finite CPU float32 moments, and nonnegative second moments. The critic step is checked against `updates`; the actor step is checked against `updates // policy_delay`. Missing history and a changed update count therefore fail before state installation.

Tests at [test_td3.py:637](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:637) cover missing/extra history and fields, malformed moments and steps, and parameter mapping errors for both optimizers. The original changed-update-count case has a direct regression at [test_td3.py:717](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:717). Premature histories are rejected by tests at line 723. Existing exact-continuation coverage at [test_td3.py:427](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:427) now explicitly checks legitimate empty histories and scheduled counts at zero through three updates, while retaining weights-only serialization, subsequent update/action parity, RNG parity, and snapshot independence.

### R3: ADDRESSED

After replay validation, [td3.py:361](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/td3.py:361) rejects cumulative per-scenario samples above `updates * min(restored_group_lengths)`. Earlier integer, equality, and lower-bound checks remain. This closes both the impossible 4,000-sample history and positive training history with an empty replay group.

Tests at [test_td3.py:734](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:734) reject 17 and 4,000 samples after two updates with eight stored transitions, and reject a smaller or empty limiting group. The positive boundary test at [test_td3.py:746](C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL/work/sdmpc_rl_multi_20260929/test_td3.py:746) accepts the exact limit with unequal replay sizes and checks identical continuation.

## Regression assessment

The difference between the original and round 1 carry-to-multi diffs is confined to the optimizer validator and the new pre-installation checks. Initialization, equal scenario sampling, replay eviction, target computation, loss/optimizer steps, delayed actor/target updates, and checkpoint copying/RNG restoration are unchanged by this fix round.

The stricter validation preserves legitimate zero-update and pre-actor-update checkpoints, parameter ordering, and the replay bound's inclusive boundary. The helper is shared by the two optimizers and the added tests exercise rejection behavior and recipient preservation. No concrete new concern warranted a numerical rerun.

## Verification and limits

Read the initial review, the fix-round addendum in the implementer report, both supplied diffs, current restore implementation, and relevant current tests. Recomputed the carry-to-current diff and compared it with `task-1-fix-round1.diff`: exact match after line-ending normalization. Git's no-index diff exit code was 1 because the compared source files differ.

Accepted the implementer's reported **211 tests passed in 4.87 seconds**, including 89 added cases, as execution evidence. No tests or numerical probes were run during this re-review. Thus numerical execution is implementer-reported; the dispositions above additionally rest on direct code and test inspection.

Reviewed source SHA256:

```text
work/sdmpc_rl_multi_20260929/td3.py
E27665133CCB3FAC2B1769DA1C93DF73A60C73572FA4CCEDE24A1296292E2E8E
work/sdmpc_rl_multi_20260929/test_td3.py
18FBD3BC93534E1ED1DCF9218A3C6D114B0D760C0DEE794F217FB392B689A3D0
```

Only this new re-review report was written. Production and test files, the earlier review, and the implementer report were not edited. Independent runner/collector files and arbitrary corrupt network-payload recovery remain outside this scoped re-review. No traffic simulations or commits were performed.
