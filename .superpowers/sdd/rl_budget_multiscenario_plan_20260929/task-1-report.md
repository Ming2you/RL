# Task 1 report: balanced shared TD3 learner

Status: complete. The focused synthetic suite passed all 122 tests in 4.51 seconds.

## Scope and changed files

All paths below are relative to `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

- `work/sdmpc_rl_multi_20260929/td3.py` (new)
- `work/sdmpc_rl_multi_20260929/test_td3.py` (new)
- `.superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-1-report.md` (new)

Requirements came from `task-1-brief.md` and the user's clarification about the
module-level tuple and central trainer ownership. The numerical core was ported
directly from `work/sdmpc_rl_carry_20260929/td3.py`; relevant tests were ported from
`work/sdmpc_rl_budget_20260929/test_td3.py` and extended for balanced replay.

No collector, runner, prior experiment source, or experiment results were edited.
No conversation history was inspected, agents spawned, commits made, packages
installed, ACLs changed, or traffic simulations launched.

## Implementation and integration contract

- `from td3 import SCENARIOS` exports the tuple in this fixed order:
  `sweet_155_w`, `sweet_170_w`, `sweet_170_incident_w`,
  `sweet_170_skew15_w`, `sweet_190_w`.
- `TD3(observation_dim, seed, hidden=64)` retains the prior network initialization,
  local random generators, one Torch thread, CPU float32, one shared actor and two
  shared critics. Scenario metadata does not extend any network input.
- `act(observation)` retains a copied, bounded NumPy float32 vector of shape `(2,)`.
- `add(obs, requested_action, reward, next_obs, terminated, scenario)` requires a
  valid scenario string and true boolean termination. Numeric input shape, finite
  float32 representability, original-precision action bounds, copied arrays, and
  unclipped reward handling remain as in the carry core.
- Each scenario owns a FIFO deque capped at 2,000 entries. In the new learner,
  `replay_capacity` and `spec()['replay_capacity']` mean capacity per scenario.
  Total maximum storage is 10,000. Private storage is `_replay`; there is no global
  `.replay` API. Public `replay_counts()` returns a fresh scenario-to-count dict;
  `total_transition_count()` returns the sum of stored counts after eviction.
- `update(batch_size=40)` rejects nonpositive, noninteger, boolean, or nondivisible
  batch sizes. It returns `{}` without changing any state until all five groups
  have at least `batch_size / 5` entries. It then samples with replacement from
  each group using the checkpointed NumPy generator, in fixed scenario order.
- Default updates train on exactly eight samples per scenario. All samples enter
  one combined twin-critic loss and optimizer step. The delayed actor uses the
  same balanced observations. TD3 gamma 1, learning rate 3e-4, tau 0.005, smoothing
  0.2 clipped at 0.5, and delay 2 are unchanged.
- Successful update metrics include `critic_loss`, `updates`, and
  `sampled_per_scenario`; delayed actor updates also include `actor_loss`.
  Cumulative sampled transition counts are available as `learner.sample_counts`
  and checkpoint key `sample_counts`. These counts include replacement draws.
- The frozen collector needs only `act` and checkpoint restoration. Only the
  coordinator's central trainer should call `add(..., scenario=...)` and
  `update(40)`, as instructed by the user.

## Checkpoint contract

Format is `sdmpc-multi-td3-v1`. The checkpoint contains the full specification,
observation dimension, four network state dictionaries, two optimizer state
dictionaries, local NumPy and Torch RNG states, update counter, cumulative sample
counts, and independent replay copies.

The fixed ordered scenario list is recorded in `state['spec']['scenarios']`.
`state['replay']` is a dict keyed by all five scenario IDs, with columns:

| Column | Shape | CPU dtype |
| --- | --- | --- |
| `observations` | `(n, observation_dim)` | float32 |
| `actions` | `(n, 2)` | float32 |
| `rewards` | `(n,)` | float32 |
| `next_observations` | `(n, observation_dim)` | float32 |
| `terminated` | `(n,)` | bool |

Empty groups preserve these shapes. Checkpoints support weights-only
`torch.load`. Exported and restored tensors, arrays, optimizer state, RNG state,
and sampling dictionaries do not alias their source snapshots.

Restoration rejects the prior format, mismatched specifications or dimensions,
wrong scenario order, unknown/missing/duplicate listed scenarios, unknown/missing
replay or sample-count groups, malformed replay columns, excess per-scenario
capacity, wrong tensor types/dtypes/shapes, nonfinite values, out-of-bounds
actions, and invalid update/sampling counters. Sample counts must be nonnegative
integers, equal across groups, and consistent with the update count and maximum
possible batch contribution. Seed is validated and restored, allowing a receiver
constructed with a different initial seed, as in the prior core. Replay,
specification, counters, and local RNGs are validated before installing state.

## Validation commands and results

All commands ran with the repository path above as the working directory.

### Focused numerical suite

Exact PowerShell command:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
& '.venv-torch/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --confcutdir=work/sdmpc_rl_multi_20260929 work/sdmpc_rl_multi_20260929/test_td3.py
```

The initial sandboxed launch exited 1 before test collection:
`No module named pytest.__main__; 'pytest' is a package and cannot be directly executed`.
The following read-only diagnostics confirmed sandbox access denial for the
existing dependency files:

```powershell
Get-Item -LiteralPath '.deps-budget/pytest/__main__.py','.deps-budget/pytest/__init__.py' | Select-Object FullName,Length
Get-Content -LiteralPath '.deps-budget/pytest/__main__.py'
```

Both diagnostic calls reported access denied. The exact same test command was
then run with sandbox escalation, as allowed by the brief. It exited 0:

```text
........................................................................ [ 59%]
..................................................                       [100%]
122 passed in 4.51s
```

Only this one process executed numerical tests. The earlier failed launch never
collected tests. No numerical processes were launched concurrently. Bytecode
writes and pytest cache writes were disabled, including when the parity test
read/imported the prior carry source.

### Behavioral coverage

- Five-group warmup, including each possible underfilled group, and no update
  from a single populated group even with the smallest valid batch size.
- Actual critic batches with group sizes 8, 13, 47, 211, and 997: exactly eight
  per group by default, observed repeated samples, and equal delayed actor
  observation batches. Custom divisible batch sizes and cumulative counts.
- Equal FIFO capacity and full wraparound for all five groups, minority-group
  preservation under majority overflow, storage counts, and wrapped replay
  checkpoint continuation.
- Independent source input copies and returned count/metric dictionaries.
- Original architecture, zero final actor layer, action bounds, gradient-free
  action calls, target twin minimum, clipped target smoothing, unclipped rewards,
  requested actions, and true terminal masking without cross-episode bootstrap.
- One shared critic optimizer step per update, delayed actor/target updates,
  exact Polyak averaging, and unchanged global NumPy/Torch RNG state.
- Direct numerical parity against the unchanged carry core on identical sampled
  batches: losses, online/target parameters, both optimizers, target-noise RNG,
  and actions match exactly across both delayed and nondelayed update phases.
- Save/load at zero through three completed updates, next-action parity, three
  subsequent updates of different sizes, exact optimizer/RNG/counter parity,
  weights-only serialization, and checkpoint mutation independence.
- Strict constructor, observation, action, reward, termination, scenario, and
  batch-size validation; malformed checkpoint scenarios, replay, and counters
  rejected before mutation of the recipient learner.

### Source comparison and preservation

```powershell
git diff --no-index -- work/sdmpc_rl_carry_20260929/td3.py work/sdmpc_rl_multi_20260929/td3.py
Get-FileHash -Algorithm SHA256 -LiteralPath 'work/sdmpc_rl_carry_20260929/td3.py','work/sdmpc_rl_budget_20260929/test_td3.py' | Select-Object Path,Hash | Format-List
```

The comparison command exited 1 because the files intentionally differ. Its
diff confirms that target computation and the TD3 optimizer math are preserved;
changes concern replay grouping, sampling, counters, and checkpoint handling.

Prior source SHA256 values recorded before implementation and rechecked after
the tests:

- Carry `td3.py`: `746B9C6C45ABA37CF9DC169274CB64A22EC51FCB1B2C5DC06F2D7DD3104EE1B3`
- Budget `test_td3.py`: `F238BCBC92C193C63B97AA14C958B8DA3E8C31E9CF7FD6C6366AE24B598CA01D`

The repeated hash command exited 0 and both hashes matched exactly.

Final scope and whitespace commands:

```powershell
git status --short --untracked-files=all -- work/sdmpc_rl_multi_20260929/td3.py work/sdmpc_rl_multi_20260929/test_td3.py .superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-1-report.md
git diff --no-index --check -- NUL work/sdmpc_rl_multi_20260929/td3.py
git diff --no-index --check -- NUL work/sdmpc_rl_multi_20260929/test_td3.py
git diff --no-index --check -- NUL .superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-1-report.md
```

Scoped status exited 0 and listed both new Python files as untracked; the report
exists in the repository's `.superpowers` area and was not listed by status.
Git reported inability to read the user's global ignore file in the sandbox.
The three no-index whitespace checks emitted no whitespace errors and returned
1 for new-file differences. The report check additionally emitted Git's existing
LF-to-CRLF conversion warning. No Git staging or configuration changes were made.

## Concerns and remaining integration work

No Task 1 blocker remains. Tests are synthetic; collector/round-runner integration
and traffic behavior were deliberately outside this task and remain with the
coordinator. Running the suite in this environment requires dependency read
access through sandbox escalation. Restore validation for network and optimizer
payloads otherwise follows the prior core's PyTorch loading behavior; this task
does not add transactional recovery for arbitrary corrupt network/optimizer
payloads.

## Fix round 1: R1-R3 (2026-09-29)

Status: all three findings addressed. The focused suite now passes **211 tests
in 4.87 seconds**, including 89 added negative and boundary regressions. The full
`task-1-review.md` was read before editing. This round changes only the two owned
Python files and appends this report; it does not change the production collector,
round runner, prior experiment source, or results.

### R1: effective Adam settings

Added `_validate_optimizer` and invoke it for both optimizers before any network
or optimizer state is installed. Each saved parameter group must exactly match a
fresh Adam constructed with the same parameters and learning rate as the learner.
This validates the actual learning rate, betas, epsilon, weight decay, AMSGrad,
maximization, execution options supported by the installed Adam, group structure,
and ordered parameter IDs. Missing, extra, mistyped, or conflicting options are
rejected even when checkpoint `spec` is intact. Using a fresh reference avoids
trusting potentially modified settings on the recipient optimizer.

The regressions mutate each optimizer's settings, including zero learning rate,
and assert rejection without changing the recipient learner.

### R2: coherent optimizer history

Critic history must contain exactly one state entry per optimized parameter when
`updates > 0`; each Adam step must equal `updates`. Actor history follows the same
rule with `updates // policy_delay`. Before each optimizer's first scheduled step,
its history must be empty. Thus both histories may legitimately be empty at zero
updates, and actor history remains legitimately empty after one critic update.

Every populated entry must contain exactly `step`, `exp_avg`, and `exp_avg_sq`.
Moments must have the matching saved network parameter's shape and CPU float32
dtype, be finite, and have nonnegative second moments. Saved parameter shapes must
also match the receiving network. Steps must be finite scalar CPU floating-point
tensors with the exact scheduled count. Missing or extra entries, reordered or
duplicated parameter IDs, malformed moments, and step/count disagreements fail
before installing the checkpoint.

Negative regressions cover both optimizers, including the review's empty-history
case and changed `updates` case, moment corruption, malformed steps and parameter
groups, and premature history before a scheduled step. Existing exact continuation
tests at 0, 1, 2, and 3 updates now also assert the expected empty histories and
step counts before restoration. They continue to pass.

### R3: sampling history bounded by replay

After replay validation, the loader additionally requires the equal cumulative
sample count to be at most `updates * min(restored_group_lengths)`. Existing integer,
equality, nonnegative, and update lower-bound checks remain. This uses the public
API's monotone replay lengths and rejects positive training history with an empty
group.

Regressions reject 17 and 4,000 samples per group after two updates with eight
stored transitions, and reject histories made impossible by one smaller or empty
group. A positive regression restores the exact upper bound with unequal group
sizes and verifies identical continuation.

### Exact validation command and result

Working directory: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
& '.venv-torch/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --confcutdir=work/sdmpc_rl_multi_20260929 work/sdmpc_rl_multi_20260929/test_td3.py
```

Run once with approved sandbox escalation for the previously confirmed dependency
ACL restriction. One numerical process, no concurrent numerical tests, no bytecode
or pytest cache writes. Exit code 0; complete output:

```text
........................................................................ [ 34%]
........................................................................ [ 68%]
...................................................................      [100%]
211 passed in 4.87s
```

The existing carry-core numerical parity, terminal handling, balanced batches,
eviction, exact checkpoint continuation, and global RNG isolation tests all pass.
No TD3 initialization, sampling, target equation, loss, optimizer update, delay,
or Polyak averaging math was changed by this round.

### Preservation and static checks

Exact commands:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath 'work/sdmpc_rl_carry_20260929/td3.py','work/sdmpc_rl_budget_20260929/test_td3.py' | Select-Object Path,Hash | Format-List
git diff --no-index --check -- NUL work/sdmpc_rl_multi_20260929/td3.py
git diff --no-index --check -- NUL work/sdmpc_rl_multi_20260929/test_td3.py
git status --short --untracked-files=all -- work/sdmpc_rl_multi_20260929/td3.py work/sdmpc_rl_multi_20260929/test_td3.py .superpowers/sdd/rl_budget_multiscenario_plan_20260929/task-1-report.md
```

Hash verification exited 0 and retained the previously recorded values:

- Carry `td3.py`: `746B9C6C45ABA37CF9DC169274CB64A22EC51FCB1B2C5DC06F2D7DD3104EE1B3`
- Budget `test_td3.py`: `F238BCBC92C193C63B97AA14C958B8DA3E8C31E9CF7FD6C6366AE24B598CA01D`

Both no-index whitespace checks returned 1 for file differences with no output or
whitespace errors. Scoped status exited 0, showing both owned Python files as
untracked and repeating the sandbox's global-ignore access warning. No files were
staged or committed. No packages were installed, ACLs modified, or traffic runs
launched.

No unresolved R1-R3 concerns remain. This addendum supersedes the earlier report's
statement that optimizer payload validation follows only PyTorch loading: optimizer
configuration and history now undergo explicit validation. Transactional recovery
for arbitrary corrupt network weights remains outside these three findings.
