# Carry controller implementation report

## Scope and result

Implemented the bounded controller brief and the controller portion of
`docs/rl_budget_carry_plan_20260929.md` directly in the shared workspace.
All 54 fake-runtime controller tests pass. No numerical simulations, native
controller runs, commits, branch changes, index edits, or schedules were run.

Files created by this task, relative to the RL repository root:

- `work/sdmpc_rl_carry_20260929/budget_controller.py`
- `work/sdmpc_rl_carry_20260929/test_carry_controller.py`
- `.superpowers/sdd/rl_budget_carry_plan_20260929/controller_report.md`

No other files were edited by this task. The frozen physical snapshot, v1
implementation, and coordinator-owned environment/runner/probe were read only.
Pre-existing staged, unstaged, and untracked shared-workspace changes were retained.

## Implementation and integration contract

- Constructor defaults to `guard_mode="physical"`; only `physical` and `h3` are
  accepted. No native mode or native candidate search exists in this version.
- `prepare_reference` clips the encoded control to current coordinate bounds
  before quantization and evaluates the mapped control with the original lower
  execution checks. Physical/budget check failures raise
  `InvalidReference(ValueError)` and leave the interval unprepared. Unrelated
  runtime exceptions propagate; repeated preparation is still an error.
- `reference["source"]` is `previous`, `pfo_initial`, or `pfo_recovery`.
  `controller.action_anchor` is an independent ndarray copied from the prior
  executed budget, or the initial reference witness if no prior budget exists.
  `reference["action_anchor"]` stores another copy for serialization. Restoring
  `lower.last_budget` before preparation reconstructs the same anchor.
- `evaluate_and_commit` accepts only `rl` and `center`, uses exactly one lower
  candidate, and rejects a previously evaluated interval. Center supplies zero
  action. Requests use anchor + `[50, 1000] * action`; only NUF is clipped to
  `[0, capacity]`. Signed NP is retained.
- V1 lower response, archive selection, solver request-identity checks, and
  transactional restoration of prices/prior budget are preserved. A successful
  final candidate commits its own dual. Archive recovery keeps incoming prices.
- Physical or budget failures select `reference_fallback`. Fallback executes
  the current validated witness budget, preserves incoming prices, and retains
  the failed request and lower diagnostics separately. H3-only deterioration
  triggers fallback only in `h3` mode.

Detached audit fields include `reference_source`, `action_anchor`,
`h3_guard_enabled`, and `h3_guard_would_reject`, matching the coordinator's
environment contract. `B_raw` retains the v1 two-component vector shape;
`B_requested` retains the v1 list-of-budget shape, with exactly one entry from
`evaluate_and_commit`. `B_executed` and `G_achieved` remain separate.

Additional diagnostics:

- `budget_clipping`: requested budget minus raw budget, including signed NUF
  clipping at either bound.
- `anchor_offset_from_reference`: carried anchor minus current witness budget.
- `request_offset_from_reference`: clipped request minus current witness budget.
- Top-level `h3_TTT_delta`: selected lower response TTT minus reference TTT,
  before fallback; `None` when no lower response can be selected.
- Each candidate's `h3_TTT_delta`: that candidate's TTT minus reference TTT,
  retained even when archive selection differs from the final solver point or
  there is no selectable lower response.

The H3 rejection flag describes the selected lower response before fallback;
candidate-specific deltas and lower-selection identity/checks remain visible.
Full detached solver rows, local rows, stationarity, multiplier evidence,
derivatives, counters, duals, and execution checks are retained without rollout
trajectory objects.

## Tests

Working directory: the RL repository root. Exact PowerShell command:

```powershell
$env:PYTHONPATH = '.deps-budget;work/sdmpc_rl_carry_20260929'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& '.venv-torch/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider work/sdmpc_rl_carry_20260929/test_carry_controller.py
```

Final output, exit code 0:

```text
......................................................                   [100%]
54 passed in 0.32s
```

The same command first failed before collection because sandbox access to
`.deps-budget/pytest/__main__.py` and `__init__.py` was denied. It was retried
with elevated dependency access, as anticipated by the brief. The first
elevated collection found pytest's reserved `request` parameter name in a new
test; it was renamed to `requested` using `apply_patch`, then the suite passed.
Bytecode generation, pytest's cache, and third-party plugin autoload were
disabled for all test attempts.

Coverage includes negative NP, both NUF clipping bounds, explicit clipped
reference projection, all reference sources, invalid reference recovery,
unrelated runtime errors, invalid modes/actions/guards, single-candidate center
execution, both guard modes, physical/budget fallback, actual executed-budget
carry across accepted and rejected intervals, serialized reference
reconstruction, solver exceptions with mutated arguments, mismatched solver
requests, archive/request/dual identity, and detached nested logs.

## Caveats and blockers

No outstanding blockers. Dependency access was resolved without installation
or permission changes. Tests use fake runtime modules only and make no claims
about numerical traffic behavior, influence gates, or training readiness.
Environment/runner/probe source interfaces were inspected, including restore
of `lower.last_budget` and reference source; their end-to-end execution remains
the coordinator's independent task.
