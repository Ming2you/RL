# Carry controller implementation

Repository: C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL.
Read docs/rl_budget_carry_plan_20260929.md and v1 budget_controller.py.
Write ONLY work/sdmpc_rl_carry_20260929/budget_controller.py and
work/sdmpc_rl_carry_20260929/test_carry_controller.py plus your report here.
Do not change frozen physical snapshot, v1 code, other v2 files, git index,
commits, branches, schedules or launch real simulations. Work in shared checkout.

Implement a new BudgetController based on the reviewed v1 operations with these
changes. Keep the tested lower response and transactional price behavior.

- Constructor BudgetController(runtime, guard_mode='physical'), allow only
  'physical' or 'h3'. It is valid to preserve source structure in this independently
  versioned experiment; avoid a new generic framework or global monkeypatches.
- prepare_reference(state, forecast, previous, warm, reference_source='previous')
  maps/quantizes a CLIPPED coordinate vector into current bounds before evaluation.
  Validate physical/budget validity exactly as v1. Invalid reference must raise a
  dedicated InvalidReference(ValueError) rather than obscure other exceptions.
  Calling prepare on an already prepared interval remains an error.
  reference_source must be previous, pfo_initial or pfo_recovery. Save source in
  reference. Add self.action_anchor = prior executed budget if available, else
  current reference witness budget. Preserve signed NP. It is NOT last rejected
  budget and is not silently reanchored to the new witness. Persist this as part
  of reference or reconstruct from prior_budget on restore.
- evaluate_and_commit(action, mode='rl') allows only rl or center, always one lower
  candidate. center uses zero action. Request = action_anchor + [50,1000]*action,
  NUF clipped to [0,capacity]. Retain B_raw, B_requested shape list-of-budget,
  B_executed, G_achieved and all full detached diagnostics. Log anchor and source,
  clipping vector, offset from current witness and H3 candidate-vs-reference delta.
- select_and_commit: physical/budget failures always fall back. H3-only excess
  falls back only when guard_mode='h3'. 'physical' permits a feasible physically
  valid candidate with worse predicted H3 TTT. Keep that diagnostic visible.
  Fallback selected source is 'reference_fallback', not 'PFO_reference'. Selected
  fallback B_executed is current validated reference budget, not failed request;
  incoming price state unchanged. Successful lower solution keeps its own dual.
  Record h3_guard_enabled, h3_guard_would_reject and reference_source.
- Preserve native-v1 implementation untouched, do not add native mode here.
- Fake-runtime tests must cover negative NP, clipping, actual previous-executed
  anchoring, both guards, invalid references, transactional failed solves,
  fallback dual/request identity and detached logs.

Tests: .venv-torch/Scripts/python.exe -B -m pytest (insert .deps-budget and this
version directory if needed). Windows dependency ACL may require escalation.
Report exact command/output, changed files and caveats in controller_report.md.
