"""Budget proposal, lower response and guarded commit are separate operations."""
import copy
import time
import numpy as np


def residual_budget(action, reference, capacity):
    action = np.asarray(action, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
        raise ValueError("Action must be finite and inside [-1,1]^2")
    if reference.shape != (2,) or not np.isfinite(reference).all() or capacity <= 0:
        raise ValueError("Invalid reference or metering capacity")
    raw = reference + np.array([50., 1000.]) * action
    requested = raw.copy()
    requested[1] = np.clip(requested[1], 0., capacity)
    return raw, requested


def next_native_request(center, requests, result, index, capacity):
    from multiplier import direction_from_recovery
    hint = direction_from_recovery(result["leader_multiplier"]) if result["feasible"] else np.zeros(2)
    origin = "selected_action_multiplier"
    if not np.any(hint):
        hint = np.array([1., 0.]) if index == 0 else np.array([0., 1.])
        origin = "explicit_axis_search_no_valid_multiplier_direction"
    radius = np.array([50., 1000.])
    for request, source in ((center + radius * hint, origin),
                            (center - radius * hint, origin + "_opposite"),
                            (center + radius * [0., 1.], "explicit_NUF_axis"),
                            (center - radius * [0., 1.], "explicit_NUF_axis_opposite")):
        request[1] = np.clip(request[1], 0., capacity)
        if not any(np.array_equal(request, old) for old in requests):
            return request, source
    return None, None


class BudgetController:
    def __init__(self, runtime):
        self.runtime = runtime
        self.lower = runtime["baseline"](runtime["cfg"], runtime["options"])
        self.prepared = False

    def prepare_reference(self, state, forecast, previous, warm):
        if self.prepared:
            raise RuntimeError("Previous interval was not committed")
        from fixed_policy import mask
        lower = self.lower
        lower.dual = np.where(mask(), lower.dual, 0.)
        self.incoming = lower.dual.copy()
        self.prior_budget = None if lower.last_budget is None else lower.last_budget.copy()
        coords = self.runtime["coordinates"](lower.cfg, lower.options, previous)
        mapped = coords.decode(coords.quantize(coords.encode(warm)))
        lower.begin(state, forecast, previous)
        self.seed = lower.coords.encode(mapped)
        witness = lower.evaluate(self.seed)
        budget = witness.budget_vector.copy()
        check = lower.execution_check(self.seed, budget)
        if not check["physical_control_valid"] or not check["budget_feasible"]:
            raise RuntimeError("invalid_PFO_reference_fail_closed")
        self.reference = dict(budget=budget, point=self.seed.copy(), evaluation=witness,
                              check=check, control=lower.coords.decode(self.seed))
        self.results = []
        self.prepared = True
        return self.reference

    def evaluate_budget(self, request, origin="actor"):
        if not self.prepared:
            raise RuntimeError("Missing reference")
        request = np.asarray(request, dtype=float)
        if request.shape != (2,) or not np.isfinite(request).all():
            raise ValueError("Invalid requested budget")
        lower = self.lower
        try:
            result = lower.solve(request.copy(), self.seed.copy(), self.incoming.copy())
        finally:
            # Persistent state is transactionally isolated even on solver exceptions.
            lower.dual = self.incoming.copy()
            lower.last_budget = None if self.prior_budget is None else self.prior_budget.copy()
        result["request_origin"] = origin
        if not np.array_equal(result["budget"], request):
            raise RuntimeError("Lower solver changed the requested budget")
        self.results.append(result)
        return result

    def _lower_selection(self):
        from exception_controller import choose_archive
        lower = self.lower
        eligible = [row for row in self.results if row["feasible"]]
        if eligible:
            selected = min(eligible, key=lambda r: (r["evaluation"].total_ttt, tuple(r["budget"])))
            index = next(i for i, row in enumerate(self.results) if row is selected)
            selected = dict(selected)
            selected.update(execution_exception=False, execution_allowed=True,
                            selection_identity=dict(kind="candidate", candidate_index=index,
                                                    point=selected["point"]),
                            execution_check=lower.execution_check(selected["point"], selected["budget"]))
            return selected
        archive = []
        for index, (result, points) in enumerate(zip(self.results, lower.execution_archives)):
            for point, source in points.values():
                row = lower.execution_check(point, result["budget"])
                row.update(candidate_index=index, source=source)
                archive.append(row)
        chosen = choose_archive(archive)
        if chosen is None:
            return None
        point = np.asarray(chosen["point"])
        control = lower.coords.decode(point)
        control.N_P_star, control.N_UF_star = chosen["budget"]
        return dict(point=point, budget=chosen["budget"], control=control,
                    evaluation=lower.evaluate(point), feasible=chosen["budget_feasible"],
                    converged=False, execution_exception=not chosen["budget_feasible"],
                    execution_check=chosen, dual=self.incoming.copy(), stationarity=None,
                    selection_identity=dict(kind="archive", candidate_index=chosen["candidate_index"],
                                            source=chosen["source"], point=point),
                    reason="archive_recovery_after_all_final_candidates_failed",
                    original_residual=chosen["original_residual"], rows=[], local_rows=[])

    def _candidate_audit(self, index, result):
        # Exclude rollout objects: they contain full predicted TrafficState trajectories.
        row = {key: value for key, value in result.items() if key not in ("evaluation", "control")}
        row.update(candidate_index=index, achieved=result["evaluation"].budget_vector.tolist(),
                   predicted_TTT=result["evaluation"].total_ttt,
                   response_check=self.lower.execution_check(result["point"], result["budget"]))
        return row

    def select_and_commit(self):
        if not self.prepared or not self.results:
            raise RuntimeError("No evaluated budget")
        lower, ref = self.lower, self.reference
        selected = self._lower_selection()
        lower_identity = None if selected is None else selected["selection_identity"]
        lower_check = None if selected is None else selected["execution_check"]
        reasons = []
        if selected is None:
            reasons.append("no_lower_execution")
        else:
            if not selected["feasible"] or not selected["execution_check"]["budget_feasible"]:
                reasons.append("requested_budget_infeasible")
            if not selected["execution_check"]["physical_control_valid"]:
                reasons.append("lower_physical_control_invalid")
            if selected["evaluation"].total_ttt > ref["evaluation"].total_ttt:
                reasons.append("H3_TTT_guard")
        if reasons:
            control = ref["control"].copy()
            control.N_P_star, control.N_UF_star = ref["budget"]
            selected = dict(point=ref["point"].copy(), budget=ref["budget"].tolist(),
                            control=control, evaluation=ref["evaluation"], feasible=True,
                            converged=False, execution_exception=False,
                            execution_check=ref["check"], dual=self.incoming.copy(),
                            stationarity=None, reason="PFO_reference_TTT_guard",
                            selection_source="PFO_reference", rows=[], local_rows=[],
                            selection_identity=dict(kind="PFO_reference", candidate_index=None,
                                                    point=ref["point"]),
                            original_residual=ref["check"]["original_residual"])
        else:
            selected["selection_source"] = "lower_solution"
        lower.dual = np.asarray(selected["dual"]).copy()
        lower.last_budget = np.asarray(selected["budget"]).copy()
        self.prepared = False
        self.runtime["audit_rows"](self.results)
        candidates = [self._candidate_audit(i, row) for i, row in enumerate(self.results)]
        audit = dict(B_requested=[row["budget"] for row in self.results],
                     B_executed=selected["budget"],
                     G_achieved=selected["execution_check"]["achieved"],
                     reference_budget=ref["budget"], reference_TTT=ref["evaluation"].total_ttt,
                     selected_TTT=selected["evaluation"].total_ttt,
                     selection_source=selected["selection_source"], fallback_reasons=reasons,
                     incoming_dual=self.incoming.copy(), committed_dual=lower.dual.copy(),
                     execution_check=selected["execution_check"], converged=selected["converged"],
                     candidates=candidates, selected_identity=selected["selection_identity"],
                     lower_selection_identity=lower_identity, lower_selection_check=lower_check,
                     derivatives=getattr(lower, "derivative_rows", []),
                     lower_candidate_count=len(self.results), counts=copy.deepcopy(lower.counts))
        if audit["selected_TTT"] > audit["reference_TTT"]:
            raise RuntimeError("PFO TTT guard violated")
        if not audit["execution_check"]["budget_feasible"]:
            raise RuntimeError("Executed budget is infeasible")
        return selected, copy.deepcopy(audit)

    def evaluate_and_commit(self, action=None, mode="rl"):
        if mode not in ("rl", "center", "native"):
            raise ValueError("Unknown budget mode")
        ref = self.reference["budget"]
        raw, request = residual_budget(np.zeros(2) if mode != "rl" else action,
                                      ref, self.lower.cfg.network.total_ramp_capacity)
        count = self.lower.options.max_candidates if mode == "native" else 1
        source = "PFO_witness" if mode != "rl" else "actor"
        requests = []
        tick, cpu = time.perf_counter(), time.process_time()
        for index in range(count):
            requests.append(request.copy())
            result = self.evaluate_budget(request, source)
            if index + 1 < count:
                request, source = next_native_request(ref, requests, result, index,
                                                     self.lower.cfg.network.total_ramp_capacity)
                if request is None:
                    break
        solve_wall, solve_cpu = time.perf_counter() - tick, time.process_time() - cpu
        tick, cpu = time.perf_counter(), time.process_time()
        selected, audit = self.select_and_commit()
        audit.update(B_raw=raw, lower_wall_seconds=solve_wall, lower_cpu_seconds=solve_cpu,
                     guard_wall_seconds=time.perf_counter() - tick,
                     guard_cpu_seconds=time.process_time() - cpu)
        return selected, audit
