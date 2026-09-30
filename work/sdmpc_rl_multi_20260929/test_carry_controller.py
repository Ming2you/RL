"""Carry-controller contracts using only fake coordinates and lower responses."""
import copy
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from budget_controller import BudgetController, InvalidReference, residual_budget


class Control:
    def __init__(self, point):
        self.point = np.asarray(point, dtype=float).copy()
        self.N_P_star, self.N_UF_star = 0., 0.

    def copy(self):
        return copy.deepcopy(self)


class Coordinates:
    def __init__(self, cfg, options, previous):
        self.lower = np.array([-300., 0., -5.])
        self.upper = np.array([300., 6000., 5.])

    def encode(self, control):
        # Deliberately does not clip, so projection must occur in the controller.
        return control.point.copy()

    def quantize(self, point):
        assert np.all(point >= self.lower) and np.all(point <= self.upper)
        step = np.array([5., 100., .5])
        return np.round(point / step) * step

    def decode(self, point):
        return Control(point)


class Lower:
    def __init__(self, cfg, options):
        self.cfg, self.options = cfg, options
        self.dual = np.array([[2., 3.], [0., 0.]])
        self.last_budget = None
        self.coords = Coordinates(cfg, options, None)
        self.solve_calls = []
        self.invalid_reference = None
        self.candidate_ttt_delta = -1.
        self.candidate_budget_delta = -5.
        self.candidate_physical_valid = True
        self.report_feasible = True
        self.archive_points = []
        self.begin(None, None, None)

    def begin(self, state, forecast, previous):
        self.coords = Coordinates(self.cfg, self.options, previous)
        self.execution_archives = []
        self.derivative_rows = []
        self.counts = {"scalar_rollouts": 0, "local_qp_calls": 0}

    def evaluate(self, point):
        return SimpleNamespace(total_ttt=100. + point[2], budget_vector=point[:2].copy(),
                               states=["DO_NOT_EXPORT"])

    def execution_check(self, point, budget):
        ev = self.evaluate(point)
        residual = ev.budget_vector - budget
        width = np.array([10., .05 * abs(budget[1])])
        excess = np.maximum(abs(residual) - width, 0.)
        physical = (np.isfinite(ev.total_ttt) and np.isfinite(ev.budget_vector).all() and
                    np.all(point >= self.coords.lower) and np.all(point <= self.coords.upper))
        if self.solve_calls and not np.array_equal(point, self.solve_calls[-1][1]):
            physical = physical and self.candidate_physical_valid
        return dict(physical_control_valid=bool(physical and self.invalid_reference != "physical"),
                    budget_feasible=bool(np.all(excess == 0.) and self.invalid_reference != "budget"),
                    achieved=ev.budget_vector.tolist(), original_residual=residual.tolist(),
                    scaled_excess=(excess / [50., 1000.]).tolist(),
                    point=point.tolist(), budget=list(budget), ttt=ev.total_ttt)

    def solve(self, request, seed, incoming):
        self.solve_calls.append((request.copy(), seed.copy(), incoming.copy()))
        point = np.array([request[0] + self.candidate_budget_delta, request[1], self.candidate_ttt_delta])
        self.dual = np.array([[31., 37.], [0., 0.]])
        self.last_budget = request.copy()
        self.execution_archives.append({p.tobytes(): (p.copy(), "iteration_anchor_0_quantized")
                                        for p in self.archive_points})
        self.derivative_rows.append(dict(anchor=seed.copy(), gradient=[1., 2.]))
        self.counts["local_qp_calls"] += 1
        control = self.coords.decode(point)
        control.N_P_star, control.N_UF_star = request
        return dict(budget=request.copy(), point=point, evaluation=self.evaluate(point), control=control,
                    feasible=self.report_feasible, converged=False, dual=self.dual,
                    status="feasible_uncertified" if self.report_feasible else "algorithm_failed_no_feasible",
                    reason="iteration_limit", original_residual=[self.candidate_budget_delta, 0.],
                    stationarity=3., stationarity_point=point.copy(),
                    rows=[dict(dual_before=incoming.copy(), dual_after=self.dual.copy())],
                    local_rows=[dict(success=False, local_qp_stationarity=3.)],
                    leader_multiplier=dict(rejection_reasons=["fit_failed"]))


@pytest.fixture(autouse=True)
def frozen_module_stubs():
    modules = {
        "fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]]),
        "exception_controller": SimpleNamespace(choose_archive=lambda rows: rows[0] if rows else None),
    }
    with patch.dict(sys.modules, modules):
        yield


def controller(guard_mode="physical", prior=None, source="previous", warm=None):
    def audit_rows(rows):
        for row in rows:
            row["externality_audit"] = {"checked": [True]}

    runtime = dict(cfg=SimpleNamespace(network=SimpleNamespace(total_ramp_capacity=6000.)),
                   options=SimpleNamespace(max_candidates=3), baseline=Lower,
                   coordinates=Coordinates, audit_rows=audit_rows)
    obj = BudgetController(runtime, guard_mode)
    obj.lower.last_budget = None if prior is None else np.asarray(prior, dtype=float).copy()
    previous = Control([-100., 2000., 0.])
    obj.prepare_reference(None, None, previous, warm or previous, source)
    return obj


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value.tolist() if hasattr(value, "tolist") else value


@pytest.mark.parametrize("action", [[np.nan, 0.], [np.inf, 0.], [2., 0.], [0.], [[0., 0.]], None])
def test_invalid_actions(action):
    with pytest.raises(ValueError):
        residual_budget(action, [-110., 2000.], 6000.)


@pytest.mark.parametrize("budget,capacity", [([0.], 6000.), ([np.nan, 2.], 6000.),
                                            ([0., 2.], 0.), ([0., 2.], np.inf), ([0., 2.], np.nan)])
def test_invalid_mapping_reference_or_capacity(budget, capacity):
    with pytest.raises(ValueError):
        residual_budget([0., 0.], budget, capacity)


@pytest.mark.parametrize("prior,action,raw,requested,clipping", [
    ([-110., 5800.], [-1., 1.], [-160., 6800.], [-160., 6000.], [0., -800.]),
    ([-110., 200.], [-1., -1.], [-160., -800.], [-160., 0.], [0., 800.]),
    ([-110., 2000.], [0., 0.], [-110., 2000.], [-110., 2000.], [0., 0.]),
])
def test_signed_np_and_clipped_nuf_request(prior, action, raw, requested, clipping):
    obj = controller(prior=prior)
    selected, audit = obj.evaluate_and_commit(action)
    np.testing.assert_array_equal(audit["B_raw"], raw)
    np.testing.assert_array_equal(audit["B_requested"], [requested])
    np.testing.assert_array_equal(audit["B_executed"], requested)
    np.testing.assert_array_equal(audit["budget_clipping"], clipping)
    np.testing.assert_array_equal(audit["action_anchor"], prior)
    np.testing.assert_array_equal(audit["anchor_offset_from_reference"], np.asarray(prior) - [-100., 2000.])
    np.testing.assert_array_equal(audit["request_offset_from_reference"], np.asarray(requested) - [-100., 2000.])
    np.testing.assert_array_equal(audit["G_achieved"], [requested[0] - 5., requested[1]])
    assert selected["control"].N_P_star == requested[0]
    assert len(obj.lower.solve_calls) == audit["lower_candidate_count"] == 1


@pytest.mark.parametrize("source", ["previous", "pfo_initial", "pfo_recovery"])
def test_reference_source_and_initial_anchor(source):
    obj = controller(source=source)
    assert obj.reference["source"] == source
    np.testing.assert_array_equal(obj.action_anchor, [-100., 2000.])
    np.testing.assert_array_equal(obj.reference["action_anchor"], obj.action_anchor)
    obj.reference["action_anchor"][0] = 999.
    np.testing.assert_array_equal(obj.action_anchor, [-100., 2000.])
    selected, audit = obj.evaluate_and_commit(mode="center")
    assert selected["reference_source"] == audit["reference_source"] == source


def test_reference_clipped_before_quantization_and_evaluation():
    obj = controller(warm=Control([-900., 6500., .3]))
    np.testing.assert_array_equal(obj.seed, [-300., 6000., .5])
    np.testing.assert_array_equal(obj.reference["control"].point, obj.seed)
    assert obj.reference["check"]["physical_control_valid"]
    assert obj.reference["check"]["budget_feasible"]


@pytest.mark.parametrize("reason", ["physical", "budget"])
def test_invalid_reference_fails_closed_and_allows_recovery(reason):
    obj = controller(prior=[-120., 1800.])
    obj.prepared = False
    incoming, prior = obj.lower.dual.copy(), obj.lower.last_budget.copy()
    obj.lower.invalid_reference = reason
    with pytest.raises(InvalidReference, match="invalid_reference_fail_closed"):
        obj.prepare_reference(None, None, obj.reference["control"], obj.reference["control"])
    assert not obj.prepared
    np.testing.assert_array_equal(obj.lower.dual, incoming)
    np.testing.assert_array_equal(obj.lower.last_budget, prior)
    with pytest.raises(RuntimeError, match="Missing reference"):
        obj.evaluate_and_commit(mode="center")
    obj.lower.invalid_reference = None
    obj.prepare_reference(None, None, obj.reference["control"], Control([-50., 3000., 0.]), "pfo_recovery")
    assert obj.reference["source"] == "pfo_recovery"
    np.testing.assert_array_equal(obj.action_anchor, prior)


def test_reference_errors_do_not_hide_unrelated_runtime_failures():
    obj = controller()
    obj.prepared = False
    error = RuntimeError("broken evaluator")
    with patch.object(obj.lower, "evaluate", side_effect=error):
        with pytest.raises(RuntimeError) as raised:
            obj.prepare_reference(None, None, obj.reference["control"], obj.reference["control"])
    assert raised.value is error
    assert not obj.prepared
    assert issubclass(InvalidReference, ValueError)


def test_prepare_twice_and_invalid_source_do_not_change_interval():
    obj = controller()
    ref = obj.reference
    with pytest.raises(RuntimeError, match="not committed"):
        obj.prepare_reference(None, None, ref["control"], ref["control"])
    assert obj.reference is ref and obj.prepared
    obj.prepared = False
    with pytest.raises(ValueError, match="reference source"):
        obj.prepare_reference(None, None, ref["control"], ref["control"], "pfo")
    assert obj.reference is ref and not obj.prepared


@pytest.mark.parametrize("mode", ["native", "pfo", "PHYSICAL", None])
def test_only_rl_and_center_modes(mode):
    obj = controller()
    with pytest.raises(ValueError, match="budget mode"):
        obj.evaluate_and_commit([0., 0.], mode)
    assert obj.lower.solve_calls == [] and obj.prepared


@pytest.mark.parametrize("guard", ["native", "H3", None])
def test_invalid_guard_rejected_before_runtime_construction(guard):
    with pytest.raises(ValueError, match="guard mode"):
        BudgetController({}, guard)


def test_center_uses_zero_action_about_carried_budget():
    obj = controller(prior=[-140., 1800.])
    _, audit = obj.evaluate_and_commit([np.nan, np.inf], "center")
    np.testing.assert_array_equal(audit["B_requested"], [[-140., 1800.]])
    assert audit["candidates"][0]["request_origin"] == "carried_center"
    assert len(obj.lower.solve_calls) == 1
    with pytest.raises(RuntimeError, match="Missing reference"):
        obj.evaluate_and_commit(mode="center")


def test_evaluate_and_commit_does_not_add_a_second_candidate():
    obj = controller()
    obj.evaluate_budget([-100., 2000.])
    with pytest.raises(RuntimeError, match="already has an evaluated budget"):
        obj.evaluate_and_commit(mode="center")
    assert len(obj.lower.solve_calls) == 1


@pytest.mark.parametrize("guard", ["physical", "h3"])
@pytest.mark.parametrize("delta", [-1., 0., 1.])
def test_h3_guard_is_diagnostic_or_enforced(guard, delta):
    obj = controller(guard, prior=[-120., 1800.])
    obj.lower.candidate_ttt_delta = delta
    incoming = obj.incoming.copy()
    selected, audit = obj.evaluate_and_commit([0., 0.])
    fallback = guard == "h3" and delta > 0.
    assert audit["guard_mode"] == guard
    assert audit["h3_guard_enabled"] == (guard == "h3")
    assert audit["h3_guard_would_reject"] == (delta > 0.)
    assert audit["h3_TTT_delta"] == audit["candidates"][0]["h3_TTT_delta"] == delta
    assert audit["fallback_reasons"] == (["H3_TTT_guard"] if fallback else [])
    assert selected["selection_source"] == ("reference_fallback" if fallback else "lower_solution")
    np.testing.assert_array_equal(audit["B_requested"], [[-120., 1800.]])
    np.testing.assert_array_equal(audit["B_executed"], [-100., 2000.] if fallback else [-120., 1800.])
    np.testing.assert_array_equal(audit["committed_dual"], incoming if fallback else [[31., 37.], [0., 0.]])
    assert audit["selected_TTT"] == (100. if fallback else 100. + delta)


@pytest.mark.parametrize("guard", ["physical", "h3"])
@pytest.mark.parametrize("failure", ["physical", "budget", "no_candidate"])
def test_strict_execution_failures_fall_back_in_both_guards(guard, failure):
    obj = controller(guard, prior=[-120., 1800.], source="pfo_recovery")
    if failure == "physical":
        obj.lower.candidate_physical_valid = False
    elif failure == "budget":
        obj.lower.candidate_budget_delta = 50.
    else:
        obj.lower.report_feasible = False
    incoming = obj.incoming.copy()
    selected, audit = obj.evaluate_and_commit([-1., 0.])
    expected_reason = {"physical": "lower_physical_control_invalid", "budget": "requested_budget_infeasible",
                       "no_candidate": "no_lower_execution"}[failure]
    assert audit["fallback_reasons"] == [expected_reason]
    assert audit["selection_source"] == audit["selected_identity"]["kind"] == "reference_fallback"
    assert audit["reference_source"] == "pfo_recovery"
    np.testing.assert_array_equal(audit["B_requested"], [[-170., 1800.]])
    np.testing.assert_array_equal(audit["B_executed"], [-100., 2000.])
    np.testing.assert_array_equal(obj.lower.last_budget, [-100., 2000.])
    np.testing.assert_array_equal(audit["G_achieved"], [-100., 2000.])
    np.testing.assert_array_equal(obj.lower.dual, incoming)
    assert selected["control"].N_P_star == -100.
    assert selected["control"].N_UF_star == 2000.


@pytest.mark.parametrize("rejected", [False, True])
def test_next_interval_anchors_to_executed_budget_not_achieved_request_or_new_witness(rejected):
    obj = controller(prior=[-120., 1800.])
    obj.lower.report_feasible = not rejected
    selected, audit = obj.evaluate_and_commit([-1., 0.])
    executed = np.asarray(audit["B_executed"]).copy()
    expected = [-100., 2000.] if rejected else [-170., 1800.]
    np.testing.assert_array_equal(executed, expected)
    obj.prepare_reference(None, None, selected["control"], Control([-50., 3000., 0.]))
    np.testing.assert_array_equal(obj.action_anchor, expected)
    np.testing.assert_array_equal(obj.reference["budget"], [-50., 3000.])
    obj.lower.report_feasible = True
    _, next_audit = obj.evaluate_and_commit(mode="center")
    np.testing.assert_array_equal(next_audit["B_requested"], [expected])


@pytest.mark.parametrize("prior", [None, [-120., 1800.]])
def test_reference_serialization_and_reconstruction_preserve_anchor(prior):
    obj = controller(prior=prior, source="pfo_recovery")
    saved = pickle.loads(pickle.dumps(dict(reference=obj.reference, last_budget=obj.lower.last_budget)))
    restored = BudgetController(obj.runtime)
    restored.lower.last_budget = saved["last_budget"]
    ref = saved["reference"]
    restored.prepare_reference(None, None, ref["control"], ref["control"], ref["source"])
    np.testing.assert_array_equal(restored.action_anchor, ref["action_anchor"])
    np.testing.assert_array_equal(restored.action_anchor, obj.action_anchor)
    _, original_audit = obj.evaluate_and_commit([.25, -.25])
    _, restored_audit = restored.evaluate_and_commit([.25, -.25])
    for key in ("B_raw", "B_requested", "B_executed", "G_achieved", "committed_dual", "action_anchor"):
        np.testing.assert_array_equal(restored_audit[key], original_audit[key])


@pytest.mark.parametrize("prior", [None, [-120., 1800.]])
def test_failed_solver_is_transactional_even_when_it_mutates_arguments(prior):
    obj = controller(prior=prior)
    incoming, seed, anchor = obj.incoming.copy(), obj.seed.copy(), obj.action_anchor.copy()

    def fail(request, point, dual):
        obj.lower.dual[:] = 999.
        obj.lower.last_budget = np.array([999., 999.])
        request[:], point[:], dual[:] = 888., 888., 888.
        raise RuntimeError("injected failure")

    with patch.object(obj.lower, "solve", side_effect=fail):
        with pytest.raises(RuntimeError, match="injected failure"):
            obj.evaluate_and_commit([0., 0.])
    np.testing.assert_array_equal(obj.lower.dual, incoming)
    np.testing.assert_array_equal(obj.incoming, incoming)
    np.testing.assert_array_equal(obj.seed, seed)
    np.testing.assert_array_equal(obj.action_anchor, anchor)
    if prior is None:
        assert obj.lower.last_budget is None
    else:
        np.testing.assert_array_equal(obj.lower.last_budget, prior)
    assert obj.prepared and obj.results == []


def test_changed_lower_request_fails_without_commit():
    obj = controller(prior=[-120., 1800.])
    original_solve = obj.lower.solve

    def wrong_request(*args):
        result = original_solve(*args)
        result["budget"][0] += 1.
        return result

    with patch.object(obj.lower, "solve", side_effect=wrong_request):
        with pytest.raises(RuntimeError, match="changed the requested budget"):
            obj.evaluate_and_commit([0., 0.])
    np.testing.assert_array_equal(obj.lower.dual, obj.incoming)
    np.testing.assert_array_equal(obj.lower.last_budget, [-120., 1800.])
    assert obj.prepared and obj.results == []


@pytest.mark.parametrize("guard", ["physical", "h3"])
def test_archive_retains_request_and_incoming_prices_and_separate_identity(guard):
    obj = controller(guard, prior=[-120., 1800.])
    obj.lower.report_feasible = False
    obj.lower.candidate_budget_delta = 50.
    obj.lower.archive_points = [np.array([-120., 1800., 1.])]
    _, audit = obj.evaluate_and_commit([0., 0.])
    assert audit["lower_selection_identity"]["kind"] == "archive"
    assert audit["lower_selection_identity"]["source"] == "iteration_anchor_0_quantized"
    assert audit["lower_selection_identity"]["candidate_index"] == 0
    np.testing.assert_array_equal(audit["lower_selection_identity"]["point"], [-120., 1800., 1.])
    np.testing.assert_array_equal(audit["candidates"][0]["point"], [-70., 1800., -1.])
    assert audit["candidates"][0]["status"] == "algorithm_failed_no_feasible"
    assert not audit["candidates"][0]["response_check"]["budget_feasible"]
    assert audit["lower_selection_check"]["budget_feasible"]
    assert audit["h3_TTT_delta"] == 1. and audit["candidates"][0]["h3_TTT_delta"] == -1.
    assert audit["h3_guard_would_reject"]
    assert audit["selected_identity"]["kind"] == ("archive" if guard == "physical" else "reference_fallback")
    np.testing.assert_array_equal(audit["B_requested"], [[-120., 1800.]])
    np.testing.assert_array_equal(audit["B_executed"], [-120., 1800.] if guard == "physical" else [-100., 2000.])
    np.testing.assert_array_equal(audit["committed_dual"], obj.incoming)


@pytest.mark.parametrize("fallback", [False, True])
def test_full_audits_remain_detached_after_mutation_and_next_preparation(fallback):
    obj = controller()
    obj.lower.report_feasible = not fallback
    selected, audit = obj.evaluate_and_commit([0., 0.])
    candidate = audit["candidates"][0]
    assert candidate["stationarity"] == 3.
    assert candidate["predicted_TTT"] == 99.
    assert candidate["leader_multiplier"]["rejection_reasons"] == ["fit_failed"]
    assert candidate["local_rows"] == [dict(success=False, local_qp_stationarity=3.)]
    assert candidate["externality_audit"] == {"checked": [True]}
    assert "evaluation" not in candidate and "control" not in candidate and "states" not in candidate
    assert audit["counts"]["local_qp_calls"] == 1
    for phase in ("lower", "guard"):
        for clock in ("wall", "cpu"):
            assert audit[f"{phase}_{clock}_seconds"] >= 0.
    expected = plain(copy.deepcopy(audit))
    obj.results[0]["budget"][0] = 999.
    obj.results[0]["rows"][0]["dual_before"][:] = 999.
    obj.results[0]["local_rows"][0]["local_qp_stationarity"] = 999.
    obj.results[0]["externality_audit"]["checked"][0] = False
    obj.reference["check"]["achieved"][0] = 999.
    obj.reference["budget"][0] = 999.
    obj.action_anchor[:] = 999.
    obj.lower.dual[:] = 999.
    obj.lower.derivative_rows[0]["gradient"][0] = 999.
    obj.lower.counts["local_qp_calls"] = 999
    selected["point"][:] = 999.
    obj.prepare_reference(None, None, Control([-50., 3000., 0.]), Control([-50., 3000., 0.]))
    assert obj.results == []
    assert plain(audit) == expected
