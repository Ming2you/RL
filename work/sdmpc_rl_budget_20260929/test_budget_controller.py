"""Unit contracts that do not load historical traffic modules."""
import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from budget_controller import BudgetController, residual_budget


class BudgetMappingTests(unittest.TestCase):
    def test_zero_and_signed_np(self):
        raw, requested = residual_budget([0., 0.], [-110., 6000.], 6000.)
        np.testing.assert_array_equal(raw, [-110., 6000.])
        np.testing.assert_array_equal(raw, requested)
        _, requested = residual_budget([-1., 1.], [-110., 6000.], 6000.)
        np.testing.assert_array_equal(requested, [-160., 6000.])

    def test_invalid_actions(self):
        for action in ([float("nan"), 0.], [2., 0.], [0.], [[0., 0.]]):
            with self.assertRaises(ValueError):
                residual_budget(action, [0., 100.], 6000.)

    def test_evaluation_cannot_commit_even_on_failure(self):
        obj = BudgetController.__new__(BudgetController)
        obj.prepared, obj.results = True, []
        obj.incoming = np.array([[2., 3.], [0., 0.]])
        obj.prior_budget, obj.seed = np.array([1., 2.]), np.zeros(3)
        obj.lower = SimpleNamespace(dual=obj.incoming.copy(), last_budget=obj.prior_budget.copy())
        def bad_solve(*args):
            obj.lower.dual[:] = 100.
            obj.lower.last_budget[:] = 100.
            raise RuntimeError("injected")
        obj.lower.solve = bad_solve
        with self.assertRaisesRegex(RuntimeError, "injected"):
            obj.evaluate_budget([1., 2.])
        np.testing.assert_array_equal(obj.lower.dual, obj.incoming)
        np.testing.assert_array_equal(obj.lower.last_budget, obj.prior_budget)
        self.assertEqual(obj.results, [])


class AuditTests(unittest.TestCase):
    def setUp(self):
        from test_budget_contract import environment
        self.env = environment()
        self.obj = self.env.controller
        point = np.array([1., 2.])
        self.failed = dict(budget=[-50., 1000.], point=point,
                           evaluation=self.obj.lower.evaluate(point),
                           control=self.obj.lower.coords.decode(point), feasible=False,
                           converged=False, status="algorithm_failed_no_feasible", reason="iteration_limit",
                           original_residual=[51., 0.], stationarity=3., stationarity_point=point.copy(),
                           dual=np.ones((2, 2)), rows=[dict(dual_before=[[2., 3.]], dual_after=[[4., 5.]])],
                           local_rows=[dict(success=False, local_qp_stationarity=3.)],
                           leader_multiplier=dict(rejection_reasons=["fit_failed"]))
        self.obj.results = [self.failed]
        self.obj.lower.execution_archives = [{}]
        module = patch.dict(sys.modules, {"exception_controller": SimpleNamespace(
            choose_archive=lambda rows: rows[0] if rows else None)})
        module.start()
        self.addCleanup(module.stop)

    def test_fallback_preserves_failure_evidence_and_incoming_dual(self):
        incoming = self.obj.incoming.copy()
        _, audit = self.obj.select_and_commit()
        np.testing.assert_array_equal(self.obj.lower.dual, incoming)
        np.testing.assert_array_equal(audit["B_requested"], [[-50., 1000.]])
        np.testing.assert_array_equal(audit["B_executed"], [0., 1000.])
        candidate = audit["candidates"][0]
        self.assertFalse(candidate["response_check"]["budget_feasible"])
        self.assertTrue(audit["execution_check"]["budget_feasible"])
        self.assertEqual(candidate["status"], "algorithm_failed_no_feasible")
        self.assertEqual(candidate["stationarity"], 3.)
        self.assertEqual(candidate["predicted_TTT"], 12.)
        self.assertEqual(candidate["achieved"], [1., 1000.])
        self.assertEqual(candidate["rows"], self.failed["rows"])
        self.assertEqual(candidate["local_rows"], self.failed["local_rows"])
        self.assertNotIn("evaluation", candidate)
        self.assertNotIn("states", candidate)

    def test_audit_detached_across_next_preparation(self):
        from budget_runtime import plain
        _, audit = self.obj.select_and_commit()
        expected = copy.deepcopy(plain(audit))
        self.failed["budget"][0] = 999.
        self.failed["rows"][0]["dual_before"][0][0] = 999.
        self.obj.reference["check"]["achieved"][0] = 999.
        with patch.dict(sys.modules, {"fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]])}):
            self.obj.prepare_reference(self.env.sim.state, self.env.forecast, self.env.previous, self.env.previous)
        self.assertEqual(self.obj.results, [])
        self.assertEqual(plain(audit), expected)

    def test_archive_identity_is_separate_from_failed_final_point(self):
        point = np.array([-50., -1.])
        self.obj.lower.execution_archives = [{point.tobytes(): (point, "iteration_anchor_0_quantized")}]
        _, audit = self.obj.select_and_commit()
        identity = audit["selected_identity"]
        self.assertEqual(identity["kind"], "archive")
        self.assertEqual(identity["candidate_index"], 0)
        self.assertEqual(identity["source"], "iteration_anchor_0_quantized")
        np.testing.assert_array_equal(identity["point"], point)
        np.testing.assert_array_equal(audit["candidates"][0]["point"], [1., 2.])
        np.testing.assert_array_equal(audit["committed_dual"], self.obj.incoming)

    def test_guard_rejected_archive_check_survives_pfo_fallback(self):
        point = np.array([-50., 1.])
        self.obj.lower.execution_archives = [{point.tobytes(): (point, "lower_final_quantized")}]
        _, audit = self.obj.select_and_commit()
        self.assertEqual(audit["selected_identity"]["kind"], "PFO_reference")
        self.assertEqual(audit["lower_selection_identity"]["kind"], "archive")
        self.assertEqual(audit["lower_selection_check"]["ttt"], 11.)
        self.assertEqual(audit["execution_check"]["ttt"], 10.)
        self.assertEqual(audit["fallback_reasons"], ["H3_TTT_guard"])
        np.testing.assert_array_equal(audit["committed_dual"], self.obj.incoming)

    def test_feasible_candidate_commits_its_prices(self):
        point = np.array([-50., -1.])
        self.failed.update(feasible=True, point=point, evaluation=self.obj.lower.evaluate(point),
                           control=self.obj.lower.coords.decode(point), status="feasible_uncertified")
        _, audit = self.obj.select_and_commit()
        self.assertEqual(audit["selected_identity"]["kind"], "candidate")
        self.assertEqual(audit["selected_identity"]["candidate_index"], 0)
        self.assertEqual(audit["fallback_reasons"], [])
        np.testing.assert_array_equal(audit["committed_dual"], self.failed["dual"])

    def test_endpoint_queue_estimate_threshold_units_and_duration(self):
        self.obj.lower.solve = lambda *args: self.failed
        self.env.warm.solve = lambda *args: SimpleNamespace(control=self.env.previous)
        self.env.sim.state.ramp_queue = dict(low=161., edge=162., over=181.)
        self.env.sim.state.boundary_queue = dict(low=89., edge=90., over=101.)
        self.env.observer.names = None
        self.env.observer.encode(self.env)
        with patch.dict(sys.modules, {"fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]])}):
            _, _, _, audit = self.env.step([-1., 0.])
        estimate = audit["queue_near_capacity_estimate"]
        self.assertEqual(estimate["method"], "interval_endpoint_sampled")
        self.assertFalse(estimate["exact_substep_exposure"])
        self.assertEqual(estimate["threshold_fraction"], .9)
        self.assertEqual(estimate["duration_units"], "s")
        self.assertEqual(estimate["capacity_units"], "veh")
        self.assertEqual(estimate["interval_seconds"], 180.)
        for kind, cap in (("ramp_queue", 180.), ("boundary_queue", 100.)):
            self.assertEqual(estimate[kind]["low"]["near_capacity_seconds"], 0.)
            self.assertEqual(estimate[kind]["edge"]["near_capacity_seconds"], 180.)
            self.assertEqual(estimate[kind]["over"]["near_capacity_seconds"], 180.)
            self.assertEqual(estimate[kind]["edge"]["capacity_veh"], cap)
        self.assertIsNone(audit["decision_cpu_seconds"])
        self.env.sim.state.ramp_queue["edge"] = 0.
        self.assertEqual(estimate["ramp_queue"]["edge"]["queue_veh"], 162.)
        shorter = self.env._queue_near_capacity_estimate(60.)
        self.assertEqual(shorter["boundary_queue"]["edge"]["near_capacity_seconds"], 60.)

    def test_nonterminal_step_keeps_plant_and_candidate_audits(self):
        self.obj.lower.solve = lambda *args: self.failed
        self.env.warm.solve = lambda *args: SimpleNamespace(control=self.env.previous)
        with patch.dict(sys.modules, {"fixed_policy": SimpleNamespace(mask=lambda: [[True, True], [False, False]])}):
            _, reward, terminal, audit = self.env.step([-1., 0.], actor_seconds=.2, actor_cpu_seconds=.1)
        self.assertEqual(reward, -.05)
        self.assertFalse(terminal)
        self.assertEqual(self.obj.results, [])
        self.assertEqual(audit["candidates"][0]["status"], "algorithm_failed_no_feasible")
        self.assertEqual(audit["plant_log"]["diagnostics"], {"conservation": 0.})
        self.assertEqual(audit["plant_state"]["time_sec"], 1080.)
        self.env.sim.state.ramp_queue["r"] = 99.
        self.env.sim.logs[-1].diagnostics["conservation"] = 99.
        self.assertEqual(audit["plant_state"]["ramp_queue"], {"r": 2.})
        self.assertEqual(audit["plant_log"]["diagnostics"], {"conservation": 0.})
        phases = ("forecast", "observation", "pfo", "reference", "lower", "guard", "actor")
        for clock in ("wall", "cpu"):
            self.assertAlmostEqual(audit[f"decision_{clock}_seconds"],
                                   sum(audit[f"{phase}_{clock}_seconds"] for phase in phases))
            self.assertGreaterEqual(audit[f"plant_{clock}_seconds"], 0.)


if __name__ == "__main__":
    unittest.main()
