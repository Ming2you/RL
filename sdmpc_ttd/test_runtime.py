"""배포본의 입력 재현·회계·상위 탐색 계약. 최적성 인증 시험은 아니다."""
import copy
from types import SimpleNamespace as NS
import unittest

from .runtime import bootstrap, protocol, SCENARIOS, verify_sources

rc, cfg, options, _, _ = bootstrap()
import numpy as np
from controller import SelectedDualSDMPC
from upper_search import decide


class FakeSolver:
    def __init__(self, center, cap, gradient=(0., 0.), feasible=True):
        self.center = np.array(center, dtype=float)
        self.options = NS(max_candidates=cap)
        self.cfg = NS(network=NS(total_ramp_capacity=6000.))
        self.dual = np.array([[.2, .3], [0., 0.]])
        self.last_budget = None
        self.coords = NS(encode=lambda warm: np.array([.4, .5]))
        self.gradient, self.valid = gradient, feasible
        self.incoming = []

    def begin(self, *args):
        pass

    def evaluate(self, y):
        return NS(budget_vector=self.center.copy())

    def solve(self, budget, seed, dual):
        self.incoming.append(dual.copy())
        dual[:] += 1
        cost = float((budget[0]-self.center[0]-12.5)**2+(budget[1]-self.center[1])**2/1e6)
        return dict(budget=budget.tolist(), feasible=self.valid, dual=dual,
            leader_multiplier=dict(accepted_for_leader_direction=True, gradient_estimate=self.gradient),
            evaluation=NS(total_ttt=cost), point=seed.copy())


class RuntimeTests(unittest.TestCase):
    def test_original_source_manifest(self):
        self.assertGreater(verify_sources(), 100)

    def test_all_five_protocol_fingerprints(self):
        for scenario in SCENARIOS:
            data = protocol(rc, cfg, scenario)
            self.assertEqual(len(data['forecast']), 87)
        self.assertEqual((options.max_iterations, options.max_candidates, options.horizon_steps), (6, 3, 3))

    def test_three_candidates_match_original_search(self):
        for center in [(0, 0), (10, 1000), (-40, 5999), (90, 6000)]:
            for gradient in [(0, 0), (-1, 0), (0, -1), (1, 1), (-1, -1)]:
                old, new = FakeSolver(center, 3, gradient), FakeSolver(center, 3, gradient)
                a, ar = SelectedDualSDMPC.decide(old, None, None, None, None)
                b, br = decide(new, None, None, None, None)
                self.assertEqual([r['budget'] for r in ar], [r['budget'] for r in br])
                self.assertEqual([r['request_origin'] for r in ar], [r['request_origin'] for r in br])
                self.assertEqual(a['budget'], b['budget'])

    def test_candidate_prices_independent_and_ten_bounded(self):
        for center in [(0, 0), (10, 1), (-50, 3000), (90, 5999), (90, 6000)]:
            solver = FakeSolver(center, 10)
            selected, rows = decide(solver, None, None, None, None)
            self.assertEqual(len(rows), 10)
            self.assertEqual(len({tuple(r['budget']) for r in rows}), 10)
            for r in rows:
                self.assertLessEqual(abs(r['budget'][0]-center[0]), 50)
                self.assertLessEqual(abs(r['budget'][1]-center[1]), 1000)
                self.assertTrue(0 <= r['budget'][1] <= 6000)
            self.assertTrue(all(np.array_equal(x, solver.incoming[0]) for x in solver.incoming))
            self.assertEqual(selected['evaluation'].total_ttt, min(r['evaluation'].total_ttt for r in rows))

    def test_failed_candidates_do_not_commit_prices(self):
        solver = FakeSolver((20, 3000), 10, feasible=False)
        before = solver.dual.copy()
        selected, rows = decide(solver, None, None, None, None)
        self.assertIsNone(selected)
        self.assertEqual(len(rows), 10)
        np.testing.assert_array_equal(before, solver.dual)

    def test_distance_observer_preserves_plant(self):
        from ttd_accounting import DistanceCapture, install_scalar
        # observer를 설치하기 전과 후의 동일 한 구간 plant를 비교한다.
        data = protocol(rc, cfg, 'sweet_170_w')
        profile = rc.FrozenProfile(data['forecast'], cfg.simulation.T_c_sec)
        before, after = rc.MixedTrafficSimulator(cfg), rc.MixedTrafficSimulator(cfg)
        u = rc.ControlAction.uncontrolled(cfg)
        before_log = before.step(u, profile.at(0), 0)
        install_scalar()
        with DistanceCapture(cfg) as capture:
            after_log = after.step(u, profile.at(0), 0)
        self.assertEqual(rc.to_plain_dict(before.state), rc.to_plain_dict(after.state))
        self.assertEqual(rc.to_plain_dict(before_log), rc.to_plain_dict(after_log))
        self.assertGreater(float(capture.total_ttd), 0.)

    def test_objective_accounting_and_budget_invariance(self):
        from sdmpc_objective import create_controller_class
        from fixed_controller import GridCoordinates
        data = protocol(rc, cfg, 'sweet_170_w')
        profile = rc.FrozenProfile(data['forecast'], cfg.simulation.T_c_sec)
        previous = rc.ControlAction.uncontrolled(cfg)
        coords = GridCoordinates(cfg, options, previous)
        z = coords.quantize(coords.encode(previous))
        values = []
        for alpha in (0., .018):
            solver = create_controller_class(alpha, 'ttd', 3)(cfg, options)
            solver.begin(rc.TrafficState.initial(cfg), profile.horizon(0, 3), previous)
            ev = solver.evaluate(z)
            self.assertAlmostEqual(ev.objective_value, ev.total_ttt-alpha*ev.total_ttd_veh_km, places=10)
            self.assertAlmostEqual(ev.objective_value, sum(ev.objective_player_costs.values()), places=10)
            values.append(ev)
        np.testing.assert_array_equal(values[0].budget_vector, values[1].budget_vector)
        self.assertEqual(values[0].total_ttt, values[1].total_ttt)
        self.assertEqual(rc.to_plain_dict(values[0].states), rc.to_plain_dict(values[1].states))


if __name__ == '__main__':
    unittest.main(verbosity=2)
