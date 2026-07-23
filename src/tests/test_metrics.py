import unittest

from src.evaluation.metrics import (
    _balance_from_rows,
    _net_inflow_tracking_from_rows,
    boundary_cv,
    improvement_rate,
)
from src.models.state import ExperimentConfig
from src.models.urban_queue_model import safe_balance_index


def _cfg():
    return ExperimentConfig.from_file("src/config/default.yaml")


class MetricTests(unittest.TestCase):
    def test_improvement_rate_lower_is_better(self):
        self.assertAlmostEqual(improvement_rate(100.0, 92.0, "lower_is_better"), 8.0)

    def test_improvement_rate_higher_is_better(self):
        self.assertAlmostEqual(improvement_rate(100.0, 108.0, "higher_is_better"), 8.0)

    def test_boundary_cv_zero_queue_case(self):
        self.assertEqual(boundary_cv([0.0, 0.0, 0.0]), 0.0)

    def test_balance_index_equal_queue_is_zero_or_near_zero(self):
        self.assertLessEqual(safe_balance_index([10.0, 10.0, 10.0]), 1e-12)

    def test_balance_index_unbalanced_queue_is_positive(self):
        self.assertGreater(safe_balance_index([1.0, 9.0, 20.0]), 0.0)


class GateRedefinitionTests(unittest.TestCase):
    """acceptance 게이트 재정의(부하구간 시간집계 B, dN_P/dt 추적) 단위 검증."""

    def _balance_row(self, b_in, b_out, degenerate):
        return {
            "B_in": b_in,
            "B_out": b_out,
            "boundary_balance_degenerate": degenerate,
            "boundary_empty_ratio": 1.0 if degenerate else 0.1,
            "boundary_saturation_ratio": 0.0,
            "boundary_in_empty_ratio": 0.0,
            "boundary_out_empty_ratio": 0.0,
            "boundary_in_saturation_ratio": 0.0,
            "boundary_out_saturation_ratio": 0.0,
        }

    def test_balance_aggregates_over_controllable_intervals_only(self):
        cfg = _cfg()
        # 부하 구간(비-degenerate) 2개 + 피크 후 공큐 구간 2개: B는 부하 구간 평균,
        # 제어가능 비율 0.5 ≥ 0.25라 degenerate 아님("잘 비운 성공"을 무효화하지 않음).
        rows = [
            self._balance_row(0.02, 0.01, 0.0),
            self._balance_row(0.04, 0.03, 0.0),
            self._balance_row(0.0, 0.0, 1.0),
            self._balance_row(0.0, 0.0, 1.0),
        ]
        out = _balance_from_rows(rows, cfg)
        self.assertAlmostEqual(out["B_in"], 0.03)
        self.assertAlmostEqual(out["B_out"], 0.02)
        self.assertEqual(out["boundary_balance_degenerate"], 0.0)
        self.assertAlmostEqual(out["boundary_controllable_fraction"], 0.5)

    def test_balance_is_load_weighted(self):
        cfg = _cfg()
        # 경부하(잔여 5대, B=0.2 노이즈)와 중부하(500대, B=0.01) interval:
        # 부하 가중이면 B_in ≈ 중부하 값에 수렴해야 한다(스케일 불변 노이즈 차단).
        light = self._balance_row(0.2, 0.0, 0.0)
        light["boundary_in_load_veh"] = 5.0
        heavy = self._balance_row(0.01, 0.0, 0.0)
        heavy["boundary_in_load_veh"] = 500.0
        out = _balance_from_rows([light, heavy], cfg)
        expected = (0.2 * 5.0 + 0.01 * 500.0) / 505.0
        self.assertAlmostEqual(out["B_in"], expected)
        self.assertLess(out["B_in"], 0.02)

    def test_balance_degenerate_when_controllable_fraction_too_low(self):
        cfg = _cfg()
        rows = [self._balance_row(0.0, 0.0, 1.0) for _ in range(9)]
        rows.append(self._balance_row(0.02, 0.01, 0.0))
        out = _balance_from_rows(rows, cfg)
        self.assertEqual(out["boundary_balance_degenerate"], 1.0)
        self.assertAlmostEqual(out["boundary_controllable_fraction"], 0.1)

    def test_net_inflow_tracking_uses_accumulation_rate(self):
        cfg = _cfg()
        t_c_h = cfg.simulation.T_c_h
        window = max(1, int(round(cfg.leader.N_P_feedback_horizon_h / t_c_h)))
        # N_P가 매 interval 5대씩 증가하고 목표도 5/T_c_h veh/h이면 추적오차 0.
        rate = 5.0 / t_c_h
        rows = [
            {"urban_accumulation_veh": 100.0 + 5.0 * i, "net_inflow_target": rate}
            for i in range(window + 3)
        ]
        self.assertAlmostEqual(_net_inflow_tracking_from_rows(rows, cfg), 0.0, places=6)
        # 목표는 0인데 N_P가 같은 속도로 늘면 오차 = rate.
        rows_off = [
            {"urban_accumulation_veh": 100.0 + 5.0 * i, "net_inflow_target": 0.0}
            for i in range(window + 3)
        ]
        self.assertAlmostEqual(_net_inflow_tracking_from_rows(rows_off, cfg), rate, places=6)


if __name__ == "__main__":
    unittest.main()
