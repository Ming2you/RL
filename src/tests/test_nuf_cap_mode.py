# N_UF 조정 모드(equality|cap): cap이 자율 metering을 존중하고 budget 초과만 누르는지 검증
import unittest

from src.controllers.leader import LeaderAction
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def _build_cfg(nuf_mode: str):
    return ExperimentConfig.from_file(
        "src/config/default.yaml",
        {
            "simulation": {"T_total": 360.0},
            "mpc": {
                "horizon_steps": 1,
                "relaxed_quantized_controls": True,
                "grid_parallel_backend": "serial",
                "wu_faithful_nuf_coordination_mode": nuf_mode,
            },
            "freeway_follower": {
                "freeway_prediction_horizon_steps": 1,
                "vsl_sequence_search": False,
            },
        },
    )


def _setup(cfg):
    follower = WuFaithfulFollower(cfg)
    state = TrafficState.initial(cfg)
    demand = DemandProfile(
        cfg,
        ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
    ).horizon(0.0, 1)[0]
    snapshot = ControlAction.fixed(cfg)
    control = ControlAction.uncontrolled(cfg)
    coupling = follower._wu._coupling(state, control, demand)
    # 소유 ramp가 있고 ω_F>0인 link를 하나 고른다(budget이 유의하게 정의되는 곳).
    link = next(
        l for l in cfg.network.freeway_links
        if any(cfg.network.ramp_to_freeway.get(r) == l for r in cfg.network.ramps)
        and float(follower._wu._omega_f.get(l, 0.0)) > 0.0
    )
    return follower, state, demand, snapshot, coupling, link


class TestNufCapMode(unittest.TestCase):
    def test_cap_with_loose_budget_matches_autonomous(self):
        # budget ≥ Σcap이면 cap은 어디에도 안 닿는다 → 자율(PFO, leader=None) metering과 동일.
        cfg = _build_cfg("cap")
        follower, state, demand, snapshot, coupling, link = _setup(cfg)
        _, meter_auto, _ = follower._solve_freeway_agent_metered(
            link, state, coupling, demand, snapshot, None,
        )
        loose = LeaderAction(0.0, 1.0e9)
        _, meter_cap, _ = follower._solve_freeway_agent_metered(
            link, state, coupling, demand, snapshot, loose,
        )
        self.assertEqual(set(meter_auto), set(meter_cap))
        for ramp, value in meter_auto.items():
            self.assertAlmostEqual(
                meter_cap[ramp], value, places=6,
                msg=f"loose-budget cap must not override autonomous metering ({ramp})",
            )

    def test_cap_binds_only_as_upper_bound(self):
        # 빡빡한 budget: cap 모드 합 ≤ budget(상한), equality 모드 합 == budget(등식).
        follower_e = None
        for nuf_mode in ("cap", "equality"):
            cfg = _build_cfg(nuf_mode)
            follower, state, demand, snapshot, coupling, link = _setup(cfg)
            owned = [r for r in cfg.network.ramps if cfg.network.ramp_to_freeway.get(r) == link]
            caps_sum = sum(float(cfg.network.ramp_capacity_veh_h[r]) for r in owned)
            omega_f = float(follower._wu._omega_f.get(link, 0.0))
            # link budget이 Σcap의 ~30%가 되도록 N_UF_star 선택.
            n_uf_star = 0.3 * caps_sum / omega_f
            budget = min(max(omega_f * n_uf_star, 0.0), caps_sum)
            _, meter, _ = follower._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, LeaderAction(0.0, n_uf_star),
            )
            total = sum(meter.values())
            if nuf_mode == "cap":
                self.assertLessEqual(
                    total, budget + 1.0e-6,
                    msg="cap mode must keep link metering sum within budget",
                )
            else:
                self.assertAlmostEqual(
                    total, budget, places=6,
                    msg="equality mode must realize the budget exactly",
                )


if __name__ == "__main__":
    unittest.main()
