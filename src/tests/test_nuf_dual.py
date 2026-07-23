# N_UF dual λ_UF: dual 모드 signed 적분 갱신 방향 + 기본 equality 무영향(하위호환)
import unittest

from src.controllers.leader import LeaderAction
from src.controllers.wu_faithful_follower import WuFaithfulFollower
from src.models.demand import DemandProfile, ScenarioConfig
from src.models.state import ControlAction, ExperimentConfig, TrafficState


def _build_cfg(nuf_mode="equality"):
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


def _setup(cfg, follower):
    state = TrafficState.initial(cfg)
    demand = DemandProfile(
        cfg,
        ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
    ).horizon(0.0, 1)[0]
    snapshot = ControlAction.fixed(cfg)
    coupling = follower._wu._coupling(state, ControlAction.uncontrolled(cfg), demand)
    link = next(
        l for l in cfg.network.freeway_links
        if any(cfg.network.ramp_to_freeway.get(r) == l for r in cfg.network.ramps)
        and float(follower._wu._omega_f.get(l, 0.0)) > 0.0
    )
    return state, demand, snapshot, coupling, link


class TestNufDual(unittest.TestCase):
    def test_config_accepts_dual_mode(self):
        cfg = _build_cfg("dual")
        self.assertEqual(cfg.mpc.wu_faithful_nuf_coordination_mode, "dual")

    def test_positive_lambda_suppresses_release(self):
        # λ_UF>0(방류 억제) vs λ_UF<0(방류 보상): 방류 합이 순서대로여야.
        cfg = _build_cfg("dual")
        leader = LeaderAction(0.0, 3000.0)

        def release_at(lam):
            f = WuFaithfulFollower(cfg)
            f._lambda_UF = float(lam)
            state, demand, snapshot, coupling, link = _setup(cfg, f)
            _, meter, _ = f._solve_freeway_agent_metered(
                link, state, coupling, demand, snapshot, leader,
            )
            return sum(meter.values())

        rel_reward = release_at(-0.5)   # 음수 = 방류 보상 → 더 방류
        rel_neutral = release_at(0.0)
        rel_penalty = release_at(+0.5)  # 양수 = 방류 억제 → 덜 방류
        self.assertGreaterEqual(rel_reward + 1e-6, rel_neutral)
        self.assertGreaterEqual(rel_neutral + 1e-6, rel_penalty)
        self.assertGreater(rel_reward, rel_penalty)

    def test_lambda_update_signed_direction(self):
        # Σmeter > target → λ_UF 증가(억제), < target → 감소(보상). 부호 있음.
        cfg = _build_cfg("dual")
        f = WuFaithfulFollower(cfg)
        forecast = DemandProfile(
            cfg,
            ScenarioConfig("probe", urban_scale=1.0, freeway_scale=1.0, ramp_scale=1.0),
        ).horizon(0.0, 1)
        # 낮은 N_UF target → Σmeter가 초과 → λ_UF_next > 0.
        low = WuFaithfulFollower(cfg)
        low._lambda_UF = 0.0
        nash = low.solve(TrafficState.initial(cfg), LeaderAction(0.0, 100.0), forecast,
                         ControlAction.fixed(cfg))
        lam_next = nash.control.diagnostics.get("wu_faithful_lambda_uf_next")
        self.assertIsNotNone(lam_next)
        self.assertGreater(float(lam_next), 0.0,
                           msg="Σmeter > low target → λ_UF must rise (penalize release)")


if __name__ == "__main__":
    unittest.main()
