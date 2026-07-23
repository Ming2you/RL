# capacity drop(anticipation ν regime-split) 단위테스트: toggle 동작 + 혼잡 시 속도/flow 저하 방향 검증
from __future__ import annotations

import types
import unittest

from src.models.metanet import metanet_speed_update_kmh, select_anticipation_nu


def _net(toggle: bool, nu_free=65.0, nu_cong=120.0, rho_crit=33.5):
    return types.SimpleNamespace(
        capacity_drop_anticipation=toggle,
        metanet_nu_km2_h=nu_free,
        metanet_nu_cong_km2_h=nu_cong,
        rho_crit=rho_crit,
    )


class TestCapacityDropSelection(unittest.TestCase):
    def test_toggle_off_always_free(self):
        net = _net(False)
        self.assertEqual(select_anticipation_nu(10.0, net), 65.0)
        self.assertEqual(select_anticipation_nu(100.0, net), 65.0)  # 혼잡이어도 off면 ν_free

    def test_toggle_on_regime_switch(self):
        net = _net(True)
        self.assertEqual(select_anticipation_nu(10.0, net), 65.0)   # ρ<ρ_crit → ν_free
        self.assertEqual(select_anticipation_nu(100.0, net), 120.0)  # ρ>ρ_crit → ν_cong
        self.assertEqual(select_anticipation_nu(33.5, net), 65.0)    # 경계(같음)는 free


class TestCapacityDropMechanism(unittest.TestCase):
    def test_higher_nu_lowers_speed_in_congested_gradient(self):
        # 하류가 더 혼잡한 gradient(downstream_rho > rho)에서 ν_cong>ν_free면 감속이 커져
        # 속도가 더 낮아진다(capacity drop 방향). flow=ρ·v이므로 flow도 더 낮다.
        common = dict(
            speed=60.0, upstream_speed=60.0, rho=40.0, downstream_rho=80.0,
            v_eff=55.0, dt_h=0.0028, length_km=0.5, tau_h=0.005,
            kappa_veh_km_lane=40.0, v_min=5.0,
        )
        v_free = metanet_speed_update_kmh(nu_km2_h=65.0, **common)
        v_cong = metanet_speed_update_kmh(nu_km2_h=120.0, **common)
        self.assertLess(v_cong, v_free)
        self.assertLess(40.0 * v_cong, 40.0 * v_free)  # flow도 저하


if __name__ == "__main__":
    unittest.main()
