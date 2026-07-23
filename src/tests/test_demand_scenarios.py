from __future__ import annotations

import math

from src.models.demand import DemandProfile, load_scenarios
from src.models.metanet import effective_lane_profile
from src.models.state import ExperimentConfig, TrafficState


def test_canonical_low_medium_peak_demand_scales() -> None:
    scenarios = load_scenarios("src/config/scenarios.yaml")

    assert list(scenarios) == [
        "low_demand",
        "medium_demand",
        "peak_demand",
        "medium_incident_east",
        "medium_urban_west_skew",
        "medium_surge",
    ]

    assert math.isclose(scenarios["low_demand"].urban_scale, 1.0)
    assert math.isclose(scenarios["low_demand"].freeway_scale, 1.0)
    assert math.isclose(scenarios["low_demand"].ramp_scale, 1.0)

    assert math.isclose(scenarios["medium_demand"].urban_scale, 1.0375)
    assert math.isclose(scenarios["medium_demand"].freeway_scale, 1.03)
    assert math.isclose(scenarios["medium_demand"].ramp_scale, 1.0375)

    assert math.isclose(scenarios["peak_demand"].urban_scale, 1.25)
    assert math.isclose(scenarios["peak_demand"].freeway_scale, 1.20)
    assert math.isclose(scenarios["peak_demand"].ramp_scale, 1.25)


def test_canonical_storage_cap_density() -> None:
    cfg = ExperimentConfig.from_file("src/config/default.yaml")

    assert math.isclose(cfg.network.rho_max, 95.01964207118104)


def test_canonical_demand_levels_are_strictly_ordered() -> None:
    cfg = ExperimentConfig.from_file("src/config/default.yaml")
    scenarios = load_scenarios("src/config/scenarios.yaml")
    demands = {
        name: DemandProfile(cfg, scenarios[name]).at(0.0)
        for name in ("low_demand", "medium_demand", "peak_demand")
    }

    for demand_field in ("freeway_mainline", "urban_boundary", "ramp_arrival"):
        low = getattr(demands["low_demand"], demand_field)
        medium = getattr(demands["medium_demand"], demand_field)
        peak = getattr(demands["peak_demand"], demand_field)
        assert low.keys() == medium.keys() == peak.keys()
        for key in low:
            assert low[key] < medium[key] < peak[key]


def test_medium_incident_closes_only_east_downstream_lane() -> None:
    cfg = ExperimentConfig.from_file("src/config/default.yaml")
    scenario = load_scenarios("src/config/scenarios.yaml")["medium_incident_east"]
    profile = DemandProfile(cfg, scenario)

    assert profile.at(2399.0).freeway_lane_loss == {}
    assert profile.at(2400.0).freeway_lane_loss == {"FW_E": {3: 1.0}}
    assert profile.at(4799.0).freeway_lane_loss == {"FW_E": {3: 1.0}}
    assert profile.at(4800.0).freeway_lane_loss == {}


def test_medium_incident_lane_loss_reaches_plant_profile() -> None:
    cfg = ExperimentConfig.from_file("src/config/default.yaml")
    scenario = load_scenarios("src/config/scenarios.yaml")["medium_incident_east"]
    demand = DemandProfile(cfg, scenario).at(3000.0)
    state = TrafficState.initial(cfg)

    lanes, diagnostics = effective_lane_profile(state, cfg, demand)

    assert lanes["FW_E"][3] == 1.0
    assert all(math.isclose(value, 2.0) for value in lanes["FW_W"])
    assert diagnostics["incident_lane_closure_active"] == 1.0


def test_medium_west_skew_preserves_total_and_sets_two_to_one_ratio() -> None:
    cfg = ExperimentConfig.from_file("src/config/default.yaml")
    scenarios = load_scenarios("src/config/scenarios.yaml")
    medium = DemandProfile(cfg, scenarios["medium_demand"]).at(1800.0)
    skew = DemandProfile(cfg, scenarios["medium_urban_west_skew"]).at(1800.0)

    west = sum(skew.urban_boundary[link] for link in ("in_A_left", "in_D_left"))
    east = sum(skew.urban_boundary[link] for link in ("in_C_right", "in_F_right"))
    assert math.isclose(west, 2.0 * east)
    assert math.isclose(
        sum(skew.urban_boundary.values()),
        sum(medium.urban_boundary.values()),
    )


def test_medium_surge_returns_to_medium_after_event() -> None:
    cfg = ExperimentConfig.from_file("src/config/default.yaml")
    scenarios = load_scenarios("src/config/scenarios.yaml")
    medium = DemandProfile(cfg, scenarios["medium_demand"])
    surge = DemandProfile(cfg, scenarios["medium_surge"])

    before = surge.at(1800.0)
    peak = surge.at(3000.0)
    after = surge.at(4200.0)
    assert math.isclose(
        sum(before.freeway_mainline.values()),
        sum(medium.at(1800.0).freeway_mainline.values()),
    )
    assert math.isclose(
        sum(peak.freeway_mainline.values()),
        1.15 * sum(medium.at(3000.0).freeway_mainline.values()),
    )
    assert math.isclose(
        sum(after.freeway_mainline.values()),
        sum(medium.at(4200.0).freeway_mainline.values()),
    )
