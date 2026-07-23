# local_signal_plant.rollout_local_tts_ramp_aware의 off-ramp drain(S_eff 부호) 단위 테스트
"""`rollout_local_tts_ramp_aware` step (b)(off_ramp drain)의 S_eff 부호 회귀 테스트.

`s_eff`는 receiving 링크의 **가용공간**이다(`_allocate_receiving_counts`/
`_effective_available_space`와 동일 의미). 차량이 그 링크로 유입되면 가용공간은
반드시 감소해야 한다 — `_drain_offramp_storage`(urban_queue_model.py)가
`state.urban_link_storage[receiving_link] -= actual`로 구현하는 것과 동일해야 한다.

시나리오: off-ramp storage(occupancy=40)에서 같은 receiving 링크("own_link", 가용공간
초기 5)로 방출을 시도하는 movement 2개를 순서대로 처리시킨다. 이 링크는 이 신호의
다른(non-off-ramp) movement의 origin이기도 해서(own_origin_links에 포함) step (b)의
갱신 분기가 실제로 실행된다.

- 부호가 맞으면(가용공간 감소): 1번째 movement가 가용공간 5를 모두 소진 → 2번째
  movement는 막힘 → 이 substep에 총 5대만 배출 → storage 잔류 occupancy = 35 →
  own-TTS(반환 cost) = 35.0.
- 부호가 틀리면(가용공간이 오히려 증가): 1번째 배출이 가용공간을 5+5=10으로 늘려버려
  2번째 movement도 유입돼 총 15대 배출(용량 보존 위반, occupancy=25 → cost=25.0).
"""
import unittest

from src.controllers.local_signal_plant import (
    LocalSignalModel,
    rollout_local_tts,
    rollout_local_tts_ramp_aware,
)
from src.models.state import ExperimentConfig


def _build_model(cfg: ExperimentConfig) -> LocalSignalModel:
    # off_m1/off_m2: 같은 off-ramp(OR_X)에서 같은 receiving 링크("own_link")로 배출.
    # internal_m: origin이 "own_link"인 일반 movement 하나를 둬서 "own_link"가
    # 이 신호의 own_origin_links에 포함되도록 한다(그래야 step (b)의 자기-origin
    # 갱신 분기가 실제로 실행된다).
    return LocalSignalModel(
        signal="X",
        cfg=cfg,
        movements=["off_m1", "off_m2", "internal_m"],
        specs={"off_m1": {}, "off_m2": {}, "internal_m": {}},
        phase_of={"off_m1": "p1", "off_m2": "p1", "internal_m": "p2"},
        receiving_of={
            "off_m1": "own_link",
            "off_m2": "own_link",
            "internal_m": "sink_link",
        },
        origin_of={"off_m1": "OR_X", "off_m2": "OR_X", "internal_m": "own_link"},
        beta_of={"off_m1": 0.5, "off_m2": 0.5, "internal_m": 0.0},
        cap_flow_of={"off_m1": 1.0e5, "off_m2": 1.0e5, "internal_m": 0.0},
        receiving_space_rule="proportional",
        kind_of={"off_m1": "off_ramp", "off_m2": "off_ramp", "internal_m": "internal"},
        offramp_movements={"OR_X": ["off_m1", "off_m2"]},
        offramp_storage_cap={"OR_X": 100.0},
        onramp_movements={},
        ramp_queue_max=180.0,
        has_ramps=True,
    )


class TestOffRampDrainSignConvention(unittest.TestCase):
    def test_offramp_drain_decreases_receiving_space(self):
        cfg = ExperimentConfig()
        model = _build_model(cfg)

        s_eff0 = {"own_link": 5.0}
        occ0 = {"OR_X": 40.0}

        cost = rollout_local_tts_ramp_aware(
            model,
            q0={"internal_m": 0.0},
            arr_movement={},
            s_eff0=s_eff0,
            offramp_inflow={"OR_X": 0.0},
            offramp_occ0=occ0,
            ramp_queue0={},
            reservoir_drain={},
            freeway_congestion={},
            ramp_metering_weight=0.0,
            green_p1=120.0,
            green_p2=0.0,
            substeps=1,
            dt_h=1.0,
        )

        # 용량 보존: receiving 링크의 가용공간(5)보다 많이 배출될 수 없으므로,
        # storage 잔류 occupancy는 40 - 5 = 35 이상이어야 하고, own-TTS(cost)도
        # 그만큼(35.0) 나와야 한다. 부호가 반전되면(버그) 가용공간이 오히려 늘어나
        # 두 번째 movement까지 통과해 15대가 빠져나가 cost가 25.0으로 더 작게 나온다.
        self.assertAlmostEqual(cost, 35.0, places=6)


def _build_full_model(cfg: ExperimentConfig) -> LocalSignalModel:
    # P1.5 테스트용: 세 가지 서비스 지점을 전부 갖는 모델 —
    # (a) on-ramp request(ramp_m), (b) off-ramp drain(off_m), (d) 일반 discharge(in_m).
    return LocalSignalModel(
        signal="Y",
        cfg=cfg,
        movements=["in_m", "ramp_m", "off_m", "internal_m"],
        specs={"in_m": {}, "ramp_m": {}, "off_m": {}, "internal_m": {}},
        phase_of={"in_m": "p1", "ramp_m": "p1", "off_m": "p2", "internal_m": "p2"},
        receiving_of={
            "in_m": "own_link",
            "ramp_m": "",
            "off_m": "own_link",
            "internal_m": "sink_link",
        },
        origin_of={"in_m": "", "ramp_m": "", "off_m": "OR_Y", "internal_m": "own_link"},
        beta_of={"in_m": 1.0, "ramp_m": 1.0, "off_m": 1.0, "internal_m": 0.0},
        cap_flow_of={"in_m": 900.0, "ramp_m": 600.0, "off_m": 700.0, "internal_m": 500.0},
        receiving_space_rule="proportional",
        kind_of={
            "in_m": "boundary_in",
            "ramp_m": "on_ramp",
            "off_m": "off_ramp",
            "internal_m": "internal",
        },
        offramp_movements={"OR_Y": ["off_m"]},
        offramp_storage_cap={"OR_Y": 100.0},
        onramp_movements={"R_Y": ["ramp_m"]},
        ramp_queue_max=180.0,
        has_ramps=True,
    )


class TestPhaseResolvedRampAware(unittest.TestCase):
    """P1.5: rollout_local_tts_ramp_aware의 arr_by_substep/gf_by_substep 확장 회귀."""

    def test_uniform_profiles_match_legacy_exactly(self):
        # 하위호환 동등성: 균일 도착 프로파일 + 상수 gf(green/cycle)를 명시로 주면
        # 인자 없는 legacy 호출과 비용이 정확히 동일해야 한다.
        cfg = ExperimentConfig()
        model = _build_full_model(cfg)
        cycle = max(cfg.network.cycle_length, 1.0e-9)
        green_p1, green_p2 = 70.0, 42.0
        substeps, dt_h = 6, 5.0 / 3600.0
        q0 = {"in_m": 8.0, "ramp_m": 5.0, "internal_m": 3.0}
        arr_movement = {"in_m": 400.0, "ramp_m": 300.0, "internal_m": 200.0}
        common = dict(
            q0=q0,
            arr_movement=arr_movement,
            s_eff0={"own_link": 50.0, "sink_link": 50.0},
            offramp_inflow={"OR_Y": 250.0},
            offramp_occ0={"OR_Y": 30.0},
            ramp_queue0={"R_Y": 20.0},
            reservoir_drain={"R_Y": 500.0},
            freeway_congestion={"R_Y": 0.4},
            ramp_metering_weight=0.0,
            green_p1=green_p1,
            green_p2=green_p2,
            substeps=substeps,
            dt_h=dt_h,
        )
        legacy_cost = rollout_local_tts_ramp_aware(model, **common)
        gf_const = {"p1": green_p1 / cycle, "p2": green_p2 / cycle}
        arr_by_substep = {m: [arr_movement.get(m, 0.0)] * substeps for m in q0}
        gf_by_substep = {
            m: [gf_const[model.phase_of[m]]] * substeps for m in model.movements
        }
        explicit_cost = rollout_local_tts_ramp_aware(
            model, **common, arr_by_substep=arr_by_substep, gf_by_substep=gf_by_substep,
        )
        self.assertEqual(explicit_cost, legacy_cost)

    def test_green_window_alignment_discriminates(self):
        # 시간구조 판별력: 같은 도착 총량(burst) + 같은 green 총량에서, green window가
        # burst와 정렬된 gf는 비용이 낮고 어긋난 gf는 높아야 한다(비용이 달라야 한다).
        cfg = ExperimentConfig()
        model = _build_full_model(cfg)
        substeps, dt_h = 4, 1.0
        base = dict(
            q0={"in_m": 0.0, "ramp_m": 0.0, "internal_m": 0.0},
            arr_movement={},
            s_eff0={"own_link": 1.0e9, "sink_link": 1.0e9},
            offramp_inflow={"OR_Y": 0.0},
            offramp_occ0={"OR_Y": 0.0},
            ramp_queue0={"R_Y": 0.0},
            reservoir_drain={"R_Y": 0.0},
            freeway_congestion={"R_Y": 0.0},
            ramp_metering_weight=0.0,
            green_p1=56.0,
            green_p2=56.0,
            substeps=substeps,
            dt_h=dt_h,
        )
        # burst: in_m 도착이 앞 2 substep에만 몰림(총량 = 10+10).
        burst = {"in_m": [10.0, 10.0, 0.0, 0.0]}
        zero = [0.0] * substeps
        # 같은 green 총량(mean gf 0.25): 정렬 = burst substep에 green, 어긋남 = 뒤 2 substep.
        gf_aligned = {
            m: ([0.5, 0.5, 0.0, 0.0] if m == "in_m" else zero) for m in model.movements
        }
        gf_misaligned = {
            m: ([0.0, 0.0, 0.5, 0.5] if m == "in_m" else zero) for m in model.movements
        }
        cost_aligned = rollout_local_tts_ramp_aware(
            model, **base, arr_by_substep=burst, gf_by_substep=gf_aligned,
        )
        cost_misaligned = rollout_local_tts_ramp_aware(
            model, **base, arr_by_substep=burst, gf_by_substep=gf_misaligned,
        )
        self.assertNotEqual(cost_aligned, cost_misaligned)
        self.assertLess(cost_aligned, cost_misaligned)


def _build_nonramp_model(cfg: ExperimentConfig) -> LocalSignalModel:
    # 오라클 probe 테스트용 단순 non-ramp 신호: m1이 own_link로 방출하고 own_link는
    # 이 신호 다른 movement(m2)의 origin이라 own_origin_links에 포함 → ego 방출로 s_eff가
    # 전진 중 감소한다(그래야 상수 oracle 리셋이 legacy와 갈린다).
    return LocalSignalModel(
        signal="Z",
        cfg=cfg,
        movements=["m1", "m2"],
        specs={"m1": {}, "m2": {}},
        phase_of={"m1": "p1", "m2": "p2"},
        receiving_of={"m1": "own_link", "m2": "sink_link"},
        origin_of={"m1": "src_link", "m2": "own_link"},
        beta_of={"m1": 1.0, "m2": 0.0},
        cap_flow_of={"m1": 1.0e5, "m2": 0.0},
        receiving_space_rule="proportional",
        kind_of={"m1": "internal", "m2": "internal"},
        offramp_movements={},
        offramp_storage_cap={},
        onramp_movements={},
        ramp_queue_max=180.0,
        has_ramps=False,
    )


class TestSEffBySubstepInjection(unittest.TestCase):
    """오라클 probe: rollout_local_tts의 s_eff_by_substep 주입(None-게이트 하위호환)."""

    def _common(self):
        cfg = ExperimentConfig()
        model = _build_nonramp_model(cfg)
        return dict(
            model=model,
            q0={"m1": 30.0, "m2": 0.0},
            arr_movement={},
            s_eff0={"own_link": 5.0, "sink_link": 1.0e9},
            green_p1=120.0,
            green_p2=0.0,
            substeps=3,
            dt_h=1.0,
        )

    def test_none_matches_legacy_exactly(self):
        # s_eff_by_substep=None이면 legacy 호출과 비트 동일.
        common = self._common()
        legacy = rollout_local_tts(**common)
        with_none = rollout_local_tts(**common, s_eff_by_substep=None)
        self.assertEqual(with_none, legacy)

    def test_constant_oracle_reset_nullifies_ego_reduction(self):
        # own_link S_eff를 매 substep 상수 5로 리셋하면, legacy에서 ego 방출로 감소하던
        # own_link 가용공간이 매 substep 복원돼 더 많은 배출이 가능 → cost가 legacy보다 작다.
        # 이는 주입이 실제로 s_eff를 덮어쓴다는 판별(먹히지 않으면 값이 같을 것).
        common = self._common()
        legacy = rollout_local_tts(**common)
        s_eff_by_substep = {"own_link": [5.0] * common["substeps"]}
        oracle = rollout_local_tts(**common, s_eff_by_substep=s_eff_by_substep)
        self.assertNotEqual(oracle, legacy)
        self.assertLess(oracle, legacy)


if __name__ == "__main__":
    unittest.main()
