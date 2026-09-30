"""Synthetic tests only: no environments, models, or production trajectories."""

import copy
import json

import numpy as np
import pytest

from exploration import LocalBudgetPolicy


def row_for(action, audit, executed=None, selection="lower_solution", reference="previous"):
    return {
        "action_anchor": audit["action_anchor"],
        "action_requested": action,
        "B_requested": [audit["projected_request"]],
        "B_executed": audit["projected_request"] if executed is None else executed,
        "selection_source": selection,
        "reference_source": reference,
        "reference_recovery": reference == "pfo_recovery",
        "guard_mode": "physical",
    }


def roundtrip(value):
    return json.loads(json.dumps(value, allow_nan=False))


class FixedNoise:
    def __init__(self, noise):
        self.noise = np.asarray(noise, dtype=float)

    def normal(self, size):
        assert size == 2
        return self.noise.copy()


def test_exact_formula_from_actual_anchor_and_final_float32():
    policy = LocalBudgetPolicy("local", 6901, 6000)
    rng = np.random.default_rng(6901)
    anchor = np.array([-123.456, 3456.789])
    action, audit = policy.choose(anchor)
    offset = np.clip([10, 100] * rng.normal(size=2), [-50, -500], [50, 500])
    desired = anchor + offset
    expected_action = ((desired - anchor) / [50, 1000]).astype(np.float32)
    request = anchor + [50, 1000] * expected_action.astype(np.float64)
    np.testing.assert_array_equal(action, expected_action)
    assert action.dtype == np.float32
    assert audit["base_before"] == anchor.tolist()
    assert audit["realized_offset_before"] == [0, 0]
    np.testing.assert_array_equal(audit["desired_offset"], offset)
    np.testing.assert_array_equal(audit["desired_budget"], desired)
    np.testing.assert_array_equal(audit["projected_request"], request)
    assert audit["desired_budget"] != audit["projected_request"]
    committed = policy.commit(row_for(action, audit, anchor + [30, -80]))
    assert committed["realized_offset_after"] == [30, -80]
    assert committed["committed"] is True
    next_anchor = anchor + [30, -80]
    action2, audit2 = policy.choose(next_anchor)
    offset2 = np.clip(0.8 * np.array([30, -80]) + [10, 100] * rng.normal(size=2),
                      [-50, -500], [50, 500])
    np.testing.assert_array_equal(audit2["desired_offset"], offset2)
    np.testing.assert_array_equal(action2, ((anchor + offset2 - next_anchor) / [50, 1000])
                                  .astype(np.float32))
    assert roundtrip(committed) == committed


def test_zero_noise_mean_reversion_uses_executed_offset():
    policy = LocalBudgetPolicy("local", 0, 6000)
    action, audit = policy.choose([100, 3000])
    policy.commit(row_for(action, audit, [130, 2800]))
    policy._rng = FixedNoise([0, 0])
    action, audit = policy.choose([130, 2800])
    assert audit["desired_offset"] == [24, -160]
    assert audit["desired_budget"] == [124, 2840]
    np.testing.assert_array_equal(action, np.array([-0.12, 0.04], dtype=np.float32))


@pytest.mark.parametrize("anchor", [[-20, -50], [-20, 6050], [25, 2400]])
def test_carry_projects_anchor_with_zero_action_and_no_rng_draws(anchor):
    policy = LocalBudgetPolicy("carry", 6901, 6000)
    initial_rng = policy.state_dict()["rng"]
    expected = [anchor[0], float(np.clip(anchor[1], 0, 6000))]
    for _ in range(3):
        action, audit = policy.choose(anchor)
        np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))
        assert audit["desired_budget"] == expected
        assert audit["projected_request"] == expected
        assert audit["rate_limited"] == [False, False]
        assert policy.state_dict()["rng"] == initial_rng
        committed = policy.commit(row_for(action, audit, [anchor[0] + 1, expected[1]]))
        assert committed["realized_offset_after"] == [1, expected[1] - anchor[1]]
        assert policy.state_dict()["rng"] == initial_rng


@pytest.mark.parametrize("anchor,noise", [([-10, 6000], [1, 2]), ([-10, 0], [-1, -2])])
def test_nuf_boundaries_do_not_accumulate_fictitious_offsets(anchor, noise):
    policy = LocalBudgetPolicy("local", 6901, 6000)
    policy._rng = FixedNoise(noise)
    action, audit = policy.choose(anchor)
    assert audit["desired_nuf_projected"] is True
    assert audit["desired_offset"][1] == noise[1] * 100
    assert action[1] == 0
    assert audit["projected_request"][1] == anchor[1]
    assert audit["requested_offset"][1] == 0
    committed = policy.commit(row_for(action, audit))
    assert committed["realized_offset_after"][1] == 0
    policy._rng = FixedNoise([0, 0])
    _, next_audit = policy.choose(committed["executed_budget"])
    assert next_audit["desired_offset"][1] == 0


@pytest.mark.parametrize("drift,expected", [([1000, 3000], [-1, -1]),
                                          ([-1000, -3000], [1, 1])])
def test_rate_limits_and_signed_np(drift, expected):
    policy = LocalBudgetPolicy("local", 0, 6000)
    base = np.array([-100, 3000])
    action, audit = policy.choose(base)
    anchor = base + drift
    policy.commit(row_for(action, audit, anchor))
    policy._rng = FixedNoise([0, 0])
    action, audit = policy.choose(anchor)
    np.testing.assert_array_equal(action, np.array(expected, dtype=np.float32))
    assert audit["rate_limited"] == [True, True]
    assert audit["offset_limited"] == [True, True]
    assert audit["desired_budget"][0] < 0
    assert audit["desired_budget"] != audit["projected_request"]
    np.testing.assert_array_equal(audit["projected_request"], anchor + [50, 1000] * action)


@pytest.mark.parametrize("anchor,noise,expected", [(0, 1, 1), (1, -1, 0)])
def test_final_float32_rounding_is_projected_again(anchor, noise, expected):
    policy = LocalBudgetPolicy("local", 0, 1)
    policy._rng = FixedNoise([0, noise])
    action, audit = policy.choose([0, anchor])
    assert audit["raw_request"][1] != expected
    assert audit["request_nuf_projected"] is True
    assert audit["projected_request"][1] == expected
    policy.commit(row_for(action, audit))


@pytest.mark.parametrize("behavior", ["carry", "local"])
@pytest.mark.parametrize("selection,reference,reasons", [
    ("lower_solution", "previous", []),
    ("lower_solution", "pfo_initial", []),
    ("reference_fallback", "pfo_initial", ["reference_fallback"]),
    ("reference_fallback", "previous", ["reference_fallback"]),
    ("lower_solution", "pfo_recovery", ["pfo_recovery"]),
    ("reference_fallback", "pfo_recovery", ["reference_fallback", "pfo_recovery"]),
])
def test_only_explicit_physical_flags_rebase_after_execution(behavior, selection, reference, reasons):
    policy = LocalBudgetPolicy(behavior, 4, 6000)
    action, audit = policy.choose([-100, 3000])
    row = row_for(action, audit, [-125, 3300], selection, reference)

    class RelevantFieldsOnly(dict):
        def __getitem__(self, key):
            assert key in {"action_anchor", "action_requested", "B_requested", "B_executed",
                           "selection_source", "reference_source", "reference_recovery", "guard_mode"}
            return super().__getitem__(key)

    row.update(selected_TTT=float("inf"), reference_TTT=-1e99, interval_ttt=float("nan"),
               Q=object(), price=object(), h3_guard_would_reject=True)
    committed = policy.commit(RelevantFieldsOnly(row))
    assert committed["base_before"] == [-100, 3000]
    assert committed["executed_offset"] == [-25, 300]
    assert committed["base_after"] == ([-125, 3300] if reasons else [-100, 3000])
    assert committed["realized_offset_after"] == ([0, 0] if reasons else [-25, 300])
    assert committed["rebase_reasons"] == reasons
    assert committed["rebased"] == bool(reasons)
    assert policy.state_dict()["count"] == 1
    assert roundtrip(committed) == committed


@pytest.mark.parametrize("behavior", ["carry", "local"])
def test_lifecycle_and_independent_action_audit_and_state_copies(behavior):
    policy = LocalBudgetPolicy(behavior, 0, 6000)
    with pytest.raises(RuntimeError):
        policy.commit({})
    action, audit = policy.choose([10, 3000])
    expected = policy.state_dict()
    row = row_for(action.copy(), copy.deepcopy(audit))
    action[:] = 99
    audit["base_before"][0] = 99
    state = policy.state_dict()
    state["base"][0] = 999
    state["pending"]["action"][0] = 999
    state["rng"]["state"]["state"] = 999
    state["spec"]["formula"]["noise_scales"][0] = 999
    assert policy.state_dict() == expected
    with pytest.raises(RuntimeError):
        policy.choose([10, 3000])
    assert policy.state_dict() == expected
    committed = policy.commit(row)
    expected = policy.state_dict()
    committed["base_after"][0] = 999
    committed["realized_offset_after"][0] = 999
    assert policy.state_dict() == expected
    with pytest.raises(RuntimeError):
        policy.commit(row)


@pytest.mark.parametrize("behavior", ["carry", "local"])
@pytest.mark.parametrize("pending", [False, True])
def test_json_resume_deterministic_continuation_and_pending_commit(behavior, pending):
    policy = LocalBudgetPolicy(behavior, 6903, 6000)
    anchor = np.array([-55, 2500.0])
    for step in range(6):
        action, audit = policy.choose(anchor)
        anchor = np.asarray(audit["projected_request"]) + [1, -3]
        policy.commit(row_for(action, audit, anchor,
                              reference="pfo_recovery" if step == 3 else "previous"))
    if pending:
        action, audit = policy.choose(anchor)
    state = roundtrip(policy.state_dict())
    restored = LocalBudgetPolicy(behavior, 6903, 6000)
    restored.load_state_dict(state)
    assert restored.state_dict() == policy.state_dict()
    state["base"][0] = 999
    state["rng"]["state"]["state"] = 999
    if pending:
        row = row_for(action, audit, anchor + [10, 20], selection="reference_fallback")
        assert restored.commit(row) == policy.commit(row)
        anchor = np.asarray(row["B_executed"])
    for _ in range(8):
        action, audit = policy.choose(anchor)
        restored_action, restored_audit = restored.choose(anchor)
        np.testing.assert_array_equal(restored_action, action)
        assert restored_audit == audit
        row = row_for(action, audit)
        assert restored.commit(row) == policy.commit(row)
        anchor = np.asarray(row["B_executed"])
    assert restored.state_dict() == policy.state_dict()


@pytest.mark.parametrize("behavior", ["carry", "local"])
@pytest.mark.parametrize("pending", [False, True])
def test_initial_checkpoint_restore(behavior, pending):
    policy = LocalBudgetPolicy(behavior, 6901, 6000)
    if pending:
        action, audit = policy.choose([-123, 3000])
    restored = LocalBudgetPolicy(behavior, 6901, 6000)
    restored.load_state_dict(roundtrip(policy.state_dict()))
    assert restored.state_dict() == policy.state_dict()
    if pending:
        assert restored.commit(row_for(action, audit)) == policy.commit(row_for(action, audit))


@pytest.mark.parametrize("kwargs", [
    {"behavior": "other"}, {"behavior": None}, {"seed": -1}, {"seed": 1.5},
    {"seed": True}, {"seed": "1"}, {"capacity": 0}, {"capacity": -1},
    {"capacity": float("inf")}, {"capacity": float("nan")}, {"capacity": True},
    {"capacity": "6000"}, {"capacity": [6000]},
])
def test_invalid_constructor(kwargs):
    spec = dict(behavior="local", seed=0, capacity=6000)
    spec.update(kwargs)
    with pytest.raises(ValueError):
        LocalBudgetPolicy(**spec)


@pytest.mark.parametrize("anchor", [None, [1], [1, 2, 3], [[1, 2]], [1, float("nan")],
                                    [float("inf"), 1], ["1", "2"], [True, False]])
def test_invalid_anchor_is_atomic(anchor):
    policy = LocalBudgetPolicy("local", 0, 6000)
    before = policy.state_dict()
    with pytest.raises(ValueError):
        policy.choose(anchor)
    assert policy.state_dict() == before


@pytest.mark.parametrize("key,value", [
    ("action_anchor", [100, 3001]), ("action_requested", [0, 0]),
    ("B_requested", [[100, 3000]]), ("B_requested", []), ("B_requested", [1, 2]),
    ("B_executed", [1, float("nan")]), ("B_executed", [1]),
    ("selection_source", "unknown"), ("reference_source", "unknown"),
    ("reference_source", "pfo_recovery"), ("reference_recovery", True),
    ("reference_recovery", "false"), ("reference_recovery", 0), ("guard_mode", "h3"),
    ("selection_source", np.array(["lower_solution"])),
    ("reference_source", np.array(["previous"])), ("guard_mode", np.array(["physical"])),
])
def test_bad_commit_preserves_pending_rng_and_state(key, value):
    policy = LocalBudgetPolicy("local", 6901, 6000)
    action, audit = policy.choose([100, 3000])
    row = row_for(action, audit)
    before = policy.state_dict()
    bad = copy.deepcopy(row)
    bad[key] = value
    with pytest.raises(ValueError):
        policy.commit(bad)
    assert policy.state_dict() == before
    assert policy.commit(row)["committed"] is True


@pytest.mark.parametrize("field", ["action_anchor", "action_requested", "B_requested", "B_executed",
                                   "selection_source", "reference_source", "reference_recovery", "guard_mode"])
def test_missing_commit_field_is_rejected(field):
    policy = LocalBudgetPolicy("carry", 1, 6000)
    action, audit = policy.choose([100, 3000])
    row = row_for(action, audit)
    del row[field]
    before = policy.state_dict()
    with pytest.raises(ValueError):
        policy.commit(row)
    assert policy.state_dict() == before


@pytest.mark.parametrize("path,value", [
    (("format",), "other"), (("spec", "behavior"), "carry"), (("spec", "seed"), 6902),
    (("spec", "capacity"), 5000), (("spec", "formula", "mean_reversion"), 0.9),
    (("spec", "formula", "action_scales"), [50, 100]),
    (("count",), -1), (("count",), True), (("count",), 1.5), (("count",), 4),
    (("base",), None), (("base",), [1]), (("base",), [float("nan"), 0]),
    (("realized_offset",), [0, 0]), (("realized_offset",), [0, float("inf")]),
    (("last_executed",), None), (("last_executed",), [99, 99]),
    (("rng", "bit_generator"), "MT19937"), (("rng", "state", "state"), 42),
    (("rng", "state", "inc"), 0), (("rng", "has_uint32"), 1),
    (("pending", "step"), 99), (("pending", "action"), [0, 0]),
    (("pending", "raw_action"), [0, 0]), (("pending", "projected_request"), [1, 2]),
    (("pending", "desired_offset"), [0, 0]), (("pending", "base_before"), [0, 0]),
    (("pending", "realized_offset_before"), [0, 0]), (("pending", "noise"), [0, 0]),
    (("pending", "rate_limited"), [True, True]), (("pending", "committed"), True),
    (("pending", "action_anchor"), [1]), (("pending",), []), (("pending",), None),
])
def test_corrupt_checkpoint_rejected_without_mutation(path, value):
    policy = LocalBudgetPolicy("local", 6901, 6000)
    action, audit = policy.choose([-100, 3000])
    policy.commit(row_for(action, audit, [-95, 3100]))
    policy.choose([-95, 3100])
    before = policy.state_dict()
    bad = copy.deepcopy(before)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        policy.load_state_dict(bad)
    assert policy.state_dict() == before


@pytest.mark.parametrize("change", ["missing", "extra", "base", "offset", "execution", "rng"])
def test_invalid_initial_state_and_no_partial_load(change):
    policy = LocalBudgetPolicy("local", 0, 6000)
    before = policy.state_dict()
    bad = copy.deepcopy(before)
    if change == "missing":
        del bad["rng"]
    elif change == "extra":
        bad["extra"] = 1
    elif change == "base":
        bad["base"] = [0, 0]
    elif change == "offset":
        bad["realized_offset"] = [0, 1]
    elif change == "execution":
        bad["last_executed"] = [0, 0]
    else:
        bad["rng"]["state"]["state"] += 1
    with pytest.raises(ValueError):
        policy.load_state_dict(bad)
    assert policy.state_dict() == before


def test_nonzero_realized_initial_pending_and_false_first_base_rejected():
    policy = LocalBudgetPolicy("local", 0, 6000)
    policy.choose([-10, 3000])
    before = policy.state_dict()
    for key in ("base", "realized_offset"):
        bad = copy.deepcopy(before)
        bad[key][0] += 1
        with pytest.raises(ValueError):
            policy.load_state_dict(bad)
        assert policy.state_dict() == before


def test_overflowing_choice_and_commit_are_atomic():
    policy = LocalBudgetPolicy("local", 0, 6000)
    action, audit = policy.choose([1e308, 3000])
    before = policy.state_dict()
    with pytest.raises(ValueError):
        policy.commit(row_for(action, audit, [-1e308, 3000]))
    assert policy.state_dict() == before
    policy.commit(row_for(action, audit))
    before = policy.state_dict()
    with pytest.raises(ValueError):
        policy.choose([-1e308, 3000])
    assert policy.state_dict() == before
