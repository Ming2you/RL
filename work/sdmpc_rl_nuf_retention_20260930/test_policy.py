"""Only the changed NUF bookkeeping and its unchanged action/RNG boundary."""
import copy
import importlib.util
import json
import numpy as np
import pytest
from exploration import LocalBudgetPolicy
import references


def commit(policy, anchor, executed=None, fallback=False, recovery=False):
    action, choice = policy.choose(anchor)
    row = dict(action_anchor=choice["action_anchor"], action_requested=choice["action"],
        B_requested=[choice["projected_request"]],
        B_executed=choice["projected_request"] if executed is None else executed,
        selection_source="reference_fallback" if fallback else "lower_solution",
        reference_source="pfo_recovery" if recovery else "previous", reference_recovery=recovery,
        guard_mode="physical")
    return action, policy.commit(row)


@pytest.mark.parametrize("fallback,recovery", [(True, False), (False, True), (True, True), (False, False)])
def test_retained_nuf_np_rule_and_realized_offset(fallback, recovery):
    policy = LocalBudgetPolicy("local", 6902, 6000.)
    _, audit = commit(policy, [-114., 6000.], [-90., 2840.], fallback, recovery)
    expected_np = -90. if fallback or recovery else -114.
    assert audit["base_after"] == [expected_np, 6000.]
    assert audit["realized_offset_after"] == [-90.-expected_np, -3160.]
    np.testing.assert_array_equal(np.array(audit["executed_budget"])-audit["base_after"], audit["realized_offset_after"])
    clone = LocalBudgetPolicy("local", 6902, 6000.)
    clone.load_state_dict(json.loads(json.dumps(policy.state_dict())))
    assert clone.choose([-90., 2840.])[1] == policy.choose([-90., 2840.])[1]


def test_low_executed_budget_recovery_uses_actual_anchor_and_rate():
    policy = LocalBudgetPolicy("local", 6902, 6000.)
    commit(policy, [-10., 6000.], [-10., 100.], fallback=True)
    _, choice = policy.choose([-10., 100.])
    assert choice["desired_budget"][1] == 5500.
    assert choice["action"][1] == 1. and choice["projected_request"][1] == 1100.
    assert choice["rate_limited"][1]


@pytest.mark.parametrize("anchor,offset,noise,desired", [(6000., 0., 10., 6000.), (0., 0., -10., 0.),
    (6000., -400., 0., 5680.), (0., 400., 0., 320.)])
def test_both_cap_boundaries_and_zero_noise_mean_reversion(anchor, offset, noise, desired):
    policy = LocalBudgetPolicy("local", 1, 6000.)
    policy._realized_offset = np.array([0., offset])
    choice = policy._choice(np.array([0., anchor+offset]), np.array([0., anchor]), np.array([0., noise]))
    assert choice["desired_budget"][1] == desired
    assert 0 <= choice["projected_request"][1] <= 6000.
    assert np.asarray(choice["action"], dtype=np.float32).dtype == np.float32


def test_pending_rng_roundtrip_and_old_format_rejected():
    policy = LocalBudgetPolicy("local", 6902, 6000.)
    policy.choose([-10., 6000.])
    saved = policy.state_dict()
    clone = LocalBudgetPolicy("local", 6902, 6000.)
    clone.load_state_dict(saved)
    assert clone.state_dict() == saved
    for field in ("rng", "format", "realized_offset"):
        broken = copy.deepcopy(saved)
        if field == "rng":
            broken[field]["state"]["state"] += 1
        elif field == "format":
            broken[field] = "local-budget-policy-v1"
        else:
            broken[field][1] += 1.
        with pytest.raises(ValueError):
            clone.load_state_dict(broken)


def test_saved_170_prefix_step10_arithmetic_and_all75_noise():
    done = references.read(references.ROOT / "completion.json")
    references.check_hash(references.ROOT / "completion.json", references.ROOT_PINS["completion.json"])
    folder = references.ROOT / "local/sweet_170_w"
    references.check_hash(folder / "completion.json", done["child_completion_sha256"]["local/sweet_170_w"])
    child = references.read(folder / "completion.json")
    references.check_hash(folder / "trace.json", child["outputs_sha256"]["trace.json"])
    old_trace = references.read(folder / "trace.json")
    policy = LocalBudgetPolicy("local", 6902, 6000.)
    for i in range(10):
        row = old_trace[i]
        action, choice = policy.choose(row["action_anchor"])
        np.testing.assert_array_equal(action, np.asarray(row["action_requested"], dtype=np.float32))
        np.testing.assert_array_equal(choice["noise"], row["exploration_audit"]["noise"])
        audit = policy.commit(row)
        assert audit["base_after"][0] == row["exploration_audit"]["base_after"][0]
    assert audit["base_after"][1] == 6000.
    assert audit["base_after"][1] != old_trace[9]["exploration_audit"]["base_after"][1]
    _, choice = policy.choose(old_trace[10]["action_anchor"])
    assert choice["projected_request"][1] == pytest.approx(5552.771855, abs=1e-6, rel=0)
    assert old_trace[10]["B_requested"][0][1] == pytest.approx(5504.421625, abs=1e-6, rel=0)
    rng = np.random.default_rng(6902)
    for row in old_trace:
        np.testing.assert_array_equal(rng.normal(size=2), row["exploration_audit"]["noise"])
    # Subsequent mock executions verify draw ordering only, never hypothetical plant rows.
    for i in range(10, 75):
        if i > 10:
            _, choice = policy.choose(policy.state_dict()["last_executed"])
        np.testing.assert_array_equal(choice["noise"], old_trace[i]["exploration_audit"]["noise"])
        policy.commit(dict(action_anchor=choice["action_anchor"], action_requested=choice["action"],
            B_requested=[choice["projected_request"]], B_executed=choice["projected_request"],
            selection_source="lower_solution", reference_source="previous", reference_recovery=False, guard_mode="physical"))
