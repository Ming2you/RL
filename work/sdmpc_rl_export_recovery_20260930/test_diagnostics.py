"""Only output-boundary tests; no runtime boot or numerical collection."""
import copy
import json
import math

import pytest

from export_recovery import tag_diagnostics


def test_known_failure_and_exact_tags():
    trace = [dict(candidates=[dict(stationarity=float("inf"), converged=False,
        rows=[dict(primal_stationarity=float("inf"))])],
        selection_source="reference_fallback", reward=-1.,
        execution_check=dict(physical_control_valid=True, budget_feasible=True))]
    before = copy.deepcopy(trace)
    with pytest.raises(ValueError, match="Out of range"):
        json.dumps(trace, allow_nan=False)
    exported, changes = tag_diagnostics(trace)
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported
    assert [c["path"] for c in changes] == [
        [0, "candidates", 0, "stationarity"],
        [0, "candidates", 0, "rows", 0, "primal_stationarity"]]
    assert all(c["original_value"] == "+inf" and c["original_type"] == "float" for c in changes)
    assert trace == before and math.isinf(trace[0]["candidates"][0]["stationarity"])
    assert exported[0]["candidates"][0]["converged"] is False
    assert exported[0]["selection_source"] == "reference_fallback"


def test_finite_identity():
    trace = [dict(candidates=[dict(stationarity=-0.0, rows=[dict(primal_stationarity=2.)])],
                  value=[None, True, 1, 1.25, "inf"], reward=-2.)]
    exported, changes = tag_diagnostics(trace)
    assert json.dumps(exported) == json.dumps(trace)
    assert changes == []


@pytest.mark.parametrize("value", [float("nan"), -float("inf")])
def test_negative_infinity_and_nan_forbidden_even_in_diagnostics(value):
    with pytest.raises(ValueError, match="nonfinite"):
        tag_diagnostics([dict(candidates=[dict(stationarity=value)])])


@pytest.mark.parametrize("row", [
    dict(reward=float("inf")), dict(plant_state=dict(density=float("inf"))),
    dict(action_requested=[float("inf"), 0.]), dict(total_ttt=float("inf")),
    dict(stationarity=float("inf")), dict(candidates=dict(stationarity=float("inf"))),
    dict(candidates={0: dict(stationarity=float("inf"))}),
    dict(candidates=[dict(rows=dict(primal_stationarity=float("inf")))]),
    dict(candidates=[dict(nested=dict(stationarity=float("inf")))]),
    dict(candidates=[dict(stationarity={"nonfinite_float": "+inf"})]),
])
def test_no_structural_or_physical_allowlist_widening(row):
    with pytest.raises((ValueError, TypeError)):
        tag_diagnostics([row])
