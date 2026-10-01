from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_sdmpc_recovery_20260929 as analysis


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True])
def test_inventory_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        analysis.validate_inventory(dict(terminal_inventory=value), [dict(inventory=value)])


def test_inventory_matches_final_trace_and_checks_intermediates():
    analysis.validate_inventory(dict(terminal_inventory=4.), [dict(inventory=2.), dict(inventory=4.)])
    with pytest.raises(ValueError):
        analysis.validate_inventory(dict(terminal_inventory=3.), [dict(inventory=4.)])
    with pytest.raises(ValueError):
        analysis.validate_inventory(dict(terminal_inventory=4.), [dict(inventory=-2.), dict(inventory=4.)])


@pytest.mark.parametrize("mode", ["rl", "center"])
def test_reference_uses_authenticated_completed_run_and_expected_policy(monkeypatch, mode):
    settings = dict(source_pins={"p": 1}, runtime_versions={"r": 1},
                    environment_contract={"c": 1}, profile_sha256=["canonical"])
    summary = dict(ttt=12., warmup_ttt=10., terminal_inventory=4.)
    seen = {}

    def completed(folder, received_mode, seeds, source, runtime, **kwargs):
        seen.update(kwargs)
        assert received_mode == mode and seeds == [None]
        assert source == settings["source_pins"] and runtime == settings["runtime_versions"]
        return {"episodes": [summary]}, deepcopy(settings), [[dict(inventory=4.)]], {"names": ["a"]}

    monkeypatch.setattr(analysis, "load_completed_run", completed)
    assert analysis.reference_run("s", mode, settings, {"names": ["a"]}, 10.) == summary
    assert seen == dict(model_sha256=analysis.BASE_HASH if mode == "rl" else None, scenario="s")
    with pytest.raises(ValueError):
        analysis.reference_run("s", mode, settings, {"names": ["different"]}, 10.)
    with pytest.raises(ValueError):
        analysis.reference_run("s", mode, settings, {"names": ["a"]}, 11.)
    for invalid in (0., -1., float("nan"), float("inf")):
        summary["ttt"] = invalid
        with pytest.raises(ValueError):
            analysis.reference_run("s", mode, settings, {"names": ["a"]}, 10.)


def test_failed_reference_validation_is_not_bypassed(monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("Wrong prior model hash")
    monkeypatch.setattr(analysis, "load_completed_run", invalid)
    with pytest.raises(ValueError, match="Wrong prior model"):
        analysis.reference_run("s", "rl", dict(source_pins={}, runtime_versions={}), {}, 10.)
