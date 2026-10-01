import json
import pytest
import build_preflight as gate


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    for name in ("parity_step30_repaired.json", "isolation_step30.json", "smoke_resume_config_contract.json"):
        (tmp_path / name).write_text(json.dumps({"passed": True}), encoding="utf-8")
    for step in (5, 10, 30, 50):
        folder = tmp_path / f"probe_step{step}"
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps({
            "status": "completed", "action_influence_observed": True}), encoding="utf-8")
    (tmp_path / "preflight_tests.xml").write_text(
        '<testsuites><testsuite tests="20" errors="0" failures="0" skipped="0"/></testsuites>',
        encoding="utf-8")
    review = tmp_path / "review.md"
    review.write_text("PILOT_REVIEW: PASS", encoding="utf-8")
    monkeypatch.setattr(gate, "pins", lambda root: {"frozen": "test"})
    return tmp_path, review, tmp_path / "preflight.json"


def test_gate_freezes_evidence_without_claiming_performance(evidence):
    root, review, output = evidence
    gate.build(root, [review], output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["ready_for_bounded_pilot"]
    assert not result["performance_acceptance"]
    assert not result["legacy_ddqn_resumed"]
    assert len(result["evidence_sha256"]) == 9
    with pytest.raises(FileExistsError):
        gate.build(root, [review], output)


@pytest.mark.parametrize("mutation", ["diagnostic", "influence", "review", "tests", "empty_tests", "no_review"])
def test_gate_rejects_missing_acceptance(evidence, mutation):
    root, review, output = evidence
    reviews = [review]
    if mutation == "diagnostic":
        (root / "smoke_resume_config_contract.json").write_text('{"passed": false}', encoding="utf-8")
    elif mutation == "influence":
        (root / "probe_step30/summary.json").write_text(
            '{"status": "completed", "action_influence_observed": false}', encoding="utf-8")
    elif mutation == "review":
        review.write_text("HOLD", encoding="utf-8")
    elif mutation == "tests":
        (root / "preflight_tests.xml").write_text('<testsuite tests="20" failures="1"/>', encoding="utf-8")
    elif mutation == "empty_tests":
        (root / "preflight_tests.xml").write_text('<testsuite tests="0"/>', encoding="utf-8")
    else:
        reviews = []
    with pytest.raises(ValueError):
        gate.build(root, reviews, output)
    assert not output.exists()
