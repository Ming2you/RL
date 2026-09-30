from copy import deepcopy
import pytest
from evaluate import FORMAT, validate_checkpoint


def fixture():
    settings = dict(model_sha256="actor-a", scenario="s", profile_sha256=["p"])
    checkpoint = dict(format=FORMAT, settings=deepcopy(settings), environment=dict(k=7, profile_hash="p"),
                      trace=[{}, {}], observations=[[], []])
    return checkpoint, settings


def test_resume_identity():
    ck, settings = fixture()
    validate_checkpoint(ck, settings)
    settings["model_sha256"] = "actor-b"
    with pytest.raises(ValueError):
        validate_checkpoint(ck, settings)


@pytest.mark.parametrize("resume", [False, True])
def test_preexisting_stop_prevents_boot_and_model_load(tmp_path, monkeypatch, resume):
    import evaluate
    out = tmp_path / "scenario"
    (tmp_path / "STOP").touch()
    if resume:
        out.mkdir()
        (out / "checkpoint.pt").write_bytes(b"preserve-checkpoint")
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--actor", str(tmp_path / "actor.pt"),
        "--scenario", evaluate.SCENARIOS[0], "--output", str(out)] + (["--resume"] if resume else []))
    monkeypatch.setattr(evaluate, "boot", lambda *args: pytest.fail("environment boot after STOP"))
    monkeypatch.setattr(evaluate, "load_base", lambda: pytest.fail("model load after STOP"))
    evaluate.main()
    assert evaluate.read(out / "status.json")["before_environment_initialization"] is True
    if resume:
        assert (out / "checkpoint.pt").read_bytes() == b"preserve-checkpoint"


@pytest.mark.parametrize("field,value", [("k", 81), ("k", True), ("k", 6), ("profile_hash", "other")])
def test_resume_boundaries(field, value):
    ck, settings = fixture()
    ck["environment"][field] = value
    with pytest.raises(ValueError):
        validate_checkpoint(ck, settings)


def test_observation_count():
    ck, settings = fixture()
    ck["observations"].pop()
    with pytest.raises(ValueError):
        validate_checkpoint(ck, settings)
