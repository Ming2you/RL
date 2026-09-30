"""Authenticate five fixed-policy episodes and load only their 375 MC labels."""
import json
import subprocess
import sys
from mc_common import (HERE, REPO, PARENT, WAVE, READOUT, READOUT_SHA, MODEL_SHA,
    COMPLETION_SHA, SCENARIOS, SPEC, np, torch, read, sha, finite, verify,
    check_hash, episode_arrays, projected, tensor_hash)


def manifest():
    check_hash(READOUT, READOUT_SHA)
    check_hash(PARENT / "completion.json", COMPLETION_SHA)
    check_hash(PARENT / "model_final.pt", MODEL_SHA)
    parent = read(PARENT / "completion.json")
    files = dict(parent["settings"]["sources"])
    files.update(parent["settings"]["data"]["files"])

    def bind(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(REPO.resolve()):
            raise ValueError("Artifact path escaped repository")
        value = sha(path) if expected is None else expected
        key = path.relative_to(REPO).as_posix()
        if key in files and files[key] != value:
            raise ValueError("Conflicting input hashes")
        files[key] = value

    bind(READOUT, READOUT_SHA)
    bind(PARENT / "completion.json", COMPLETION_SHA)
    for name, expected in parent["outputs_sha256"].items():
        bind(PARENT / name, expected)
    bind(PARENT / "settings.json")
    episodes = []
    for i, scenario in enumerate(SCENARIOS):
        folder = WAVE / scenario
        done, settings = read(folder / "completion.json"), read(folder / "settings.json")
        if (done["status"] != "completed" or done["settings"] != settings or
                settings["model_sha256"] != MODEL_SHA or settings["training_seed"] != 7301+i or
                settings["scenario"] != scenario or settings["evaluation"] is not False or
                settings["exploration"] is not False or settings["behavior"] != "frozen_shared_actor"):
            raise ValueError("On-policy episode provenance mismatch")
        for mapping in (settings["sources"], settings["authentication"]["helper_sources"]):
            for name, expected in mapping.items():
                bind(REPO / name, expected)
        for name in ("settings.json", "completion.json"):
            bind(folder / name)
        for name, expected in done["outputs_sha256"].items():
            bind(folder / name, expected)
        bind(folder / done["checkpoint"]["path"], done["checkpoint"]["sha256"])
        for name, expected in read(folder / "timing.json")["session_sha256"].items():
            bind(folder / name, expected)
        for path in (folder / "operations").glob("*.json"):
            bind(path)
        episodes.append(dict(path=folder.relative_to(REPO).as_posix(), scenario=scenario,
            training_seed=7301+i, run_id=settings["run_id"], profile_sha256=settings["profile_sha256"]))
    snapshot = REPO / "artifacts/sdmpc_budget_baseline_20260929"
    bind(snapshot / "manifest.json")
    for row in read(snapshot / "manifest.json")["files"]:
        bind(snapshot / row["snapshot_relative"], row["sha256"])
    verify(files)
    return dict(files=files, episodes=episodes, rows=375, model_sha256=MODEL_SHA,
        readout_sha256=READOUT_SHA, parent_completion_sha256=COMPLETION_SHA,
        evaluation=False, continuation="pi1_recorded_on_policy_MC")


def validate_receipt(receipt, identity):
    if (receipt["status"] != "authenticated" or receipt["manifest"] != identity or
            receipt["readout"] != read(READOUT) or not receipt["readout"]["integrity_pass"] or
            not receipt["readout"]["all_health_gates_pass"] or
            receipt["raw_environment_loads"] != 0 or receipt["environment_calls"] != 0):
        raise ValueError("Isolated source validation/readout mismatch")


def authenticate():
    before = manifest()
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "authenticate.py")],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    receipt = json.loads(result.stdout)
    validate_receipt(receipt, before)
    if manifest() != before:
        raise ValueError("Inputs changed during isolated authentication")
    return receipt


def load_data(identity):
    verify(identity["files"])
    if identity != manifest():
        raise ValueError("Data identity differs")
    rows = []
    for entry in identity["episodes"]:
        folder = REPO / entry["path"]
        settings, trace = read(folder / "settings.json"), read(folder / "trace.json")
        payload = torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)
        if payload["settings"] != settings or payload["format"] != settings["format"]:
            raise ValueError("Experience/settings mismatch")
        own = episode_arrays(payload["transitions"], trace, settings, entry["scenario"], "frozen_shared_actor")
        for row, source in zip(own, trace):
            # The request is retained as float64 from the trace, never decoded from obs.
            row["request"] = np.asarray(source["B_requested"][0], dtype=np.float64)
            np.testing.assert_array_equal(projected(torch.from_numpy(row["anchor"]),
                torch.from_numpy(row["action"])).numpy(), row["request"])
        rows.extend(own)
    data = {key: torch.from_numpy(np.asarray([row[key] for row in rows])) for key in
            ("obs", "action", "reward", "terminal", "anchor", "request", "returns", "scenario", "horizon")}
    finite(data)
    if (len(rows) != SPEC["rows"] or data["anchor"].dtype != torch.float64 or
            data["request"].dtype != torch.float64 or not torch.all(data["request"][:, 1] == 6000.)):
        raise ValueError("On-policy data size/precision/cap coverage differs")
    verify(identity["files"])
    return data


def load_parent(identity):
    verify(identity["files"])
    check_hash(PARENT / "model_final.pt", MODEL_SHA)
    payload = torch.load(PARENT / "model_final.pt", map_location="cpu", weights_only=False)
    finite(payload)
    if payload["settings"] != read(PARENT / "settings.json"):
        raise ValueError("Parent settings mismatch")
    check_hash(PARENT / "model_final.pt", MODEL_SHA)
    return payload["learner"]


def sample_indices(data, rng):
    indices, forced = [], []
    for scenario in range(5):
        pool = torch.where(data["scenario"] == scenario)[0].numpy()
        terminal = pool[data["terminal"][pool].numpy()]
        if len(pool) != 75 or len(terminal) != 1:
            raise ValueError("Need 75 own rows and one true terminal per scenario")
        indices.extend([int(terminal[0]), *rng.choice(pool, 7, replace=True).tolist()])
        forced.extend([True] + [False]*7)
    order = rng.permutation(40)
    return np.asarray(indices, dtype=np.int64)[order], np.asarray(forced, dtype=bool)[order]
