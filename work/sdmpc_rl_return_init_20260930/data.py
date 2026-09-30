"""Exactly five paired training trajectories; anchors come only from float64 traces."""
import subprocess
import sys
from runtime import (HERE, REPO, GOAL, SCENARIOS, SPEC, np, torch, read, sha,
                     finite, digest, verify)

ROOT = GOAL / "nuf_retention_v1"
CARRY_ROOT = GOAL / "local_budget_v2"
ROOT_PINS = {"completion.json": "2d5c7a881fe5cb818686b13dd83f2c98eb08059e4b698cbe3a1be8757fc0c117",
             "comparison.json": "60f7b670e262348d3d24548809b2aafc26bc7c2a7e7c8f17340fbd17c2803801"}


def manifest():
    for name, expected in ROOT_PINS.items():
        if sha(ROOT / name) != expected:
            raise ValueError("NUF root hash mismatch: " + name)
    done = read(ROOT / "completion.json")
    plan = done["plan"]
    if (done["status"] != "completed" or plan != read(ROOT / "plan.json") or
            done["comparison_sha256"] != ROOT_PINS["comparison.json"] or
            done["all_integrity_and_screens_pass"] is not True):
        raise ValueError("Incomplete NUF provenance")
    files = {str((ROOT / name).relative_to(REPO).as_posix()): sha(ROOT / name)
             for name in ("completion.json", "comparison.json", "plan.json")}
    files.update(plan["identity"]["local_sources"])
    refs = plan["paired_references"]
    for name, expected in refs["root_sha256"].items():
        files[(CARRY_ROOT / name).relative_to(REPO).as_posix()] = expected
    files.update(refs["source_identity"]["local_sources"])
    episodes = []
    for scenario in SCENARIOS:
        for behavior, base in (("carry", CARRY_ROOT), ("local", ROOT)):
            folder = base / behavior / scenario
            key = f"{behavior}/{scenario}"
            expected = (refs["entries"][key]["completion_sha256"] if behavior == "carry"
                        else done["child_completion_sha256"][key])
            if sha(folder / "completion.json") != expected:
                raise ValueError("Episode completion hash mismatch")
            child, settings = read(folder / "completion.json"), read(folder / "settings.json")
            if child["status"] != "completed" or child["settings"] != settings:
                raise ValueError("Episode completion/settings mismatch")
            if (settings["scenario"] != scenario or settings["behavior"] != behavior or
                    settings["evaluation"] is not False or settings["training_seed"] != 6801+SCENARIOS.index(scenario)
                    or settings["capacity"] != 6000.):
                raise ValueError("Training-only episode identity mismatch")
            canonical = settings["contract"]["environment_contract"]["scenarios"]
            if settings["profile_sha256"] in {row["profile_sha256"] for row in canonical}:
                raise ValueError("Canonical evaluation leakage")
            if (settings["contract"]["gamma"] != 1. or settings["contract"]["reward_divisor"] != 100.
                    or len(settings["contract"]["observation_schema"]["names"]) != SPEC["observations"]):
                raise ValueError("Observation/reward contract differs")
            for name in ("completion.json", "settings.json"):
                files[(folder / name).relative_to(REPO).as_posix()] = sha(folder / name)
            for name, expected in child["outputs_sha256"].items():
                path = (folder / name).resolve()
                if path.parent != folder.resolve():
                    raise ValueError("Output manifest path escape")
                files[path.relative_to(REPO).as_posix()] = expected
            episodes.append(dict(path=folder.relative_to(REPO).as_posix(), scenario=scenario,
                behavior=behavior, run_id=settings["run_id"], profile_sha256=settings["profile_sha256"],
                training_seed=settings["training_seed"]))
    verify(files)
    return dict(files=files, episodes=episodes, root_pins=ROOT_PINS,
                frozen_identity=plan["identity"], references=refs,
                rows=750, carry_rows=375, local_rows=375, evaluation=False)


def authenticate():
    import json
    result = subprocess.run([sys.executable, "-B", str(HERE / "authenticate.py")], cwd=REPO,
        capture_output=True, text=True, encoding="utf-8", check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    checked = json.loads(result.stdout)
    if checked["status"] != "authenticated" or checked["manifest"] != manifest():
        raise ValueError("Isolated authentication receipt differs")
    return checked


def projected(anchor, action):
    """Float32 policy/replay action, then float64 physics; NP remains signed."""
    if anchor.dtype != torch.float64:
        raise ValueError("Exact float64 trace anchors required")
    raw = anchor + action.float().double() * anchor.new_tensor([50., 1000.])
    return torch.stack((raw[..., 0], raw[..., 1].clamp(0., 6000.)), dim=-1)


def network_budget(anchor, action):
    return (projected(anchor, action) / anchor.new_tensor([1000., 10000.])).float()


def episode_arrays(transitions, trace, settings, scenario, behavior):
    if len(transitions) != 75 or len(trace) != 75 or settings["evaluation"] is not False:
        raise ValueError("Expected complete training-only trajectory")
    rows, total = [], 0.
    returns = np.zeros(75, dtype=np.float64)
    for k in reversed(range(75)):
        obs, action, reward, nxt, terminal = transitions[k]
        row = trace[k]
        if (type(terminal) is not bool or terminal != (k == 74) or row["terminated"] is not terminal
                or row["truncated"] is not False or row["control_step"] != k
                or row["step"] != k+5 or row["scenario"] != scenario or row["behavior"] != behavior
                or row["profile_sha256"] != settings["profile_sha256"]):
            raise ValueError("Sequential episode identity/terminal mismatch")
        finite((obs, action, reward, nxt, row["action_anchor"], row["B_executed"]))
        if (np.asarray(obs).dtype != np.float32 or np.asarray(nxt).dtype != np.float32 or
                np.shape(obs) != (2367,) or np.shape(nxt) != (2367,) or np.shape(action) != (2,)):
            raise ValueError("Observation/action schema mismatch")
        if reward > 0 or abs(reward + row["interval_ttt"]/100.) > 1e-8:
            raise ValueError("Reward accounting mismatch")
        if terminal:
            np.testing.assert_array_equal(nxt, np.zeros(2367, dtype=np.float32))
            total = 0.
        else:
            np.testing.assert_array_equal(nxt, transitions[k+1][0])
            np.testing.assert_array_equal(row["B_executed"], trace[k+1]["action_anchor"])
        if behavior == "carry":
            np.testing.assert_array_equal(action, np.zeros(2, dtype=np.float32))
        anchor = np.asarray(row["action_anchor"], dtype=np.float64)
        if anchor.shape != (2,):
            raise ValueError("Anchor shape differs")
        request = projected(torch.from_numpy(anchor), torch.as_tensor(action)).numpy()
        np.testing.assert_array_equal(np.asarray(action, dtype=np.float32), np.asarray(row["action_requested"], dtype=np.float32))
        np.testing.assert_allclose(request, row["B_requested"][0], rtol=0, atol=1e-10)
        total += float(reward)
        returns[k] = total
    for k, ((obs, action, reward, nxt, terminal), row) in enumerate(zip(transitions, trace)):
        rows.append(dict(obs=obs, action=np.asarray(action, dtype=np.float32), reward=reward,
            next_obs=nxt, terminal=terminal, anchor=np.asarray(row["action_anchor"], dtype=np.float64),
            next_anchor=np.zeros(2, dtype=np.float64) if terminal else np.asarray(trace[k+1]["action_anchor"], dtype=np.float64),
            returns=returns[k], scenario=SCENARIOS.index(scenario), local=behavior == "local", horizon=75-k))
    return rows


def load_data(identity):
    if identity != manifest():
        raise ValueError("Data changed since authentication")
    rows = []
    for entry in identity["episodes"]:
        folder = REPO / entry["path"]
        settings = read(folder / "settings.json")
        payload = torch.load(folder / "experience.pt", map_location="cpu", weights_only=False)
        if payload["settings"] != settings:
            raise ValueError("Experience provenance differs")
        rows.extend(episode_arrays(payload["transitions"], read(folder / "trace.json"), settings,
                                  entry["scenario"], entry["behavior"]))
    verify(identity["files"])
    data = {key: torch.from_numpy(np.asarray([row[key] for row in rows])) for key in rows[0]}
    finite(data)
    if len(data["obs"]) != 750 or data["anchor"].dtype != torch.float64:
        raise ValueError("Data size/precision mismatch")
    return data


def sample(data, rng, phi=False):
    indices = []
    for scenario in range(5):
        for local in ([False] if phi else [False, True]):
            pool = torch.where((data["scenario"] == scenario) & (data["local"] == local))[0].numpy()
            terminals = pool[data["terminal"][pool].numpy()]
            if len(terminals) != 1 or len(pool) == 0:
                raise ValueError("Every scenario/behavior requires exactly one true terminal")
            indices.extend([int(terminals[0]), *rng.choice(pool, 7 if phi else 3, replace=True).tolist()])
    indices = np.asarray(indices, dtype=np.int64)
    rng.shuffle(indices)
    return {key: value[indices] for key, value in data.items()}
