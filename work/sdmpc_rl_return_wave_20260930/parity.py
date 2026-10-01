"""Isolated read-only parity with the original Actor on authenticated saved observations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def observations(settings, np, torch):
    arrays = []
    for episode in settings["data"]["episodes"]:
        payload = torch.load(REPO / episode["path"] / "experience.pt", map_location="cpu", weights_only=False)
        for obs, _, _, nxt, _ in payload["transitions"]:
            arrays.extend((obs, nxt))
    return np.stack(arrays)


def original():
    sys.path.insert(0, str(REPO / "work/sdmpc_rl_return_init_20260930"))
    from runtime import OUTPUT, read, sha, verify, np, torch
    from learner import Actor
    settings = read(OUTPUT / "settings.json")
    verify(settings["sources"])
    verify(settings["data"]["files"])
    if sha(OUTPUT / "model_final.pt") != "7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904":
        raise ValueError("Original actor checkpoint changed")
    state = torch.load(OUTPUT / "model_final.pt", map_location="cpu", weights_only=False)
    actor = Actor()
    actor.load_state_dict(state["learner"]["models"]["actor"], strict=True)
    actor.eval().requires_grad_(False)
    obs = observations(settings, np, torch)
    with torch.no_grad():
        actions = np.stack([actor(torch.from_numpy(row)).numpy() for row in obs])
    print(json.dumps(dict(observations_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),
                         actions=actions.tolist(), original_runtime=sys.modules["runtime"].__file__)))


def physical_import():
    import wave_support as w
    from actor import authenticate
    from checks import helpers, validate_controls
    from budget_env import BudgetEnv
    _, auth = authenticate()
    rt = w.boot(w.DEFAULT_SNAPSHOT, w.MASKS[0])
    w.import_boundary(physical=True)
    for i, scenario in enumerate(w.SCENARIOS):
        env = BudgetEnv(rt, scenario=scenario, training_seed=7301+i, guard_mode="physical")
        helpers().verify_environment(env, rt, auth["contract"])
        if env.profile_hash != auth["profiles"][scenario]:
            raise ValueError("Original BudgetEnv profile differs")
    old_settings = w.read(w.PREDECESSOR / "settings.json")
    control_rows = 0
    for episode in old_settings["data"]["episodes"]:
        folder = REPO / episode["path"]
        settings = w.read(folder / "settings.json")
        settings["physical_options"] = w.plain(rt["rc"].to_plain_dict(rt["options"]))
        trace = w.read(folder / "trace.json")
        validate_controls(trace, settings)
        control_rows += len(trace)
    print(json.dumps(dict(physical_runtime=sys.modules["runtime"].__file__, profiles=auth["profiles"],
        forbidden_modules_present=sorted({"learner", "data", "td3", "train_round"}.intersection(sys.modules)),
        frozen_control_rows_validated=control_rows, resets=0, physical_steps=0, solves=0)))


def child(flag):
    result = subprocess.run([sys.executable, "-B", str(HERE / "parity.py"), flag],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise ValueError("Read-only parity child failed: " + result.stderr)
    return json.loads(result.stdout)


def verify():
    import wave_support as w
    from actor import authenticate, verify_live
    from checks import helpers
    actor, auth = authenticate()
    source = w.sources()
    helpers()
    expected = child("--original")
    obs = observations(w.read(w.PREDECESSOR / "settings.json"), w.np, w.torch)
    actions = w.np.stack([actor.act(row) for row in obs])
    w.np.testing.assert_array_equal(actions.view(w.np.uint32),
        w.np.array(expected["actions"], dtype=w.np.float32).view(w.np.uint32))
    if hashlib.sha256(obs.tobytes()).hexdigest() != expected["observations_sha256"]:
        raise ValueError("Parity observations differ")
    boundary = child("--physical-import")
    verify_live(auth, source)
    return dict(bit_identical=True, observations=len(obs), observation_shape=list(obs.shape),
        observations_sha256=expected["observations_sha256"], actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        model_sha256=w.MODEL_SHA, predecessor_completion_sha256=w.COMPLETION_SHA,
        original_runtime=expected["original_runtime"], physical_import=boundary,
        source_sha256=source, training_updates=0, physical_steps=0)


if __name__ == "__main__":
    if sys.argv[1:] == ["--original"]:
        original()
    elif sys.argv[1:] == ["--physical-import"]:
        physical_import()
    else:
        print(json.dumps(verify(), indent=2))
