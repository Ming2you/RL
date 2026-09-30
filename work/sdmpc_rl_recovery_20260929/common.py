"""Read-only access to the completed balanced pilot and its physical runtime."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
sys.path[:0] = [str(REPO / ".deps-budget"), str(REPO / "work/sdmpc_rl_multi_20260929")]
import torch
from td3 import TD3, SCENARIOS
from budget_runtime import DEFAULT_SNAPSHOT, read, save, plain
from run_budget import file_hash, pins, verify_pins, runtime_versions, exclusive_run, checkpoint_save

BASE = REPO / "results/sdmpc_rl_multi_20260929/pilot_v1"
BASE_HASH = "3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650"


def load_base():
    path = BASE / "train_round1/model_final.pt"
    if file_hash(path) != BASE_HASH or read(BASE / "completion.json")["status"] != "completed":
        raise ValueError("Frozen base pilot identity differs")
    model = torch.load(path, map_location="cpu", weights_only=False)
    verify_pins(DEFAULT_SNAPSHOT, model["contract"]["source_pins"])
    if runtime_versions() != model["contract"]["runtime_versions"]:
        raise ValueError("Runtime differs from the baseline")
    state = model["learner"]
    learner = TD3(state["observation_dim"], state["spec"]["seed"], state["spec"]["hidden"])
    learner.load_state_dict(state)
    if learner.replay_counts() != dict.fromkeys(SCENARIOS, 150):
        raise ValueError("Expected balanced completed training replay")
    for data in state["replay"].values():
        if data["terminated"].nonzero().ravel().tolist() != [74, 149]:
            raise ValueError("Episode boundaries differ")
    return model, learner


def sources(*names):
    return {name: file_hash(HERE / name) for name in names}


def stopped(folder):
    return any((p / "STOP").exists() for p in (folder, folder.parent, folder.parent.parent))
