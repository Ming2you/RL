"""Offline helpers and the fixed Task 3 contract; no simulator imports."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOAL = REPO / "results/sdmpc_rl_balanced_goal_20260930"
OUTPUT = GOAL / "return_mc_v1"
PARENT = GOAL / "return_init_v1"
WAVE = GOAL / "return_policy_wave_v1"
OLD = REPO / "work/sdmpc_rl_return_init_20260930"
WAVE_SOURCE = REPO / "work/sdmpc_rl_return_wave_20260930"
READOUT = REPO / ".superpowers/sdd/rl_budget_return_initialized_policy_20260930/task2-readout.json"
MODEL_SHA = "7723bad496cd2e2b5f9a8f52b74ac8f65f021ccff42635d396e9f3b248b23904"
COMPLETION_SHA = "34f617fc171ce0a63611f5883eac7e8af9814a59889f4a545adb3fabb3778757"
READOUT_SHA = "e8d2795c3828069d88f83797d1b644d3cba9f87f58d428b84ee387601777f1db"


def check_hash(path, expected):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Frozen source/model/data hash mismatch: " + str(path))


# Verify reused code before importing it, using the pinned parent's completion.
check_hash(PARENT / "completion.json", COMPLETION_SHA)
PARENT_COMPLETION = json.loads((PARENT / "completion.json").read_text(encoding="utf-8-sig"))
for _name, _expected in PARENT_COMPLETION["settings"]["sources"].items():
    check_hash(REPO / _name, _expected)
_spec = importlib.util.spec_from_file_location("mc_parent_offline_runtime", OLD / "runtime.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)
np, torch = rt.np, rt.torch
read, sha, digest, plain = rt.read, rt.sha, rt.digest, rt.plain
finite, tensor_hash, SCENARIOS = rt.finite, rt.tensor_hash, rt.SCENARIOS
locked, check_stop, Stopped = rt.locked, rt.check_stop, rt.Stopped
restore, process_identity = rt.restore, rt.process_identity
ANCESTOR_COUNTS = dict(phi=1000, critic=250, actor=10, polyak=125)
SPEC = dict(format="sdmpc-on-policy-return-mc-v1", seed=7201, rng="PCG64",
    observations=2367, width=64, hidden_layers=2, activation="ReLU",
    action_bounds=[.2, .1], delta_scale=[50., 1000.], budget_scale=[1000., 10000.],
    capacity=6000., gamma=1., reward_divisor=100., controlled_steps=75, warmup_steps=5,
    rows=375, critic_updates=250, batch_size=40, samples_per_scenario=8,
    forced_terminals_per_scenario=1, uniform_per_scenario=7, replacement=True,
    uniform_pool="all 75 own scenario rows including terminal", lr=3e-4,
    optimizer="Adam continued from matching parent critics", loss="sum of both head MSEs",
    target="G_pi1 - frozen Phi(s)", critic_continuation="pi1_recorded_on_policy_MC",
    phi_updates=0, actor_updates=0, bootstrap=False, target_noise=False,
    target_updates_during_fit=0, final_target_copy=True, checkpoint_period=1,
    numerical_workers=1, threads=1, canonical_evaluation=False,
    heldout=False, traffic_improvement_claim=False)


def verify(files):
    for name, expected in files.items():
        path = (REPO / name).resolve()
        if not path.is_relative_to(REPO.resolve()):
            raise ValueError("Manifest path escaped repository")
        check_hash(path, expected)


def sources():
    paths = list(HERE.glob("*.py"))
    return dict(PARENT_COMPLETION["settings"]["sources"], **{
        p.relative_to(REPO).as_posix(): sha(p) for p in sorted(paths)})


def save(path, value):
    finite(value)
    path = Path(path)
    payload = json.dumps(rt.plain(value), indent=2, allow_nan=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def definitions(path, names, namespace):
    """Reuse only pinned pure definitions, without importing old learner/collector modules."""
    check_hash(path, PARENT_COMPLETION["settings"]["sources"][path.relative_to(REPO).as_posix()])
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names) or len(nodes) != len(names):
        raise ValueError("Missing/ambiguous pure helper")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


definitions(OLD / "learner.py", ("mlp", "Actor", "residuals"), globals())
definitions(OLD / "data.py", ("projected", "network_budget", "episode_arrays"), globals())
definitions(OLD / "runtime.py", ("save_once", "checkpoint"), globals())
