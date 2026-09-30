"""Small offline-only runtime, reusing the existing serialization/identity helpers."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOAL = REPO / "results/sdmpc_rl_balanced_goal_20260930"
OUTPUT = GOAL / "return_init_v1"
NUF = REPO / "work/sdmpc_rl_nuf_retention_20260930"
sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.path.insert(0, str(REPO / ".deps-budget"))
import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serialization = helper("return_init_serialization", REPO / "work/sdmpc_rl_multi_20260929/budget_runtime.py")
process = helper("return_init_process", NUF / "launch_identity.py")
plain, digest = serialization.plain, serialization.digest
SCENARIOS = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")
SPEC = dict(format="sdmpc-return-init-v1", seed=7200, observations=2367, width=64,
    hidden_layers=2, activation="ReLU", action_bounds=[.2, .1], delta_scale=[50., 1000.],
    budget_scale=[1000., 10000.], capacity=6000., gamma=1., reward_divisor=100.,
    phi_updates=1000, critic_updates=250, actor_updates=10, batch_size=40, lr=3e-4,
    tau=.005, target_period=2, checkpoint_period=25, numerical_workers=1, threads=1,
    phi_sampling="8/scenario: 1 terminal + 7 uniform with replacement",
    mixed_sampling="4 carry + 4 local/scenario: each 1 terminal + 3 uniform with replacement",
    phi_pooled_mse_ratio_max=.5, phi_each_scenario_mse_strict_decrease=True,
    critic_continuation="carry", target_noise=False, target_actor=False,
    canonical_evaluation=False, traffic_improvement_claim=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite tensor")
    elif isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite array")
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError("Nonfinite scalar")


def read(path):
    value = serialization.read(path)
    finite(value)
    return value


def save(path, value):
    finite(value)
    json.dumps(plain(value), allow_nan=False)
    serialization.save(path, value)
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


def save_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != plain(value):
            raise FileExistsError(path)
    else:
        save(path, value)


def tensor_hash(value):
    h = hashlib.sha256()
    for name, item in sorted(value.items()):
        array = item.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str((array.dtype.str, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def sources():
    paths = list(HERE.glob("*.py")) + [Path(serialization.__file__), Path(process.__file__)]
    return {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(paths)}


def verify(manifest):
    for name, expected in manifest.items():
        if sha(REPO / name) != expected:
            raise ValueError("Frozen input/source changed: " + name)


class Stopped(RuntimeError):
    pass


def check_stop(output):
    if any((p / "STOP").exists() for p in (Path(output), GOAL, REPO)):
        raise Stopped("STOP requested")


@contextmanager
def locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity():
    status, created = process.probe_process(os.getpid())
    if status != "live" or type(created) is not int:
        raise ValueError("Actual worker creation identity unavailable")
    return dict(pid=os.getpid(), created=created, parent_pid=os.getppid(),
                executable=sys.executable, command=sys.argv, session_id=uuid.uuid4().hex)


def checkpoint(output, payload):
    """Immutable payloads plus an atomic hash-bound pointer tolerate orphan writes."""
    finite(payload)
    folder = output / "checkpoints"
    folder.mkdir(exist_ok=True)
    path = folder / (uuid.uuid4().hex + ".pt")
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    record = dict(path=path.relative_to(output).as_posix(), sha256=sha(path),
                  phase=payload["learner"]["phase"], counters=payload["learner"]["counts"])
    save(output / "latest.json", record)
    return record


def restore(output):
    record = read(output / "latest.json")
    path = (output / record["path"]).resolve()
    if path.parent != (output / "checkpoints").resolve() or sha(path) != record["sha256"]:
        raise ValueError("Checkpoint path/hash mismatch")
    value = torch.load(path, map_location="cpu", weights_only=False)
    finite(value)
    if record["phase"] != value["learner"]["phase"] or record["counters"] != value["learner"]["counts"]:
        raise ValueError("Checkpoint phase/counter mismatch")
    return value
