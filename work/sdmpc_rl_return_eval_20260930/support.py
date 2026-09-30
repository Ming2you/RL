"""Canonical-only constants and explicitly bound reuse of frozen Task 2 helpers."""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOAL = REPO / "results/sdmpc_rl_balanced_goal_20260930"
ROOT = GOAL / "return_canonical_v1"
MODEL = GOAL / "return_mc_v1"
WAVE = GOAL / "return_policy_wave_v1"
WAVE_SOURCE = REPO / "work/sdmpc_rl_return_wave_20260930"
CENTER = REPO / "results/sdmpc_rl_multi_20260929/pilot_v1/center"
MODEL_SHA = "820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6"
COMPLETION_SHA = "a815f65f315ce1a73cce8a3754b9edb3fac9dbed6a8fd0aece9b9238ea532675"
MODEL_SPEC_SHA = "3610bb43c22430a3fb9ba95f7f14175b2b29b68a2000d6bfe1af04ec4c8ff811"
ACTOR_SHA = "9956ba4f6f5d64c7a65bf05bdab47984efe041e15358dafebbb9bfd731d44877"
CONTRACT_SHA = "d6d14c4c18a7f8cfc88969bf0e9b8231ba5c452600593db49b125c1e5730eb70"
FORMAT = "sdmpc-fixed-policy-canonical-v1"
CONTINUATION = "pi1_recorded_on_policy_MC"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_hash(path, expected):
    if file_hash(path) != expected:
        raise ValueError("Frozen input changed: " + str(path))


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# Authenticate transitive code before importing any reused helper.
check_hash(MODEL / "completion.json", COMPLETION_SHA)
MODEL_DONE = json_read(MODEL / "completion.json")
DEPENDENCIES = dict(MODEL_DONE["settings"]["sources"], **MODEL_DONE["settings"]["data"]["files"])
for _name, _sha in DEPENDENCIES.items():
    _path = (REPO / _name).resolve()
    if not _path.is_relative_to(REPO.resolve()):
        raise ValueError("Dependency escaped repository")
    if _path.suffix == ".py":
        check_hash(_path, _sha)
check_hash(WAVE_SOURCE / "wave_support.py",
           "b9af68ebbe4e0b8992bd1ee672b01b0b87b1829d29635cd487f58e3d44e21418")
sys.path.insert(0, str(WAVE_SOURCE))
import wave_support as old
sys.path.insert(0, str(HERE))

np, torch = old.np, old.torch
SCENARIOS, MASKS = old.SCENARIOS, old.MASKS
BASE_TTT = dict(zip(SCENARIOS, (3103.0110715680044, 3935.903236508048,
    5546.224352256691, 4250.876599300032, 6604.2970168093225)))
SPEC = dict(format=FORMAT, model_sha256=MODEL_SHA, model_spec_sha256=MODEL_SPEC_SHA,
    model_completion_sha256=COMPLETION_SHA, actor_sha256=ACTOR_SHA,
    physical_contract_sha256=CONTRACT_SHA, scenarios=list(SCENARIOS), baseline_ttt=BASE_TTT,
    training_seed=None, evaluation=True, eval_only=True, exploration=False, policy_q=None,
    learning=False, guard_mode="physical", gamma=1., reward_divisor=100.,
    warmup_steps=5, controlled_steps=75, total_seconds=14400, control_seconds=180,
    observations=2367, width=64, hidden_layers=2, activation="ReLU", output="tanh",
    action_bounds=[.2, .1], action_dtype="float32", projection_dtype="float64",
    delta_scale=[50., 1000.], capacity=6000., anchor="previous_executed",
    initial_recovery_pfo=True, lower_candidates=1, threads=1,
    critic_continuation=CONTINUATION, acceptance="each delta > max(1e-6,1e-8*baseline)",
    passing_status="eligible_for_separate_reproduction", goal_achieved=False)
read, plain, digest = old.read, old.plain, old.digest
finite, same_cost, finite_tree = old.finite, old.same_cost, old.finite_tree
save, save_once, save_tensor = old.save, old.save_once, old.save_tensor
persist_checkpoint, checkpoint_record = old.persist_checkpoint, old.checkpoint_record
exclusive_run, process_identity, session_ids = old.exclusive_run, old.process_identity, old.session_ids
boot, observation_schema = old.boot, old.observation_schema
DEFAULT_SNAPSHOT, OLD, GATE, GATE_SHA = old.DEFAULT_SNAPSHOT, old.OLD, old.GATE, old.GATE_SHA
TIMING_SCOPE = ("Locked worker wall/process-CPU sessions include authentication, boot, reset/restore, actor, "
    "forecast/observation/PFO/reference/lower/guard/plant work, validation, checkpoints, export and close. "
    "Exclude pre-lock review/input admission, final end/timing/summary/completion publication, lock release, "
    "interpreter and parent overhead, and readout. Finalization-only resumes do no physical work and are "
    "publication overhead outside the frozen session ledger. Interrupted ends are UNKNOWN; known seconds "
    "are a lower bound. Whole restore is already inside session totals once. Parallel sums are not wall elapsed.")
Stopped = old.Stopped
SPEC_SHA = digest(SPEC)


def reuse(filename, names, namespace):
    path = WAVE_SOURCE / filename
    expected = DEPENDENCIES[path.relative_to(REPO).as_posix()]
    return old.definitions(path, names, namespace, expected)


def sources():
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(HERE.glob("*.py"))}


def check_stop(output):
    if any((p / "STOP").exists() for p in (REPO, GOAL, ROOT, output)):
        raise Stopped("STOP requested; marker retained")


def import_boundary(physical=False):
    old.import_boundary(physical)
    if any(name in sys.modules for name in ("mc_common", "mc_data", "mc_learner", "mc_parent_offline_runtime")):
        raise ValueError("Offline learner imported into canonical process")


def preservation():
    paths = {REPO / name for name in DEPENDENCIES}
    for folder in (MODEL, CENTER, WAVE):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    paths.update(WAVE_SOURCE.glob("*.py"))
    paths.update(OLD.glob("*.py"))
    manifest = DEFAULT_SNAPSHOT.parent / "manifest.json"
    paths.update((manifest, GATE))
    paths.update((REPO / "docs/rl_budget_return_mc_results_20260930.md",
                  REPO / ".superpowers/sdd/rl_budget_return_initialized_policy_20260930/task-4-brief.md",
                  WAVE_SOURCE / "test-evidence/d45d6eca/evidence.json"))
    paths.update(DEFAULT_SNAPSHOT.parent / row["snapshot_relative"] for row in read(manifest)["files"])
    return {p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(paths)}


def verify_files(files):
    old.verify_files(files)


# The original function uses a module-level model constant. Bind its unchanged
# definition to this contract in a private namespace, without patching that module.
_timing = dict(vars(old), MODEL_SHA=MODEL_SHA, TIMING_SCOPE=TIMING_SCOPE)
reuse("wave_support.py", ("session_timing",), _timing)
import restore_accounting


def session_timing(output, settings, required=()):
    result = _timing["session_timing"](output, settings, required)
    known = sum(finite(read(output / "sessions" / f"{sid}.end.json")["elapsed_cpu_seconds"], "session CPU")
                for sid in result["session_ids"] if sid not in result["unknown_sessions"])
    result.update(known_elapsed_cpu_seconds=known,
                  elapsed_cpu_seconds=None if result["unknown_sessions"] else known)
    return result

