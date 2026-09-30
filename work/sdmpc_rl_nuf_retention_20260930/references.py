"""Read-only, pinned v2 references; no environment checkpoint deserialization."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_balanced_goal_20260930/local_budget_v2"
RECOVERY = REPO / "work/sdmpc_rl_export_recovery_20260930"
ROOT_PINS = {
    "completion.json": "5b6e645f6f6ca64c4f593c52d19ee3ab5bda7ae3c3d31f423d6fae2d2f3ef4a3",
    "comparison.json": "c294c1f5b2100daeaccb4c927357a8bc73d44391f34641fd37b4ba5b2555c4cd",
}
HELPER_PINS = {
    "export_recovery.py": "f79513942cd892e346dd68a188ecd2221e862b55558170c3a35d90ed8f80a4b3",
    "supervise.py": "a119ee8a87b9c280d9a887f253f02308331543f04eb9331cc474eb8ca945e5a9",
    "coordinator_bootstrap.py": "9aead7b555b2bb192932a718ac29c07c28b1a4c8aa3673b3bbd7466f871622e3",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_hash(path, expected):
    if sha(path) != expected:
        raise ValueError("Reference source/output changed: " + str(path))


def verify_helper_sources():
    for name, expected in HELPER_PINS.items():
        check_hash(RECOVERY / name, expected)


def manifest():
    """Fast hash reconciliation after the original validators authenticate all ten runs."""
    verify_helper_sources()
    for name, expected in ROOT_PINS.items():
        check_hash(ROOT / name, expected)
    done = read(ROOT / "completion.json")
    plan = done["plan"]
    if done["status"] != "completed" or read(ROOT / "plan.json") != plan:
        raise ValueError("Reference plan/completion mismatch")
    for name, expected in plan["identity"]["local_sources"].items():
        check_hash(REPO / name, expected)
    entries = {}
    for job in plan["jobs"]:
        key = job["key"]
        folder = ROOT / key
        check_hash(folder / "completion.json", done["child_completion_sha256"][key])
        result = read(folder / "completion.json")
        settings = read(folder / "settings.json")
        if settings != result["settings"] or settings["identity"] != plan["identity"]:
            raise ValueError("Reference settings/source mismatch")
        for field in ("behavior", "scenario", "training_seed", "exploration_seed", "cpu_mask"):
            if settings[field] != job[field]:
                raise ValueError("Reference slot mismatch: " + field)
        for name, expected in result["outputs_sha256"].items():
            check_hash(folder / name, expected)
        repair = result.get("export_recovery")
        if repair:
            if repair["receipt"] != "export-receipt.json":
                raise ValueError("Unknown reference export receipt")
            check_hash(folder / repair["receipt"], repair["sha256"])
            receipt = read(folder / repair["receipt"])
            if receipt["helper_sha256"] != HELPER_PINS or receipt["output_sha256"] != result["outputs_sha256"]:
                raise ValueError("Reference repaired receipt mismatch")
            for name, expected in receipt["input_sha256"].items():
                path = folder / (".export-recovery/original-status.json" if name == "status.json" else name)
                check_hash(path, expected)
        entries[key] = dict(
            path=str(folder.resolve()), completion_sha256=done["child_completion_sha256"][key],
            outputs_sha256=result["outputs_sha256"], export_recovery=repair,
            **{k: settings[k] for k in ("run_id", "scenario", "behavior", "training_seed",
                "exploration_seed", "cpu_mask", "profile_sha256", "capacity", "warmup_ttt")},
            contract_sha256=digest(settings["contract"]),
            physical_config_sha256=digest(settings["physical_config"]),
            source_identity_sha256=digest(settings["identity"]))
    if len(entries) != 10 or len({r["run_id"] for r in entries.values()}) != 10:
        raise ValueError("Reference cohort must contain ten unique actual runs")
    return dict(root=str(ROOT.resolve()), root_sha256=ROOT_PINS, helper_sha256=HELPER_PINS,
                source_identity=plan["identity"], entries=entries)


def authenticate():
    """Fresh interpreter prevents same-name old/new collector module collisions."""
    result = subprocess.run([sys.executable, "-B", str(HERE / "reference_check.py")],
        cwd=REPO, text=True, encoding="utf-8", capture_output=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    checked = json.loads(result.stdout)
    if checked != manifest():
        raise ValueError("References changed during isolated authentication")
    return checked


def pair(scenario, references=None):
    references = manifest() if references is None else references
    return {behavior: references["entries"][f"{behavior}/{scenario}"] for behavior in ("carry", "local")}


def validate_binding(settings, references=None, warmup=True):
    references = manifest() if references is None else references
    expected = pair(settings["scenario"], references)
    if settings["paired_references"] != expected:
        raise ValueError("Paired reference source/hash mismatch")
    from exploration import TREATMENT
    if settings["treatment"] != TREATMENT:
        raise ValueError("Treatment differs")
    old_identity = references["source_identity"]
    for key in ("physical", "runtime", "gate_sha256", "contract_sha256"):
        if settings["identity"][key] != old_identity[key]:
            raise ValueError("Paired physical/runtime source mismatch: " + key)
    if not settings["run_id"] or settings["run_id"] in {r["run_id"] for r in references["entries"].values()}:
        raise ValueError("New run must have its own actual run ID")
    for ref in expected.values():
        for key in ("scenario", "profile_sha256", "training_seed", "exploration_seed", "cpu_mask", "capacity"):
            if settings[key] != ref[key]:
                raise ValueError("Paired reference/profile mismatch: " + key)
        if (digest(settings["contract"]) != ref["contract_sha256"] or
                digest(settings["physical_config"]) != ref["physical_config_sha256"]):
            raise ValueError("Paired physical contract mismatch")
        if warmup and abs(settings["warmup_ttt"] - ref["warmup_ttt"]) > 1e-8:
            raise ValueError("Paired warmup mismatch")
    return expected
