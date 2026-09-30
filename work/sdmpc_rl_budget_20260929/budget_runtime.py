"""Load the frozen S-DMPC runtime in a fresh process, never the old RL src."""
from dataclasses import replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SNAPSHOT = REPO / "artifacts/sdmpc_budget_baseline_20260929/source"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def plain(value):
    from dataclasses import asdict, is_dataclass
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_frozen_source(root, expected=None):
    """Verify all payload files and pin the manifest itself across a run."""
    from freeze_runtime import sha256, verify_snapshot
    root = Path(root).resolve()
    snapshot = root.parent if root.name == "source" else root
    manifest_path = snapshot / "manifest.json"
    manifest_hash = sha256(manifest_path)
    if expected is not None and manifest_hash != expected["manifest_sha256"]:
        raise RuntimeError("Frozen snapshot identity mismatch")
    manifest = verify_snapshot(snapshot)
    if sha256(manifest_path) != manifest_hash:
        raise RuntimeError("Frozen snapshot identity changed during verification")
    identity = dict(manifest_sha256=manifest_hash, file_count=manifest["file_count"],
                    total_bytes=manifest["total_bytes"])
    if expected is not None and identity != expected:
        raise RuntimeError("Frozen snapshot identity mismatch")
    return identity


def boot(root, cpu_mask=1):
    root = Path(root).resolve()
    if any(name == "src" or name.startswith("src.") for name in sys.modules):
        raise RuntimeError("Budget runtime requires a fresh process without imported src")
    snapshot_identity = verify_frozen_source(root)
    if root.name != "source":
        root = root / "source"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / ".deps-budget"))
    sys.path.insert(0, str(root / "work/sdmpc_upper_ttt_20260922"))
    from matrix_common import environment, pin
    affinity = pin(cpu_mask)
    environment("SDMPC")
    sys.path.insert(0, str(root / "work/sdmpc_slide_alignment_20260922"))
    from runtime import load, restore_state
    rc, cfg, options, *_ = load(30)
    environment("SDMPC")
    options = replace(options, max_iterations=6)
    os.environ["SDMPC_BUDGET_VARIANT"] = "upper"
    os.environ["SDMPC_EXTERNALITY"] = "on"
    names = ("sdmpc_externality_ablation_20260923", "sdmpc_budget_exception_20260922",
             "sdmpc_central_kkt_reuse_20260922", "sdmpc_selected_dual_20260922",
             "sdmpc_np10_20260922", "sdmpc_relative_band_20260922",
             "sdmpc_group_proxlinear_20260922")
    sys.path[:0] = [str(root / "work" / name) for name in names]
    from anchor_controller import PFOAnchorSDMPC
    from fixed_controller import GridCoordinates
    from externality_policy import audit_rows
    import reused_model
    expected = root / "work/sdmpc_externality_ablation_20260923"
    for name in ("fixed_policy", "band_math", "prox_controller", "fixed_controller",
                 "central", "exception_controller", "anchor_controller", "externality_policy"):
        if Path(sys.modules[name].__file__).resolve().parent != expected:
            raise RuntimeError("Mixed baseline module: " + name)
    if options.horizon_steps != 3 or options.max_iterations != 6:
        raise RuntimeError("Unexpected S-DMPC contract")
    return dict(root=root, rc=rc, cfg=cfg, options=options, baseline=PFOAnchorSDMPC,
                coordinates=GridCoordinates, restore_state=restore_state,
                audit_rows=audit_rows, affinity=affinity, snapshot_identity=snapshot_identity,
                verify=partial(verify_frozen_source, root, snapshot_identity.copy()))


def protocol(runtime, scenario="sweet_170_incident_w"):
    folder = runtime["root"] / "outputs/sdmpc_budget_exception_all_20260922/protocols_0" / scenario
    p = read(folder / "protocol.json")
    cfg, rc = runtime["cfg"], runtime["rc"]
    if rc.to_plain_dict(cfg) != read(folder / "config.json"):
        raise RuntimeError("Historical config differs from scenario protocol")
    for name, expected in p["input_sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Protocol input mismatch: " + name)
    if (p["duration_sec"], p["expected_steps"], p["warmup_steps"], p["seed"]) != (14400, 80, 5, 42):
        raise RuntimeError("Unexpected evaluation protocol")
    profile = rc.FrozenProfile(read(folder / "forecast.json"), cfg.simulation.T_c_sec)
    return folder, p, profile
