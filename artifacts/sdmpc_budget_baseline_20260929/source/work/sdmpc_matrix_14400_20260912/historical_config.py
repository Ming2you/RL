"""Frozen d6241bb R23 protocol configuration.

This helper intentionally imports ``src`` only from ``historical_tree``.  Use it
from a fresh Python interpreter; mixing the current and historical ``src``
packages in one process would make provenance ambiguous.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


ANCHOR_COMMIT = "d6241bb"
CONTROLLER_ID = "P-STACK-WU-FAITHFUL-ALLPRICE-JOINT"
T_TOTAL_SEC = 14_400.0
CONTROL_INTERVAL_SEC = 180.0
WARMUP_NC_STEPS = 5
RANDOM_SEED = 42

SCENARIO_IDS = (
    "sweet_155_w",
    "sweet_170_w",
    "sweet_170_incident_w",
    "sweet_170_skew15_w",
    "sweet_190_w",
)

# Exact BASE_ENV entries in historical_tree/work/launch_final_r23.sh.
R23_ENV: dict[str, str] = {
    "WARMUP_NC_STEPS": "5",
    "FW_BUFFER": "8",
    "TERM_ZG": "1",
    "VFREE": "115",
    "RHO_CRIT": "31.5",
    "TAU_H": "0.0056111",
    "NU_BASE": "22.5",
    "KAPPA": "10",
    "MERGE_DELTA": "0.9",
    "BOX_WALK": "1",
    "BOX_WALK_VG": "1",
    "NP_PD_ITER": "4",
    "NP_BIAS": "1",
    "CROSS_OFF": "1",
    "FAR_STATE_AWARE": "1",
    "FAR_REAL_V": "1",
    "FAR_GATE": "3",
    "BASELINE_BOX": "1",
    "BIAS_SAMPLE": "1",
    "BIAS_POW": "0.4",
    "NASH_SMAX": "10",
    "PFO_SPLIT": "2",
    "PYTHONIOENCODING": "utf-8",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "SPILLBACK": "1",
    "SPILLBACK_WU": "2",
    "SPILLBACK_NREF_U": "800",
    "SPILLBACK_LEAD": "0.5",
    "AUTH_ADAPT": "1",
    "SUP_PFO": "1",
    "AUTH_DEM_HIGH": "23900",
    "AUTH_HIGH_WF": "0.25",
    "AUTH_TRUST_BIG": "6.0",
    "AUTH_DEM_LOW": "0",
}

# Exact ``env -u`` entries in the launch script.
R23_EXPLICIT_UNSET_ENV = ("SEG13", "METER_BOX", "VSL_BOX", "SUP_GATE")

# These names are read through loop variables in run_one, so literal-source
# discovery cannot see them. Clear them when sanitizing a historical process.
R23_DYNAMIC_OPTIONAL_ENV = (
    "FAR_G_FREE",
    "FAR_G_CONG",
    "FAR_NCRIT",
    "FAR_G_FW",
    "SPILLBACK_W",
    "SPILLBACK_WU",
    "SPILLBACK_WF",
    "SPILLBACK_LEAD",
    "SPILLBACK_NREF_U",
    "SPILLBACK_NREF_F",
    "MOVE_W",
    "MOVE_WNUF",
    "MOVE_WNP",
    "URBAN_CLF_NREF",
    "URBAN_CLF_NCRIT",
    "URBAN_CLF_NJAM",
    "URBAN_CLF_PMAX",
    "URBAN_CLF_GMIN",
    "URBAN_CLF_TTAIL",
)

# Files needed to establish the config, demand, initial state, plant, factory,
# and launch provenance. Values are SHA-256 from historical_origin_manifest.json.
REFERENCE_SHA256: dict[str, str] = {
    "src/config/default.yaml": "23304dac6823b133bbd79fa495726b1b208da0c21f195b6dad3b1ba85617130c",
    "src/config/scenarios.yaml": "1974a29820eba3886efb66975de1dfe68d96317ee402ccd72b0f897b55735874",
    "src/models/demand.py": "b7f9a23ce351297169972a3534cc4c318c92e13f79758b502e9ae448269c24b4",
    "src/models/state.py": "8d02c1d227f1c93271494fdc6e78f37901cee51876589c02b2d1a1640c28ed2a",
    "src/models/metanet.py": "064ddc76e325c377853f2ac0cf84de8340cda663af7eb5d067b7ec582feaac69",
    "src/simulation/simulator.py": "34ff26420e4160b13adcef34bf17533756a787c245744ee44a35707af89003dc",
    "src/simulation/coupling.py": "b042fef7c89db1a68e234fc21921a219fc3f4cc950fd75494e6c235a74703f73",
    "src/controllers/f1_wu_faithful_follower.py": "9af39c6ec39d9442e909df969166b3a14d0eb0f4cbb0c19999ae6792e768e05b",
    "src/controllers/stackelberg_wu_metered.py": "41e8dcd2c783dfe9f0dbde119614ddd6f78f1b1b6963f125d2c7c1002a012769",
    "work/run_claude_style_five_controller.py": "ea5d7d89a93dc13b646651bafc4764abada5d7852501f05df40ec394ce20f64f",
    "work/launch_final_r23.sh": "ffd39aa81d2c14a18ca2ea0baf401abd031f707f93a9f1d39384391aafdc0fec",
}

# The integration layer adds only the optional cost_observer callback to the
# frozen coupling function. Its exact hash is admitted explicitly; any further
# edit still fails closed. With cost_observer=None the historical runner follows
# the original path. This exception does not replace REFERENCE_SHA256 provenance.
DECLARED_COMPATIBILITY_SHA256: dict[str, dict[str, str]] = {
    "src/simulation/coupling.py": {
        "f44f721c85035049882402fd79ea9dcab31188ce7aeeb8573566e2f30891820a": (
            "optional cost_observer callback only"
        ),
    },
}

_HERE = Path(__file__).resolve().parent
DEFAULT_HISTORICAL_ROOT = _HERE / "historical_tree"
COMPATIBILITY_EVIDENCE_PATH = _HERE / "historical_compatibility_check.json"
_ENV_LITERAL = re.compile(r"environ(?:\.get)?\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']|environ\[\s*[\"']([A-Z][A-Z0-9_]*)[\"']")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_historical_tree(
    historical_root: str | Path = DEFAULT_HISTORICAL_ROOT,
    *,
    allow_declared_compatibility: bool = False,
) -> dict[str, str]:
    """Verify d6241bb files, optionally admitting one exact observer-only patch."""

    root = Path(historical_root).resolve()
    observed: dict[str, str] = {}
    failures: list[str] = []
    for relative, expected in REFERENCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = _sha256(path)
        observed[relative] = actual
        declared = DECLARED_COMPATIBILITY_SHA256.get(relative, {})
        if actual != expected and not (allow_declared_compatibility and actual in declared):
            failures.append(f"sha256 mismatch: {relative}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("Historical tree verification failed:\n" + "\n".join(failures))
    return observed


def verify_compatibility_evidence(
    evidence_path: str | Path = COMPATIBILITY_EVIDENCE_PATH,
) -> dict[str, Any]:
    """Require the recorded exact-equivalence check for the observer patch."""

    path = Path(evidence_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing historical compatibility evidence: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("pass") is not True:
        raise RuntimeError(f"Historical compatibility evidence is not PASS: {path}")
    scope = str(result.get("scope", ""))
    required_scope = "original/observer-off/observer-on exact equality"
    if required_scope not in scope:
        raise RuntimeError(f"Historical compatibility evidence has unexpected scope: {scope!r}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "pass": True,
        "scope": scope,
        "ledger_max_substep_residual": result.get("ledger_max_substep_residual"),
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _import_historical_runtime(historical_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    loaded_src = sys.modules.get("src")
    loaded_file = getattr(loaded_src, "__file__", None) if loaded_src else None
    if loaded_file and not _is_within(Path(loaded_file), historical_root):
        raise RuntimeError(
            "A non-historical 'src' package is already loaded. Start a fresh Python "
            "process and call build_historical_config before importing repository src modules."
        )

    root_text = str(historical_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    state_module = importlib.import_module("src.models.state")
    demand_module = importlib.import_module("src.models.demand")
    simulator_module = importlib.import_module("src.simulation.simulator")
    for module in (state_module, demand_module, simulator_module):
        module_file = Path(module.__file__).resolve()
        if not _is_within(module_file, historical_root):
            raise RuntimeError(f"Historical runtime contamination: {module.__name__} -> {module_file}")
    return (
        state_module.ExperimentConfig,
        state_module.TrafficState,
        demand_module.DemandProfile,
        demand_module.load_scenarios,
        demand_module.apply_scenario_network_overrides,
    )


def to_plain_dict(value: Any) -> Any:
    """Serialize dataclass fields and runtime-added attributes without loss."""

    if is_dataclass(value):
        declared = {item.name for item in fields(value)}
        out = {name: to_plain_dict(getattr(value, name)) for name in declared}
        for name, item in vars(value).items():
            if name not in declared and not name.startswith("_"):
                out[name] = to_plain_dict(item)
        return out
    if isinstance(value, Mapping):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, set):
        return sorted(to_plain_dict(item) for item in value)
    return value


def restore_historical_config(raw: Mapping[str, Any]) -> Any:
    """원 검증 constructor를 거친 뒤 runtime 추가 속성을 손실 없이 복원한다."""
    import copy
    from src.models.state import ExperimentConfig

    def declared_only(value: Any, template: Any) -> Any:
        if is_dataclass(template):
            return {item.name: declared_only(value[item.name], getattr(template, item.name))
                    for item in fields(template) if item.name in value}
        return copy.deepcopy(value)

    def restore_extras(target: Any, value: Mapping[str, Any]) -> None:
        declared = {item.name for item in fields(target)}
        for name, item in value.items():
            if name in declared:
                child = getattr(target, name)
                if is_dataclass(child):
                    restore_extras(child, item)
            else:
                if name.startswith("_") or hasattr(target, name):
                    raise ValueError(f"Invalid runtime config attribute: {name}")
                setattr(target, name, copy.deepcopy(item))

    # from_dict의 dataclass constructor에는 선언 필드만 전달한다. TERM_ZG 등의
    # 동적 필드는 그 뒤 같은 section 객체에 복원하여 native run_one 설정을 보존한다.
    cfg = ExperimentConfig.from_dict(declared_only(raw, ExperimentConfig()))
    restore_extras(cfg, raw)
    cfg.validate()
    if to_plain_dict(cfg) != dict(raw):
        raise ValueError("Historical config changed on complete restoration")
    return cfg


def _canonical_json(value: Any) -> str:
    return json.dumps(to_plain_dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_sha256(value: Any) -> str:
    """Stable SHA-256 for a dataclass/config represented as canonical JSON."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _apply_r23_effective_config(cfg: Any) -> None:
    """Apply run_one and factory mutations that are active under R23_ENV."""

    # run_one physical/model mutations, in historical source order.
    cfg.network.metanet_tau_h = 0.0056111
    cfg.network.metanet_delta_merge = 0.9
    cfg.mpc.leader_mfd_far_real_speed = True
    cfg.network.metanet_nu_km2_h = 22.5
    cfg.network.metanet_kappa_veh_km_lane = 10.0
    cfg.network.v_free = 115.0
    cfg.network.rho_crit = 31.5
    cfg.mpc.leader_mfd_far_state_aware = True
    cfg.network.freeway_buffer_segments = 8
    cfg.network.terminal_zero_gradient = True
    cfg.mpc.leader_rollout_box_walk = True
    cfg.mpc.leader_rollout_box_walk_vg = True
    cfg.mpc.baseline_move_box = True
    cfg.mpc.leader_spillback_enabled = True
    cfg.mpc.leader_spillback_urban_weight = 2.0
    cfg.mpc.leader_spillback_lead_h = 0.5
    cfg.mpc.leader_spillback_nref_urban = 800.0

    # Mutations after make_controller in run_one. The controller retains this
    # same cfg object, so pre-applying them is behaviorally equivalent.
    cfg.mpc.np_bias_correction = True
    cfg.mpc.np_primal_dual_iters = 4

    # ALLPRICE-JOINT factory side effects under the launch environment.
    cfg.mpc.leader_value_depth = 3
    cfg.mpc.leader_skip_local_refinement = True
    cfg.mpc.leader_rollout_early_stop = True
    cfg.mpc.leader_bias_sample_pow = 0.4


def discover_historical_env_literals(
    historical_root: str | Path = DEFAULT_HISTORICAL_ROOT,
) -> set[str]:
    """Find literal experiment environment keys read by the frozen Python tree."""

    root = Path(historical_root).resolve()
    keys: set[str] = set()
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in _ENV_LITERAL.finditer(text):
            keys.add(match.group(1) or match.group(2))
    runner = root / "work" / "run_claude_style_five_controller.py"
    text = runner.read_text(encoding="utf-8")
    for match in _ENV_LITERAL.finditer(text):
        keys.add(match.group(1) or match.group(2))
    return keys


@contextmanager
def historical_environment(
    historical_root: str | Path = DEFAULT_HISTORICAL_ROOT,
    *,
    sanitize_experiment_knobs: bool = True,
) -> Iterator[dict[str, Any]]:
    """Temporarily reproduce the R23 launch environment and restore the host.

    Sanitization clears literal experiment knobs read by the frozen source before
    applying ``BASE_ENV``. This models the clean historical shell and prevents a
    current sweep variable from silently changing the old factory. The four
    variables explicitly removed by ``launch_final_r23.sh`` are always absent.
    """

    root = Path(historical_root).resolve()
    keys = set(R23_ENV) | set(R23_EXPLICIT_UNSET_ENV) | set(R23_DYNAMIC_OPTIONAL_ENV)
    if sanitize_experiment_knobs:
        keys |= discover_historical_env_literals(root)
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(R23_ENV)
        for key in R23_EXPLICIT_UNSET_ENV:
            os.environ.pop(key, None)
        yield {
            "set": dict(R23_ENV),
            "explicit_unset": list(R23_EXPLICIT_UNSET_ENV),
            "sanitized_keys": sorted(keys - set(R23_ENV) - set(R23_EXPLICIT_UNSET_ENV)),
        }
    finally:
        for key in keys:
            old = previous[key]
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def build_historical_config(
    scenario_name: str,
    historical_root: str | Path = DEFAULT_HISTORICAL_ROOT,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build the effective d6241bb R23 config, scenario, demand, and metadata."""

    if scenario_name not in SCENARIO_IDS:
        raise ValueError(f"R23 scenario must be one of {SCENARIO_IDS}; got {scenario_name!r}")

    root = Path(historical_root).resolve()
    source_hashes = verify_historical_tree(root, allow_declared_compatibility=True)
    compatibility_patches = {
        relative: DECLARED_COMPATIBILITY_SHA256[relative][actual]
        for relative, actual in source_hashes.items()
        if actual != REFERENCE_SHA256[relative]
    }
    compatibility_evidence = (
        verify_compatibility_evidence(root.parent / "historical_compatibility_check.json")
        if compatibility_patches
        else None
    )
    ExperimentConfig, TrafficState, DemandProfile, load_scenarios, apply_scenario = (
        _import_historical_runtime(root)
    )
    overrides = {
        "mpc": {
            "relaxed_quantized_controls": True,
            "grid_parallel_backend": "serial",
            "leader_search_mode": "grid",
            "stackelberg_leader_parallel_backend": "serial",
        },
        "simulation": {"T_total": T_TOTAL_SEC},
    }
    cfg = ExperimentConfig.from_file(root / "src" / "config" / "default.yaml", overrides)
    scenario = load_scenarios(root / "src" / "config" / "scenarios.yaml")[scenario_name]
    cfg = apply_scenario(cfg, scenario)
    _apply_r23_effective_config(cfg)
    profile = DemandProfile(cfg, scenario)
    initial_state = TrafficState.initial(cfg)

    metadata: dict[str, Any] = {
        "protocol": "final-r23-14400",
        "anchor_commit": ANCHOR_COMMIT,
        "controller_id": CONTROLLER_ID,
        "scenario_id": scenario_name,
        "historical_root": str(root),
        "duration_sec": T_TOTAL_SEC,
        "control_interval_sec": CONTROL_INTERVAL_SEC,
        "executed_steps": 80,
        "warmup_nc_steps": WARMUP_NC_STEPS,
        "control_start_sec": WARMUP_NC_STEPS * CONTROL_INTERVAL_SEC,
        "random_seed": RANDOM_SEED,
        "launch_environment": {
            "set": dict(R23_ENV),
            "explicit_unset": list(R23_EXPLICIT_UNSET_ENV),
        },
        "runtime_only": {
            "far_gate_mode": 3,
            "nash_smax": 10,
            "pfo_split": "2 (truthy; activates two regional agents per freeway link)",
            "auth_adapt": True,
            "supervisor_pfo": True,
            "cross_prices_forced_off": True,
            "biased_sampling_patch_required_from_factory": True,
        },
        "effective_config_sha256": object_sha256(cfg),
        "initial_state_sha256": object_sha256(initial_state),
        "runtime_added_config_fields": {
            "network.terminal_zero_gradient": bool(cfg.network.terminal_zero_gradient),
            "mpc.leader_skip_local_refinement": bool(cfg.mpc.leader_skip_local_refinement),
            "mpc.leader_rollout_early_stop": bool(cfg.mpc.leader_rollout_early_stop),
        },
        "source_sha256": source_hashes,
        "origin_source_sha256": dict(REFERENCE_SHA256),
        "declared_compatibility_patches": compatibility_patches,
        "compatibility_evidence": compatibility_evidence,
        "source_paths": {key: str(root / key) for key in REFERENCE_SHA256},
        "notes": [
            "Use historical_environment() while constructing and running the old factory.",
            "TrafficState.initial leaves buffer maps empty; metanet.py initializes each 8-cell buffer to density 0 and speed 115 on the first step.",
            "AUTH_ADAPT, FAR_GATE, SUP_PFO, NASH_SMAX, PFO_SPLIT, and warmup behavior are runtime logic, not fully representable in ExperimentConfig.",
        ],
    }
    return cfg, scenario, profile, metadata


def build_historical_initial_state(cfg: Any, historical_root: str | Path = DEFAULT_HISTORICAL_ROOT) -> tuple[Any, str]:
    """Return the exact pre-step historical state and its canonical SHA-256."""

    root = Path(historical_root).resolve()
    verify_historical_tree(root, allow_declared_compatibility=True)
    _, TrafficState, _, _, _ = _import_historical_runtime(root)
    state = TrafficState.initial(cfg)
    return state, object_sha256(state)


def protocol_metadata_without_import(
    historical_root: str | Path = DEFAULT_HISTORICAL_ROOT,
) -> Mapping[str, Any]:
    """Return launch/source metadata without importing NumPy or the old runtime."""

    root = Path(historical_root).resolve()
    hashes = verify_historical_tree(root, allow_declared_compatibility=True)
    compatibility_patches = {
        relative: DECLARED_COMPATIBILITY_SHA256[relative][actual]
        for relative, actual in hashes.items()
        if actual != REFERENCE_SHA256[relative]
    }
    return {
        "anchor_commit": ANCHOR_COMMIT,
        "controller_id": CONTROLLER_ID,
        "scenario_ids": list(SCENARIO_IDS),
        "duration_sec": T_TOTAL_SEC,
        "warmup_nc_steps": WARMUP_NC_STEPS,
        "random_seed": RANDOM_SEED,
        "launch_environment": dict(R23_ENV),
        "explicit_unset": list(R23_EXPLICIT_UNSET_ENV),
        "source_sha256": hashes,
        "origin_source_sha256": dict(REFERENCE_SHA256),
        "declared_compatibility_patches": compatibility_patches,
        "compatibility_evidence": (
            verify_compatibility_evidence(root.parent / "historical_compatibility_check.json")
            if compatibility_patches
            else None
        ),
    }
