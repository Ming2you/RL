"""Freeze the inspected budget baseline without importing its runtime.

Only the explicit sources/fixtures below are copied. A completed snapshot is
append- and overwrite-protected by inventory/hash verification, not by an ACL.
Original JSON (including absolute provenance) is never rewritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import subprocess


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path.home() / "Documents" / "Numerical Simulation"
DEFAULT_DESTINATION = REPO / "artifacts" / "sdmpc_budget_baseline_20260929"
HISTORICAL = "work/sdmpc_matrix_14400_20260912/historical_tree"
ENTRY = "work/sdmpc_externality_ablation_20260923/run_scenario.py"
PROTOCOLS = "outputs/sdmpc_budget_exception_all_20260922/protocols_0"
PROBES = (
    "outputs/sdmpc_budget_ablation_20260922/attempt_0/runs/"
    "sweet_170_incident_w/upper"
)
BOOTSTRAP_CELL = (
    "outputs/sdmpc_frozen_price_20260912/matrix_C_attempt_0/"
    "interval_vsl/sweet_190_w"
)
SAVED_DECISION = (
    "outputs/extended_matrix_no_slsqp_20260920/attempt_1/"
    "SDMPC6/sweet_190_skew15_w"
)
SCENARIOS = (
    "sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w",
    "sweet_190_w", "sweet_190_skew15_w", "sweet_190_incident_w", "sweet_220_w",
    "sweet_220_skew15_w", "sweet_220_incident_w",
)
PROBE_NAMES = (
    "plant_004.json", "plant_009.json", "plant_019.json", "plant_029.json",
    "plant_049.json", "plant_069.json", "completion.json", "contract.json",
)

# Includes imports made during bootstrap, even when their returned class is unused.
RUNTIME_GROUPS = {
    "work/sdmpc_externality_ablation_20260923": (
        "run_scenario.py", "anchor_controller.py", "band_math.py", "central.py",
        "exception_controller.py", "externality_policy.py", "fixed_controller.py",
        "fixed_policy.py", "prox_controller.py",
    ),
    "work/sdmpc_upper_ttt_20260922": ("matrix_common.py",),
    "work/sdmpc_slide_alignment_20260922": (
        "runtime.py", "run_validation.py", "group_block_engine.py",
    ),
    "work/sdmpc_engine_speed_20260912": (
        "profile_current.py", "baseline_controller/continuous_vsl_controller.py",
        "baseline_controller/frozen_price_controller.py",
    ),
    "work/sdmpc_matrix_14400_20260912": ("run_cell.py", "historical_config.py"),
    "work/sdmpc_group_proxlinear_20260922": ("local_qp.py",),
    "work/sdmpc_central_kkt_reuse_20260922": ("central_controller.py", "reused_model.py"),
    "work/sdmpc_selected_dual_20260922": ("controller.py", "multiplier.py"),
    "work/sdmpc_relative_band_20260922": ("policy.py",),
    "work/sdmpc_block_speed_ordered_20260919": ("blocks.py",),
    "work/sdmpc_sparse_local_20260916": ("sparse/dual.py",),
    "scripts": ("run_sdmpc_experiment.py", "player_sdmpc_validation_provenance.py"),
}

# Static transitive import closure of run_cell.load_runtime and its factory.
# Includes conditional factory imports; excludes tests, experiments, and old RL.
HISTORICAL_SOURCES = (
    "src/__init__.py", "src/analysis/__init__.py", "src/analysis/free_flow_reference.py",
    "src/controllers/__init__.py", "src/controllers/biasedsample_mpc.py",
    "src/controllers/centralized_mpc.py", "src/controllers/classical_hierarchical.py",
    "src/controllers/distributed_coordinator.py", "src/controllers/f1_wu_faithful_follower.py",
    "src/controllers/freeway_follower.py", "src/controllers/gradseed_mpc.py",
    "src/controllers/grid_parallel.py", "src/controllers/inflow_outflow_allocation.py",
    "src/controllers/joint_wu_controllers.py", "src/controllers/leader.py",
    "src/controllers/local_freeway_plant.py", "src/controllers/local_signal_plant.py",
    "src/controllers/nash_solver.py", "src/controllers/player_sensitivity_dmpc.py",
    "src/controllers/relaxed_quantization.py", "src/controllers/segment_local_plant.py",
    "src/controllers/sensitivity_dmpc.py", "src/controllers/simplified_inflow_outflow_allocation.py",
    "src/controllers/spillback_constraints.py", "src/controllers/stackelberg_mpc.py",
    "src/controllers/stackelberg_wu_metered.py", "src/controllers/structured_grid.py",
    "src/controllers/urban_follower.py", "src/controllers/wu_distributed.py",
    "src/controllers/wu_faithful_follower.py", "src/evaluation/__init__.py",
    "src/evaluation/metrics.py", "src/models/__init__.py", "src/models/demand.py",
    "src/models/grid_topology.py", "src/models/metanet.py", "src/models/state.py",
    "src/models/urban_queue_model.py", "src/simulation/__init__.py",
    "src/simulation/baseline.py", "src/simulation/coupling.py",
    "src/simulation/player_cost_accounting.py", "src/simulation/simulator.py",
    "work/run_claude_style_five_controller.py",
)
FIXTURES = (
    f"{BOOTSTRAP_CELL}/protocol_snapshot/protocol.json",
    f"{SAVED_DECISION}/decision_030/input.json",
    f"{SAVED_DECISION}/factory_config.json",
    f"{SAVED_DECISION}/solver_options.json",
    "work/sdmpc_matrix_14400_20260912/historical_environment.json",
    "work/sdmpc_matrix_14400_20260912/historical_compatibility_check.json",
    f"{HISTORICAL}/historical_origin_manifest.json",
    f"{HISTORICAL}/sdmpc_overlay_initial_manifest.json",
    f"{HISTORICAL}/src/config/default.yaml",
    f"{HISTORICAL}/src/config/scenarios.yaml",
    f"{HISTORICAL}/work/launch_final_r23.sh",
)


class SnapshotError(ValueError):
    """An unsafe path, changed input, or conflicting destination was found."""


def relative_path(value: str) -> str:
    """Accept canonical, portable relative file names, including on Windows."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise SnapshotError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive:
        raise SnapshotError(f"Absolute/drive path forbidden: {value!r}")
    for part in value.split("/"):
        if (part in ("", ".", "..") or part.endswith((" ", "."))
                or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
                or PureWindowsPath(part).is_reserved()):
            raise SnapshotError(f"Unsafe path component: {value!r}")
    return path.as_posix()


def no_links(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for candidate in (*reversed(path.parents), path):
        if os.path.lexists(candidate):
            info = candidate.lstat()
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
                raise SnapshotError(f"Symlink/junction/reparse point forbidden: {candidate}")
    return path


def checked_path(root: Path, relative: str) -> Path:
    root = no_links(root)
    path = no_links(root / relative_path(relative))
    if not path.resolve().is_relative_to(root.resolve()):
        raise SnapshotError(f"Path escapes root: {path}")
    return path


def validate_roots(source: Path, destination: Path) -> tuple[Path, Path]:
    source, destination = no_links(source).resolve(), no_links(destination).resolve()
    if not source.is_dir():
        raise SnapshotError(f"Missing source directory: {source}")
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise SnapshotError("Source and destination trees must be disjoint")
    if destination.exists() and not destination.is_dir():
        raise SnapshotError(f"Destination is not a directory: {destination}")
    return source, destination


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_files(source: Path) -> tuple[dict[str, str], list[str]]:
    selected = {f"{folder}/{name}": "runtime_source"
                for folder, names in RUNTIME_GROUPS.items() for name in names}
    selected.update({f"{HISTORICAL}/{name}": "historical_runtime_source"
                     for name in HISTORICAL_SOURCES})
    selected.update({name: "bootstrap_fixture_or_provenance" for name in FIXTURES})
    for scenario in SCENARIOS:
        for name in ("config.json", "forecast.json", "initial_state.json",
                     "protocol.json", "scenario.json"):
            selected[f"{PROTOCOLS}/{scenario}/{name}"] = "scenario_protocol"
        for name in ("historical_provenance.json", "source_manifest.json"):
            relative = f"{PROTOCOLS}/{scenario}/{name}"
            if checked_path(source, relative).exists():
                selected[relative] = "scenario_protocol"
    missing_probes = []
    for name in PROBE_NAMES:
        relative = f"{PROBES}/{name}"
        if checked_path(source, relative).exists():
            selected[relative] = "historical_probe_input"
        else:
            missing_probes.append(relative)
    for relative in selected:
        if not checked_path(source, relative).is_file():
            raise SnapshotError(f"Missing required source/input: {relative}")
    protocol_bytes = sum(checked_path(source, name).stat().st_size
                         for name, role in selected.items() if role == "scenario_protocol")
    if protocol_bytes > 2 * 1024 * 1024:
        raise SnapshotError("Protocol folders exceed the inspected tiny-input budget (2 MiB)")
    return dict(sorted(selected.items())), missing_probes


def git_provenance(source: Path) -> dict:
    warnings = set()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-c", "core.quotepath=false", "-C",
             str(source), *arguments], check=True, capture_output=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        if result.stderr:
            warnings.add(result.stderr.decode("utf-8", errors="replace").strip())
        return result.stdout.decode("utf-8", errors="surrogateescape")

    if Path(git("rev-parse", "--show-toplevel").strip()).resolve() != source.resolve():
        raise SnapshotError("Source root must be the original Git project root")
    revision = git("rev-parse", "HEAD").strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD").strip()
    status_text = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "revision": revision, "branch": branch, "dirty": bool(status_text),
        "status_command": "git --no-optional-locks -c core.quotepath=false status --porcelain=v1 --untracked-files=all",
        "status_porcelain_v1": status_text,
        "status_sha256": hashlib.sha256(status_text.encode("utf-8", "surrogateescape")).hexdigest(),
        "warnings": sorted(warnings),
    }


def destination_inventory(destination: Path) -> set[str]:
    files = set()
    if destination.exists():
        for directory, dirs, names in os.walk(destination, followlinks=False):
            for name in dirs + names:
                path = no_links(Path(directory) / name)
                if not path.is_dir():
                    if not path.is_file():
                        raise SnapshotError(f"Non-regular destination entry: {path}")
                    files.add(path.relative_to(destination).as_posix())
    return files


def copy_verified(source: Path, destination: Path, expected: str) -> str:
    """Exclusive binary creation; matching existing files remain untouched."""
    if destination.exists():
        observed = sha256(destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open("rb") as original, destination.open("xb") as copied:
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                copied.write(chunk)
                digest.update(chunk)
            copied.flush()
            os.fsync(copied.fileno())
        observed = digest.hexdigest()
    if observed != expected or sha256(destination) != expected:
        raise SnapshotError(f"Copy/hash mismatch: {destination}")
    return observed


def verify_snapshot(destination: Path) -> dict:
    """Verify the completed snapshot offline, without consulting original paths."""
    destination = no_links(destination).resolve()
    manifest = json.loads(checked_path(destination, "manifest.json").read_bytes())
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise SnapshotError("Unsupported or incomplete snapshot manifest")
    expected_files = {"manifest.json"}
    names = set()
    total_bytes = 0
    for entry in manifest["files"]:
        relative = relative_path(entry["source_relative"])
        if relative.casefold() in names:
            raise SnapshotError(f"Duplicate manifest name: {relative}")
        names.add(relative.casefold())
        snapshot_relative = f"source/{relative}"
        if entry["snapshot_relative"] != snapshot_relative:
            raise SnapshotError(f"Invalid snapshot layout: {relative}")
        path = checked_path(destination, snapshot_relative)
        expected = entry["sha256"]
        for key in ("source_sha256_before", "copied_sha256", "destination_sha256_after",
                    "source_sha256_after"):
            if entry[key] != expected:
                raise SnapshotError(f"Inconsistent manifest hash evidence: {relative}")
        if not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256(path) != expected:
            raise SnapshotError(f"Snapshot file mismatch: {relative}")
        expected_files.add(snapshot_relative)
        total_bytes += entry["size_bytes"]
    if (manifest["file_count"] != len(names) or manifest["total_bytes"] != total_bytes
            or not names):
        raise SnapshotError("Manifest inventory totals differ")
    if destination_inventory(destination) != expected_files:
        raise SnapshotError("Snapshot contains missing or unmanifested files")
    return manifest


def freeze_snapshot(source: Path, destination: Path, selected: dict[str, str],
                    missing_optional: list[str] | None = None) -> dict:
    source, destination = validate_roots(source, destination)
    if not selected:
        raise SnapshotError("Empty snapshots are forbidden")
    canonical_names = [relative_path(name) for name in selected]
    if len({name.casefold() for name in canonical_names}) != len(selected):
        raise SnapshotError("Case-insensitive destination collision")
    before_git = git_provenance(source)
    entries = []
    for relative in sorted(selected):
        original = checked_path(source, relative)
        if not original.is_file():
            raise SnapshotError(f"Missing required file: {relative}")
        digest = sha256(original)
        entries.append({
            "source_relative": relative, "snapshot_relative": f"source/{relative}",
            "original_absolute": str(original), "role": selected[relative],
            "size_bytes": original.stat().st_size, "sha256": digest,
            "source_sha256_before": digest,
        })

    # Validate the entire existing destination before creating any files.
    allowed = {entry["snapshot_relative"] for entry in entries} | {"manifest.json"}
    if destination_inventory(destination) - allowed:
        raise SnapshotError("Destination contains files outside the selected snapshot")
    for entry in entries:
        path = checked_path(destination, entry["snapshot_relative"])
        if path.exists() and (not path.is_file() or sha256(path) != entry["sha256"]):
            raise SnapshotError(f"Existing destination differs: {path}")
    manifest_path = checked_path(destination, "manifest.json")
    if manifest_path.exists():
        old = verify_snapshot(destination)
        old_selection = {entry["source_relative"]: entry["role"] for entry in old["files"]}
        if (old["source_root"] != str(source) or old_selection != selected
                or old["source_git_before"]["revision"] != before_git["revision"]
                or old["missing_optional_probe_inputs"] != (missing_optional or [])):
            raise SnapshotError("Existing manifest provenance/selection differs")
        return old

    for entry in entries:
        original = checked_path(source, entry["source_relative"])
        copied = checked_path(destination, entry["snapshot_relative"])
        entry["copied_sha256"] = copy_verified(original, copied, entry["sha256"])
    for entry in entries:
        original = checked_path(source, entry["source_relative"])
        copied = checked_path(destination, entry["snapshot_relative"])
        entry["source_sha256_after"] = sha256(original)
        entry["destination_sha256_after"] = sha256(copied)
        if (entry["source_sha256_after"] != entry["sha256"]
                or entry["destination_sha256_after"] != entry["sha256"]
                or original.stat().st_size != entry["size_bytes"]
                or copied.stat().st_size != entry["size_bytes"]):
            raise SnapshotError(f"Source/destination changed during snapshot: {entry['source_relative']}")
    after_git = git_provenance(source)
    if (before_git["revision"], before_git["status_sha256"]) != (
            after_git["revision"], after_git["status_sha256"]):
        raise SnapshotError("Source Git revision/dirty state changed during snapshot")
    manifest = {
        "schema_version": 1, "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source), "snapshot_root": str(destination),
        "layout": "source/<original-relative-path>",
        "source_git_before": before_git,
        "source_git_after": {key: value for key, value in after_git.items()
                             if key != "status_porcelain_v1"},
        "source_git_unchanged": True,
        "baseline_entry": ENTRY,
        "baseline_arguments": ["--scenario", "sweet_170_incident_w", "--variant", "upper",
                               "--externality", "on"],
        "historical_runtime_relative": f"source/{HISTORICAL}",
        "file_count": len(entries), "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "role_counts": dict(sorted(Counter(entry["role"] for entry in entries).items())),
        "missing_optional_probe_inputs": missing_optional or [],
        "runtime_notes": {
            "simulation_executed": False, "runtime_imported": False,
            "original_absolute_json_provenance_preserved": True,
            "isolated_bootstrap_required": True,
            "dependencies_copied": False,
            "new_dependency_pins_user_provided": {"numpy": "2.3.5", "scipy": "1.16.3", "pytest": "8.4.2"},
            "old_dependency_metadata_user_provided": {"numpy": "2.5.3", "scipy": "1.18.1"},
            "historical_result_user_provided_ttt_veh_h": 5508.8644,
            "historical_result_valid_for_new_timing_or_acceptance": False,
        },
        "files": entries,
    }
    # The manifest is the completion marker. Failed copies never receive one.
    with manifest_path.open("xb") as stream:
        stream.write((json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return verify_snapshot(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_snapshot(args.destination)
    else:
        source, destination = validate_roots(args.source_root, args.destination)
        selected, missing = select_files(source)
        manifest = freeze_snapshot(source, destination, selected, missing)
    print(json.dumps({
        "status": "VERIFIED" if args.verify_only else "DONE",
        "destination": str(args.destination.resolve()), "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"], "role_counts": manifest["role_counts"],
        "source_revision": manifest["source_git_before"]["revision"],
        "source_dirty": manifest["source_git_before"]["dirty"],
        "missing_optional_probe_inputs": manifest["missing_optional_probe_inputs"],
        "manifest_sha256": sha256(args.destination / "manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
