"""Bind this process to one supervisor attempt, then delegate unchanged v2."""
import argparse
import os
from pathlib import Path
import sys
import time

import export_recovery as recovery


def runner_command(root):
    return [sys.executable, "-B", "-u", str(recovery.V2 / "run_wave.py"),
            "--output", str(root), "--resume"]


def read_attempt(lr, root, path):
    path = Path(path).resolve()
    attempt = lr.read(path)
    token = attempt["id"]
    expected = root / ".export-supervisor/attempts" / (token + ".json")
    if (len(token) != 32 or any(c not in "0123456789abcdef" for c in token) or path != expected or
            attempt["kind"] != "wave" or attempt["runner_command"] != runner_command(root) or
            attempt["command"] != [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--attempt", str(path)] or
            attempt["helper_sha256"] != recovery.helper_hashes(lr) or
            attempt["source"] != lr.read(root / "plan.json")["identity"]):
        raise ValueError("Coordinator bootstrap attempt/provenance differs")
    return attempt


def check_binding(lr, attempt, binding, live=False):
    from launch_identity import verify_claim
    if (binding["attempt_id"] != attempt["id"] or binding["owner"] != attempt["owner"] or
            binding["launcher"] != attempt["process"] or
            binding["runner_command"] != attempt["runner_command"] or
            binding["helper_sha256"] != attempt["helper_sha256"] or
            binding["actual"]["command"] != attempt["command"][3:]):
        raise ValueError("Coordinator binding identity/provenance differs")
    if live:
        mode = verify_claim(binding["launcher"], binding["actual"], binding["owner"],
                            binding["parent_pid"], probe=lr.probe_process)
        if mode != binding["launch_mode"]:
            raise ValueError("Coordinator binding launch mode differs")


def register(lr, wave, root, path, actual, parent_pid):
    from launch_identity import verify_claim
    recovery.check_stops(lr, wave, root)
    attempt = read_attempt(lr, root, path)
    if recovery.authenticate(lr, wave, root) != attempt["source"]:
        raise ValueError("Coordinator source changed")
    mode = verify_claim(attempt["process"], actual, attempt["owner"], parent_pid, probe=lr.probe_process)
    binding = dict(attempt_id=attempt["id"], owner=attempt["owner"], launcher=attempt["process"],
        actual=actual, parent_pid=parent_pid, launch_mode=mode,
        runner_command=attempt["runner_command"], helper_sha256=attempt["helper_sha256"])
    check_binding(lr, attempt, binding, live=True)
    recovery.publish_bytes(Path(path).with_suffix(".coordinator.json"), recovery.json_bytes(binding))
    return binding


def acknowledged(lr, root, path, binding_hash):
    attempt = read_attempt(lr, root, path)
    state = lr.read(root / ".export-supervisor/state.json")
    return (attempt.get("binding_sha256") == binding_hash and
            [a for a in state["attempts"] if a["id"] == attempt["id"]] == [attempt])


def prove_bound_dead(lr, root, process):
    """Use the live-verified, acknowledged binding even when no workers started."""
    state_file = root / ".export-supervisor/state.json"
    if not state_file.exists():
        return False
    state = lr.read(state_file)
    matches = [a for a in state["attempts"] if a.get("coordinator_process") == process]
    if not matches:
        return False
    if len(matches) != 1:
        raise ValueError("Ambiguous coordinator attempt binding")
    saved = matches[0]
    path = root / ".export-supervisor/attempts" / (saved["id"] + ".json")
    attempt = read_attempt(lr, root, path)
    binding_file = path.with_suffix(".coordinator.json")
    binding = lr.read(binding_file)
    if (attempt != saved or attempt["binding_sha256"] != lr.file_hash(binding_file) or
            binding["actual"] != attempt["coordinator"] or process["pid"] != binding["actual"]["pid"] or
            process["command"] != attempt["runner_command"][3:] or process["started"] < attempt["started"]):
        raise ValueError("Coordinator binding/hash/attempt differs")
    check_binding(lr, attempt, binding)
    recovery.require_dead(lr, binding["launcher"])
    recovery.require_dead(lr, binding["actual"])
    return True


def run(path):
    lr, _, _, wave = recovery.modules()
    root = lr.canonical_root(lr.COHORT_ROOT)
    # The supervisor publishes its launcher identity before the child can claim it.
    for _ in range(201):
        recovery.check_stops(lr, wave, root)
        if read_attempt(lr, root, path).get("process") is not None:
            break
        time.sleep(.05)
    else:
        raise OSError("Coordinator launcher identity UNKNOWN")
    register(lr, wave, root, path, lr.process_identity(command=sys.argv), os.getppid())
    binding_hash = lr.file_hash(Path(path).with_suffix(".coordinator.json"))
    for _ in range(201):
        recovery.check_stops(lr, wave, root)
        if acknowledged(lr, root, path, binding_hash):
            attempt = read_attempt(lr, root, path)
            lr.verify_identity(attempt["source"])
            previous = sys.argv
            try:
                sys.argv = attempt["runner_command"][3:]
                return wave.main(sys.argv[1:])
            finally:
                sys.argv = previous
        time.sleep(.05)
    raise OSError("Coordinator identity acknowledgement UNKNOWN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().attempt))
