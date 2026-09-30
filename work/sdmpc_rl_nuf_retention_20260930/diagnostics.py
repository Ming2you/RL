"""Strict diagnostic-only JSON boundary within the original worker timing scope."""
import copy
import hashlib
import importlib.util
import json
from functools import lru_cache
from references import RECOVERY, HELPER_PINS, check_hash

RECEIPT = "diagnostic-audit.json"


@lru_cache(maxsize=1)
def tagger():
    path = RECOVERY / "export_recovery.py"
    check_hash(path, HELPER_PINS[path.name])
    spec = importlib.util.spec_from_file_location("pinned_diagnostic_tag_function", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.tag_diagnostics


def raw_digest(trace):
    return hashlib.sha256(json.dumps(trace, sort_keys=True, allow_nan=True).encode()).hexdigest()


def export_trace(output, trace, payload):
    from local_runtime import save, file_hash, digest
    # tagger admits only the frozen +Inf paths; digest rejects nonfinite experience.
    experience_digest = digest(payload)
    tagged, changes = tagger()(trace)
    save(output / "trace.json", tagged)
    audit = dict(format="nuf-retention-diagnostic-audit-v1", run_id=payload["settings"]["run_id"],
        helper_sha256=HELPER_PINS["export_recovery.py"], changed_paths=changes,
        checkpoint_sha256=file_hash(output / "checkpoint.pt"),
        trace_sha256=file_hash(output / "trace.json"), raw_trace_digest=raw_digest(trace),
        experience_sha256=file_hash(output / "experience.pt"), experience_digest=experience_digest,
        settings_digest=digest(payload["settings"]), core_experience_unchanged=True,
        convergence_claim=False, physical_or_training_change=False)
    save(output / RECEIPT, audit)


def validate_audit(folder, trace, payload):
    from local_runtime import read, file_hash, digest
    audit = read(folder / RECEIPT)
    raw = copy.deepcopy(trace)
    def untag(value):
        if isinstance(value, dict):
            if "nonfinite_float" in value:
                if value != {"nonfinite_float": "+inf"}:
                    raise ValueError("Unknown diagnostic tag")
                return float("inf")
            return {k: untag(v) for k, v in value.items()}
        if isinstance(value, list):
            return [untag(v) for v in value]
        return value
    raw = untag(raw)
    tagged, changes = tagger()(raw)
    expected = dict(format="nuf-retention-diagnostic-audit-v1", run_id=payload["settings"]["run_id"],
        helper_sha256=HELPER_PINS["export_recovery.py"], changed_paths=changes,
        checkpoint_sha256=file_hash(folder / "checkpoint.pt"), trace_sha256=file_hash(folder / "trace.json"),
        raw_trace_digest=raw_digest(raw), experience_sha256=file_hash(folder / "experience.pt"),
        experience_digest=digest(payload), settings_digest=digest(payload["settings"]),
        core_experience_unchanged=True, convergence_claim=False, physical_or_training_change=False)
    if trace != tagged or audit != expected:
        raise ValueError("Diagnostic audit/experience reconciliation mismatch")
    return audit
