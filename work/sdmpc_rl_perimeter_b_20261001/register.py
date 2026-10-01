"""Append-only pre-registration of a (spec, evaluator sources) pair before any canonical run.

Registering first makes the number of canonical evaluations explicit and prevents silently choosing
the best of many specs on the canonical (test) profiles. Entries are never edited or removed.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTRY = REPO / "results/sdmpc_rl_machine_b_20260930/canonical_registry.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--rationale", required=True, help="why this spec, from training-profile evidence only")
    p.add_argument("--validation", required=True, help="path(s)/summary of the training-profile validation")
    args = p.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    sources = {q.name: file_hash(q) for q in sorted(HERE.glob("*.py"))}
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else dict(entries=[])
    entry = dict(index=len(registry["entries"]), spec_sha256=digest(spec), spec_path=str(args.spec.resolve()),
                 evaluator_sources=sources, evaluator_sources_sha256=digest(sources),
                 registered_at=datetime.datetime.now().astimezone().isoformat(),
                 rationale=args.rationale, validation=args.validation)
    if any(e["spec_sha256"] == entry["spec_sha256"] and e["evaluator_sources_sha256"] == entry["evaluator_sources_sha256"]
           for e in registry["entries"]):
        raise SystemExit("already registered")
    registry["entries"].append(entry)
    tmp = REGISTRY.with_suffix(".json.tmp")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(registry, indent=1), encoding="utf-8")
    tmp.replace(REGISTRY)
    print(json.dumps(dict(index=entry["index"], spec_sha256=entry["spec_sha256"],
                          evaluator_sources_sha256=entry["evaluator_sources_sha256"]), indent=1))


if __name__ == "__main__":
    main()
