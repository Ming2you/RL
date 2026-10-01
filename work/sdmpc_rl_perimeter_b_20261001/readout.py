"""Read-only all-five readout of one canonical run root (never runs physics).

PASS requires all five scenarios completed on machine B with the same actor spec, the same
evaluator sources and matching output hashes, each improving on its machine-B center by more than
max(1e-6, 1e-8 * center). Passing only makes the spec eligible for a separate reproduction run.
"""
import argparse
import hashlib
import json
from pathlib import Path

SCENARIOS = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    rows, identities, problems = [], set(), []
    for s in SCENARIOS:
        folder = args.root / s
        try:
            done = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if done["status"] != "completed" or done["format"] != summary["format"] or summary["format"].endswith("SMOKE-NOT-CANONICAL"):
                raise ValueError("not a completed canonical slot")
            for name, sha in done["outputs_sha256"].items():
                if file_hash(folder / name) != sha:
                    raise ValueError("output hash differs: " + name)
            identities.add((summary["actor_spec_sha256"], summary["evaluator_sources_sha256"],
                            json.dumps(summary["machine_fingerprint"], sort_keys=True)))
            rows.append(dict(scenario=s, ttt=summary["ttt"], center_ttt=summary["center_ttt"],
                             improvement=summary["improvement"], improvement_pct=summary["improvement_pct"],
                             passes=summary["passes"], fallbacks=len(summary["fallback_steps"]),
                             terminal_inventory=summary["terminal_inventory"]))
        except Exception as exc:  # keep every row, including absent or invalid ones
            problems.append(f"{s}: {type(exc).__name__}: {exc}")
            rows.append(dict(scenario=s, status="absent_or_invalid"))
    if len(identities) > 1:
        problems.append("scenarios differ in spec, evaluator sources or machine")
    complete = not problems and len(rows) == 5
    passed = complete and all(r["passes"] for r in rows)
    result = dict(root=str(args.root), status="eligible_for_separate_reproduction" if passed else
                  ("failed_acceptance" if complete else "incomplete_or_invalid"),
                  rows=rows, problems=problems, identity=sorted(identities)[0] if len(identities) == 1 else None,
                  goal_achieved=False)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
