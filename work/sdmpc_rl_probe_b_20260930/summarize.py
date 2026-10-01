"""Read-only summary of probe outputs: per scenario/option full-episode TTT change vs the carry trajectory.

Usage: summarize.py <probe_root> [--json out.json]
Positive delta = worse than carry. Also reports the determinism control and interval-band deltas.
"""
import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    table = []
    for slot in sorted(x for x in args.root.iterdir() if x.is_dir()):
        carry_path = slot / "carry.json"
        if not carry_path.exists():
            print(f"{slot.name}: carry not finished")
            continue
        carry = load(carry_path)
        crow = carry["rows"]
        print(f"== {slot.name}: carry TTT {carry['ttt']:.3f}{' (TRUNCATED)' if carry.get('truncated') else ''}")
        for bpath in sorted((slot / "branches").glob("*.json")) if (slot / "branches").exists() else []:
            b = load(bpath)
            rows = b["rows"]
            k = b["branch_step"]
            # interval deltas aligned by control step (rows carry 0-based control_step)
            base = {r["control_step"]: r["interval_ttt"] for r in crow}
            diffs = {r["control_step"]: r["interval_ttt"] - base[r["control_step"]] for r in rows if r["control_step"] in base}
            first = next((c for c in sorted(diffs) if abs(diffs[c]) > 1e-12), None)
            bands = {name: sum(v for c, v in diffs.items() if lo <= c < hi)
                     for name, (lo, hi) in dict(b1_15=(0, 15), b16_25=(15, 25), b26_40=(25, 40), b41_55=(40, 55),
                                                b56_75=(55, 75)).items()}
            row = dict(slot=slot.name, label=b["label"], branch_step=k, ttt=b["ttt"], carry_ttt=carry["ttt"],
                       delta=b["ttt"] - carry["ttt"], delta_pct=100 * (b["ttt"] - carry["ttt"]) / carry["ttt"],
                       fallbacks=b["fallbacks"], carry_fallbacks=sum(r["source"] == "reference_fallback" for r in crow),
                       first_interval_difference=None if first is None else first + 1, bands=bands,
                       terminal_inventory=rows[-1]["inventory"], carry_terminal_inventory=crow[-1]["inventory"],
                       wall_seconds=b["wall_seconds"])
            table.append(row)
            print(f"   {b['label']:60s} dTTT {row['delta']:+9.3f} ({row['delta_pct']:+7.3f}%) fb {row['fallbacks']}"
                  f"/{row['carry_fallbacks']} first-diff {row['first_interval_difference']} "
                  + " ".join(f"{n}:{v:+.1f}" for n, v in bands.items()))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
