"""Authenticate completed P9 training branches; never score smoke or partial runs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wave", type=Path, required=True)
    p.add_argument("--historical", type=Path, required=True)
    args = p.parse_args()
    wave = args.wave.resolve()
    plan = json.loads((wave / "plan.json").read_text(encoding="utf-8"))
    done = wave / "completion.json"
    if not done.is_file() or json.loads(done.read_text())["status"] != "completed":
        raise RuntimeError("Wave not completed; inspect logs without scoring it")
    repo = Path(__file__).resolve().parents[2]
    for rel, expected in plan["sources"].items():
        if hashlib.sha256((repo / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Source changed: " + rel)
    table = []
    for scenario, seed in plan["jobs"]:
        slot = wave / f"{scenario}_s{seed}"
        carry = json.loads((slot / "carry.json").read_text())
        if carry["truncated"] or len(carry["rows"]) != 75:
            raise RuntimeError("Incomplete cached carry")
        files = sorted((slot / "branches").glob("*.json"))
        expected_labels = {"k16_carry"} | {"k16_p9_actor_" + s["tag"] for s in plan["options"]}
        if {f.stem for f in files} != expected_labels:
            raise RuntimeError("Missing or extra branches")
        for file in files:
            b = json.loads(file.read_text())
            rows = b["rows"]
            if [r["control_step"] for r in rows] != list(range(15, 75)):
                raise RuntimeError("Incomplete or nonsequential branch")
            if b["ttt"] != rows[-1]["total_ttt"] or b["carry_ttt"] != carry["ttt"]:
                raise RuntimeError("TTT identity mismatch")
            if abs(b["prefix_ttt"] + sum(r["interval_ttt"] for r in rows) - b["ttt"]) > 1e-7:
                raise RuntimeError("TTT timeline mismatch")
            if b["label"] == "k16_carry" and b["ttt"] != carry["ttt"]:
                raise RuntimeError("Carry reproduction failed")
            tag = b["option"].get("tag", "carry")
            old_ttt = None
            if tag == "long_np_only":
                old = args.historical / f"probe_p8/{scenario}_s{seed}/branches/k16_actor_p8f_bind50_w16_27_return.json"
                if old.exists():
                    old_ttt = json.loads(old.read_text())["ttt"]
                    if b["ttt"] != old_ttt:
                        raise RuntimeError("Legacy P8 reproduction failed: " + scenario)
            table.append(dict(scenario=scenario, seed=int(seed), policy=tag, ttt=b["ttt"],
                carry_ttt=carry["ttt"], delta_pct=100.*(b["ttt"]-carry["ttt"])/carry["ttt"],
                fallbacks=sum(r["source"] == "reference_fallback" for r in rows),
                min_nuf=min(r["B_executed"][1] for r in rows), final_nuf=rows[-1]["B_executed"][1],
                max_urban_queue=max(r["urban_queue"] for r in rows),
                historical_ttt=old_ttt, sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    with open(wave / "analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    report = dict(status="authenticated_training_diagnostic", rows=table,
                  canonical_eligible=False, reason="Only one seed per scenario; five-seed admission still required")
    (wave / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
