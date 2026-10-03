"""Pin P9 training comparisons and summarize the discovered P8 confound."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from actor import spec_sha256

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    base = json.loads((REPO / "work/sdmpc_rl_perimeter_b_20261001/specs/b1_exact_bind50_w16_25_return.json").read_text())
    options = []
    definitions = [("b1_np_only", 25, False, False), ("long_np_only", 27, False, False),
                   ("long_both_return", 27, True, False), ("long_both_fallback_return", 27, True, True)]
    (HERE / "specs").mkdir(exist_ok=True)
    for tag, end, nuf, abort in definitions:
        candidate_base = json.loads(json.dumps(base))
        candidate_base["params"]["bind_end"] = end
        spec = dict(format="sdmpc-recovery-actor-p9-v1", base=candidate_base,
                    restore_nuf=nuf, abort_on_fallback=abort)
        path = HERE / "specs" / f"{tag}.json"
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        options.append(dict(name="p9_actor", tag=tag, at=16, spec_path=str(path.relative_to(REPO)),
                            spec_sha256=spec_sha256(spec)))
    (HERE / "options.json").write_text(json.dumps(options, indent=2), encoding="utf-8")
    rows, metadata = [], []
    slots = [("sweet_190_w_s8705", "probe_p6"), ("sweet_170_skew15_w_s8804", "probe_p7")]
    for slot, prior in slots:
        for label, rel in [("old_B1_option", f"{prior}/{slot}/branches/k16_np_bind_afterreturn_np_delta50.0_window10.json"),
                           ("P8_long_actor", f"probe_p8/{slot}/branches/k16_actor_p8f_bind50_w16_27_return.json")]:
            path = args.data / rel
            data = json.loads(path.read_text())
            metadata.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 delta_pct=data["delta_pct"], ttt=data["ttt"]))
            for row in data["rows"]:
                rows.append(dict(slot=slot, policy=label, decision_step=row["control_step"]+1,
                    source=row["source"], NP=row["B_executed"][0], NUF=row["B_executed"][1],
                    urban_queue=row["urban_queue"], ramp_queue=row["ramp_queue"],
                    boundary_queue=row["boundary_queue"], total_ttt=row["total_ttt"]))
    out = REPO / "results/sdmpc_rl_p9_20261001/diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "paired_trace.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "inputs.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(dict(options=len(options), trace_rows=len(rows), output=str(out)), indent=2))


if __name__ == "__main__":
    main()
