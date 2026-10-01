"""Read-only aggregation of every probe branch (training profiles) by option label and scenario.

For each option label: per-scenario list of full-run TTT changes vs its own carry (percent; negative
is better), the mean, and a rough all-five pass probability  prod_s Phi(-mean_s / sigma)  with a
pooled per-run spread sigma (default 3.5%, the order of the measured seed-to-seed/noise spread).
This is a screening heuristic on training profiles, not a canonical result.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SCEN = ("sweet_155_w", "sweet_170_w", "sweet_170_incident_w", "sweet_170_skew15_w", "sweet_190_w")
SHORT = dict(zip(SCEN, ("155", "170", "inc", "skew", "190")))


def phi(x):
    return 0.5 * (1. + math.erf(x / math.sqrt(2.)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(r"C:\Users\alsrj\Desktop\RL\results\sdmpc_rl_machine_b_20260930"))
    p.add_argument("--sigma", type=float, default=3.5)
    p.add_argument("--min-scenarios", type=int, default=3)
    args = p.parse_args()
    table = defaultdict(lambda: defaultdict(list))
    for probe in sorted(args.root.glob("probe_p*")):
        for slot in sorted(x for x in probe.iterdir() if x.is_dir()):
            scen = next((s for s in SCEN if slot.name.startswith(s + "_s")), None)
            if scen is None or not (slot / "branches").exists():
                continue
            for f in sorted((slot / "branches").glob("*.json")):
                b = json.loads(f.read_text(encoding="utf-8"))
                if b["option"]["name"] == "carry" or len(b["rows"]) + b["branch_step"] - 1 != 75:
                    continue
                table[b["label"]][scen].append((round(b["delta_pct"], 2), slot.name.split("_s")[-1], probe.name))
    rows = []
    for label, per in table.items():
        if len(per) < args.min_scenarios:
            continue
        means = {s: sum(v for v, *_ in per[s]) / len(per[s]) for s in per}
        prob = math.prod(phi(-means[s] / args.sigma) for s in SCEN) if len(per) == 5 else None
        rows.append((prob if prob is not None else -1, label, per, means))
    for prob, label, per, means in sorted(rows, key=lambda r: -r[0]):
        cells = "  ".join(f"{SHORT[s]}:{means[s]:+5.1f}[{','.join(f'{v:+.1f}' for v, *_ in per[s])}]"
                          if s in per else f"{SHORT[s]}:  n/a" for s in SCEN)
        print(f"{'P(all5)=%.2f' % prob if prob >= 0 else 'P(all5)= n/a'}  {label[:58]:58s} {cells}")


if __name__ == "__main__":
    main()
