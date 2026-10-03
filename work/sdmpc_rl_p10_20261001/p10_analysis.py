"""Authenticate P10 full continuations and training-only transition records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def validate_experience(arrays, meta, branch, carry):
    expected = dict(obs=(60, 2367), next_obs=(60, 2367), action=(60, 2),
                    reward=(60,), terminal=(60,), memory=(60, 4), next_memory=(60, 4))
    if set(arrays) != set(expected):
        raise ValueError("Experience array keys differ")
    for key, shape in expected.items():
        a = arrays[key]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("Experience shape/value differs: " + key)
    for key in ("obs", "next_obs", "action", "memory", "next_memory"):
        if arrays[key].dtype != np.float32:
            raise ValueError("Float32 experience required")
    if arrays["terminal"].dtype != np.bool_ or arrays["terminal"].tolist() != [False]*59 + [True]:
        raise ValueError("True terminal must appear only in the last transition")
    np.testing.assert_array_equal(arrays["next_obs"][:-1], arrays["obs"][1:])
    np.testing.assert_array_equal(arrays["next_memory"][:-1], arrays["memory"][1:])
    np.testing.assert_array_equal(arrays["next_obs"][-1], np.zeros(2367, np.float32))
    np.testing.assert_array_equal(arrays["action"], np.asarray([r["action"] for r in branch["rows"]], np.float32))
    np.testing.assert_allclose(arrays["reward"], -np.array([r["interval_ttt"] for r in branch["rows"]])/100., rtol=0, atol=1e-10)
    if not (meta["format"] == "sdmpc-training-continuation-p10-v1" and meta["source_kind"] == "training_only"
            and meta["scenario"] == carry["scenario"] and type(meta["training_seed"]) is int
            and meta["training_seed"] == carry["seed"] and meta["profile_sha256"] == carry["profile_sha256"]
            and meta["first_decision"] == 16 and meta["terminal_decision"] == 75
            and meta["transitions"] == 60 and meta["reward_divisor"] == 100. and meta["gamma"] == 1.
            and meta["ttt"] == branch["ttt"] and meta["option"] == branch["option"]
            and len(meta["observation_names"]) == 2367 and len(set(meta["observation_names"])) == 2367):
        raise ValueError("Training experience provenance differs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wave", type=Path, required=True)
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
        expected_labels = {"k16_carry"} | {"k16_p10_actor_" + s["tag"] for s in plan["options"]}
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
            xp = slot / "experience" / f"{tag}.npz"
            meta = json.loads(xp.with_suffix(".json").read_text())
            if hashlib.sha256(xp.read_bytes()).hexdigest() != meta["file_sha256"]:
                raise ValueError("Experience hash differs")
            with np.load(xp, allow_pickle=False) as arrays:
                validate_experience(arrays, meta, b, carry)
                if tag != "carry":
                    from retention_actor import RetentionActor
                    actor = RetentionActor(json.loads((repo / b["option"]["spec_path"]).read_text()), meta["observation_names"])
                    for obs, act, mem, next_mem in zip(arrays["obs"], arrays["action"], arrays["memory"], arrays["next_memory"]):
                        np.testing.assert_array_equal(actor.memory(), mem)
                        np.testing.assert_array_equal(actor.act(obs), act)
                        np.testing.assert_array_equal(actor.memory(), next_mem)
            table.append(dict(scenario=scenario, seed=int(seed), policy=tag, ttt=b["ttt"],
                carry_ttt=carry["ttt"], delta_pct=100.*(b["ttt"]-carry["ttt"])/carry["ttt"],
                fallbacks=sum(r["source"] == "reference_fallback" for r in rows),
                min_nuf=min(r["B_executed"][1] for r in rows), final_nuf=rows[-1]["B_executed"][1],
                max_urban_queue=max(r["urban_queue"] for r in rows),
                experience_sha256=meta["file_sha256"], sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    with open(wave / "analysis.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    groups = {}
    for row in table:
        groups.setdefault((row["scenario"], row["policy"]), []).append(row["delta_pct"])
    summary = [dict(scenario=s, policy=p, seeds=len(v), mean_delta_pct=float(np.mean(v)),
                    worst_delta_pct=max(v), collapses=sum(x >= 10. for x in v)) for (s,p),v in groups.items()]
    report = dict(status="authenticated_training_collection", rows=table, summary=summary,
                  transitions=60*len(table), canonical_eligible=False,
                  reason="Exploratory three-seed collection; five-seed admission and gain gates still required")
    (wave / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
