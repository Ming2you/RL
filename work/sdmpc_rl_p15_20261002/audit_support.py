"""Audit allowed training behavior coverage; no holdout or canonical examples."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/sdmpc_rl_p15_20261002"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "support_audit.json"
    if target.exists():
        raise FileExistsError("Preserve support audit")
    manifest_path = REPO / "work/sdmpc_rl_p13_20261002/fit_manifest.json"
    manifest = read(manifest_path)
    pins = {str(manifest_path.relative_to(REPO)): sha(manifest_path), **manifest["analyses"]}
    rows = list(manifest["rows"])
    for name, actor in (("sdmpc_rl_p13_20261002/pilot1", "p13_actor"),
                        ("sdmpc_rl_p14_20261002/wave1", "p14_actor")):
        wave = REPO / "results" / name
        assert read(wave / "completion.json")["status"] == "completed"
        analysis_path = wave / "analysis.json"
        pins[str(analysis_path.relative_to(REPO))] = sha(analysis_path)
        for rel, digest in read(wave / "plan.json")["sources"].items():
            assert sha(REPO / rel) == digest, rel
        for row in read(analysis_path)["rows"]:
            if actor == "p14_actor" and row["policy"] == "carry":
                continue
            slot = wave / f"{row['scenario']}_s{row['seed']}"
            tag = row["policy"]
            branch = slot / "branches" / ("k16_carry.json" if tag == "carry" else f"k16_{actor}_{tag}.json")
            rows.append(dict(row, experience_path=str((slot / "experience" / (tag+".npz")).relative_to(REPO)),
                             branch_path=str(branch.relative_to(REPO)), carry_path=str((slot / "carry.json").relative_to(REPO))))
    assert len(rows) == 80 and len({r["experience_path"] for r in rows}) == 80
    for row in rows:
        assert sha(REPO / row["experience_path"]) == row["experience_sha256"]
        assert sha(REPO / row["branch_path"]) == row["sha256"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["seed"])].append(row)
    assert len(groups) == 20 and all(len(g) == 4 for g in groups.values())
    reserved = {("sweet_155_w",8801),("sweet_170_w",8802),("sweet_170_incident_w",8803),
                ("sweet_170_skew15_w",8704),("sweet_190_w",8805)}
    assert not set(groups) & reserved
    summary = []
    for (scenario, seed), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: (r["ttt"],r["policy"] != "carry",r["policy"]))
        best = ordered[0]
        summary.append(dict(scenario=scenario, seed=seed, best_policy=best["policy"],
                            best_delta_pct=best["delta_pct"], deltas={r["policy"]:r["delta_pct"] for r in ordered}))
    result = dict(status="authenticated_training_support_audit", rows=rows, profiles=summary,
                  provenance=pins, trajectories=80, profile_count=20, duplicate_controls_excluded=True,
                  reserved_holdout_loaded=False, canonical_loaded=False)
    with target.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(status=result["status"],trajectories=80,profiles=20,
                         carry_best_155=sum(r["scenario"]=="sweet_155_w" and r["best_policy"]=="carry" for r in summary)),indent=2))


if __name__ == "__main__":
    main()
