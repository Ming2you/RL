"""Authenticate allowed complete returns, then select one trajectory per profile."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import numpy as np
REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ROOT=REPO / "results/sdmpc_rl_p17_v2_20261003"
FIT={"sweet_155_w":[8701,8501,9101,9201],"sweet_170_w":[8702,8502,9102,9202],
     "sweet_170_incident_w":[8703,8503,9103,9203],"sweet_170_skew15_w":[8804,8504,9104,9204],
     "sweet_190_w":[8705,8505,9105,9205]}
sys.path.insert(0,str(REPO / "work/sdmpc_rl_p10_20261001"))
from p10_analysis import validate_experience


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=HERE / "fit_manifest.json"
    if target.exists():raise FileExistsError("Preserve selection manifest")
    if any((p / "STOP").exists() for p in (REPO,ROOT,
        REPO / "results/sdmpc_rl_balanced_goal_20260930",REPO / "results/sdmpc_rl_machine_b_20260930")):
        raise RuntimeError("STOP")
    audit_path=REPO / "results/sdmpc_rl_p15_20261002/support_audit.json"
    audit=read(audit_path)
    assert audit["status"]=="authenticated_training_support_audit"
    rows=list(audit["rows"])
    analyses={**audit["provenance"],str(audit_path.relative_to(REPO)):sha(audit_path)}
    pins={}
    for name in ("sdmpc_rl_p13_20261002/pilot1","sdmpc_rl_p14_20261002/wave1",
                 "sdmpc_rl_p15_20261002/wave2_recovery","sdmpc_rl_p16_20261003/wave1"):
        wave=REPO / "results" / name
        assert read(wave / "completion.json")["status"]=="completed"
        plan=read(wave / "plan.json")
        for rel,h in plan["sources"].items():
            assert sha(REPO / rel)==h,rel
            if rel in pins:assert pins[rel]==h
            pins[rel]=h
        analyses[str((wave / "analysis.json").relative_to(REPO))]=sha(wave / "analysis.json")
        if "p15_" not in name and "p16_" not in name:continue
        actor="p15_actor" if "p15_" in name else "p16_actor"
        for row in read(wave / "analysis.json")["rows"]:
            if row["policy"]=="carry":continue
            slot=wave / f"{row['scenario']}_s{row['seed']}";tag=row["policy"]
            rows.append(dict(row,experience_path=str((slot / "experience" / (tag+".npz")).relative_to(REPO)),
                branch_path=str((slot / "branches" / (f"k16_{actor}_{tag}.json")).relative_to(REPO)),
                carry_path=str((slot / "carry.json").relative_to(REPO))))
    assert len(rows)==105 and len({r["experience_path"] for r in rows})==105
    allowed={(s,n) for s,seeds in FIT.items() for n in seeds}
    assert {(r["scenario"],r["seed"]) for r in rows}==allowed
    groups=defaultdict(list)
    for rel,h in analyses.items():assert sha(REPO / rel)==h,rel
    for row in rows:
        xp=REPO / row["experience_path"];branch=REPO / row["branch_path"];carry_path=REPO / row["carry_path"]
        assert sha(xp)==row["experience_sha256"] and sha(branch)==row["sha256"]
        meta=read(xp.with_suffix(".json"));b=read(branch);carry=read(carry_path)
        assert meta["source_kind"]=="training_only" and meta["training_seed"]==row["seed"]
        assert meta["scenario"]==row["scenario"] and b["ttt"]==row["ttt"]
        with np.load(xp,allow_pickle=False) as a:validate_experience(a,meta,b,carry)
        row["metadata_sha256"]=sha(xp.with_suffix(".json"));row["carry_sha256"]=sha(carry_path)
        groups[(row["scenario"],row["seed"])].append(row)
    selected=[]
    for profile,group in sorted(groups.items()):
        assert sum(r["policy"]=="carry" for r in group)==1
        minimum=min(r["ttt"] for r in group)
        ties=[r for r in group if abs(r["ttt"]-minimum)<=1e-9]
        best=min(ties,key=lambda r:(r["policy"]!="carry",r["policy"],r["experience_path"]))
        assert best["ttt"]<=next(r["ttt"] for r in group if r["policy"]=="carry")+1e-9
        selected.append(best)
    assert len(selected)==20
    nuf_bounds=[0.,0.]
    for row in selected:
        with np.load(REPO / row["experience_path"],allow_pickle=False) as a:
            nuf_bounds[0]=min(nuf_bounds[0],float(a["action"][:15,1].min()))
            nuf_bounds[1]=max(nuf_bounds[1],float(a["action"][:15,1].max()))
    report=dict(status="authenticated_elite_fitting_split",rows=selected,all_rows=rows,analyses=analyses,
        source_pins=pins,fitting_seeds=FIT,nuf_window_action_bounds=nuf_bounds,
        selection="minimum full TTT; ties within 1e-9 prefer carry, policy, path",
        trajectories_audited=105,selected_trajectories=20,selected_labels=1200,
        holdout_loaded=False,canonical_loaded=False,protocol_sha256=sha(ROOT / "protocol.md"))
    with target.open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(status=report["status"],selected=[{k:r[k] for k in ("scenario","seed","policy","delta_pct")} for r in selected]),indent=2))


if __name__=="__main__":main()
