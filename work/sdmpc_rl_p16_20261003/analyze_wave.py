"""Authenticate P16 complete paired branches, then compare with frozen P13."""
import hashlib
import json
import os
from pathlib import Path
import sys

for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[k]="1"
import numpy as np
from direction_policy import DirectionPolicy
REPO=Path(__file__).resolve().parents[2]
WAVE=REPO / "results/sdmpc_rl_p16_20261003/wave1"
sys.path.insert(0,str(REPO / "work/sdmpc_rl_p10_20261001"))
from p10_analysis import validate_experience


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if (WAVE / "analysis.json").exists():raise FileExistsError("Preserve completed analysis")
    if read(WAVE / "completion.json")["status"]!="completed":raise RuntimeError("Wave incomplete")
    plan=read(WAVE / "plan.json");base=Path(plan["source_wave"])
    for rel,h in plan["sources"].items():
        if sha(REPO / rel)!=h:raise ValueError("Pinned source/model changed")
    if sha(base / "analysis.json")!=plan["source_analysis_sha256"]:raise ValueError("P13 analysis changed")
    for rel,h in plan["imported_controls"].items():
        if sha(WAVE / rel)!=h or sha(base / rel)!=h:raise ValueError("Imported control changed")
    for rel,h in plan["cache_hashes"].items():
        if sha(Path(plan["cache"]) / rel)!=h:raise ValueError("Carry cache changed")
    for rel,h in plan["decision_evidence"].items():
        if sha(REPO / rel)!=h:raise ValueError("P15 decision evidence changed")
    prior=read(base / "analysis.json");results=[]
    for scenario,seed in plan["jobs"]:
        slot=WAVE / f"{scenario}_s{seed}";carry=read(slot / "carry.json")
        cached=read(Path(plan["cache"]) / slot.name / "carry.json")
        if {k:v for k,v in carry.items() if k!="reused_from"}!=cached:
            raise ValueError("Carry metadata differs")
        if carry["truncated"] or len(carry["rows"])!=75:raise ValueError("Full carry required")
        expected={"k16_carry":{"name":"carry"},**{"k16_p16_actor_"+x["tag"]:x for x in plan["options"]}}
        if {p.stem for p in (slot / "branches").glob("*.json")}!=set(expected):raise ValueError("Branch set differs")
        baseline=next(r for r in prior["rows"] if r["scenario"]==scenario and r["seed"]==seed and r["policy"]=="rwbc_v2")
        for label,option in expected.items():
            file=slot / "branches" / (label+".json");b=read(file);rows=b["rows"]
            if b["option"]!=option or [r["control_step"] for r in rows]!=list(range(15,75)):
                raise ValueError("Branch identity/sequence differs")
            if (b["ttt"]!=rows[-1]["total_ttt"] or b["carry_ttt"]!=carry["ttt"]
                    or b["prefix_ttt"]!=carry["rows"][14]["total_ttt"]
                    or abs(b["prefix_ttt"]+sum(r["interval_ttt"] for r in rows)-b["ttt"])>1e-7):
                raise ValueError("TTT accounting differs")
            tag=option.get("tag","carry")
            if tag=="carry" and b["ttt"]!=carry["ttt"]:raise ValueError("Carry mismatch")
            xp=slot / "experience" / (tag+".npz");meta=read(xp.with_suffix(".json"))
            if sha(xp)!=meta["file_sha256"]:raise ValueError("Experience hash differs")
            with np.load(xp,allow_pickle=False) as a:
                validate_experience(a,meta,b,carry)
                if tag!="carry":
                    actor=DirectionPolicy.load(REPO / option["spec_path"],meta["observation_names"])
                    for obs,action,mem,next_mem in zip(a["obs"],a["action"],a["memory"],a["next_memory"]):
                        np.testing.assert_array_equal(actor.memory(),mem)
                        np.testing.assert_array_equal(actor.act(obs),action)
                        np.testing.assert_array_equal(actor.memory(),next_mem)
            results.append(dict(scenario=scenario,seed=seed,policy=tag,ttt=b["ttt"],carry_ttt=carry["ttt"],
                delta_pct=100*(b["ttt"]-carry["ttt"])/carry["ttt"],p13_delta_pct=baseline["delta_pct"],
                delta_vs_p13_ttt=b["ttt"]-baseline["ttt"],sha256=sha(file),experience_sha256=sha(xp),
                fallbacks=sum(r["source"]=="reference_fallback" for r in rows)))
    report=dict(status="authenticated_paired_np_direction_diagnostic",rows=results,
        model_sha256=plan["model_sha256"],new_optimizer_updates=0,new_transitions=600,reused_control_transitions=300,
        canonical_eligible=False,reason="One training seed per scenario, two shared direction variants; admission unchanged")
    with (WAVE / "analysis.json").open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
