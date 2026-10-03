"""Authenticate the complete P17 prospective pilot before interpreting scores."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import numpy as np
from bounded_policy import BoundedPolicy as NeuralPolicy
REPO=Path(__file__).resolve().parents[2]
WAVE=REPO / "results/sdmpc_rl_p17_20261003/pilot1"
sys.path.insert(0,str(REPO / "work/sdmpc_rl_p10_20261001"))
from p10_analysis import validate_experience


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if read(WAVE / "completion.json")["status"]!="completed":
        raise RuntimeError("Full pilot not completed")
    if (WAVE / "analysis.json").exists():raise FileExistsError("Preserve completed analysis")
    plan=read(WAVE / "plan.json")
    if sha(WAVE.parent / "protocol.md")!=plan["protocol_sha256"]:raise ValueError("Protocol changed")
    if sha(WAVE.parent / "preflight.json")!=plan["preflight_sha256"]:raise ValueError("Preflight changed")
    if sha(WAVE.parent / "fit_v1/completion.json")!=plan["fit_completion_sha256"]:raise ValueError("Fit receipt changed")
    for rel,digest in plan["sources"].items():
        if sha(REPO / rel)!=digest: raise ValueError("Pinned source/model changed")
    results=[]
    for scenario,seed in plan["jobs"]:
        slot=WAVE / f"{scenario}_s{seed}"; carry=read(slot / "carry.json")
        if (carry["scenario"]!=scenario or carry["seed"]!=seed or carry["truncated"]
                or len(carry["rows"])!=75 or [r["control_step"] for r in carry["rows"]]!=list(range(75))
                or carry["ttt"]!=carry["rows"][-1]["total_ttt"]):
            raise ValueError("Carry coverage/provenance mismatch")
        labels={"k16_carry","k16_p17_actor_elite_v1"}
        if {p.stem for p in (slot / "branches").glob("*.json")}!=labels:
            raise ValueError("Missing or extra pilot branch")
        for label in sorted(labels):
            file=slot / "branches" / (label+".json"); b=read(file); rows=b["rows"]
            if ([r["control_step"] for r in rows]!=list(range(15,75)) or b["ttt"]!=rows[-1]["total_ttt"]
                    or b["carry_ttt"]!=carry["ttt"] or b["prefix_ttt"]!=carry["rows"][14]["total_ttt"]
                    or abs(b["prefix_ttt"]+sum(r["interval_ttt"] for r in rows)-b["ttt"])>1e-7):
                raise ValueError("Branch TTT accounting mismatch")
            tag=b["option"].get("tag","carry")
            if tag=="carry" and b["ttt"]!=carry["ttt"]: raise ValueError("Carry reproduction failed")
            if tag!="carry" and b["option"]!=plan["options"][0]: raise ValueError("Policy identity mismatch")
            xp=slot / "experience" / (tag+".npz"); meta=read(xp.with_suffix(".json"))
            if sha(xp)!=meta["file_sha256"]: raise ValueError("Experience hash mismatch")
            with np.load(xp,allow_pickle=False) as a:
                validate_experience(a,meta,b,carry)
                if tag!="carry":
                    actor=NeuralPolicy.load(REPO / b["option"]["spec_path"],meta["observation_names"])
                    for obs,action,mem,next_mem in zip(a["obs"],a["action"],a["memory"],a["next_memory"]):
                        np.testing.assert_array_equal(actor.memory(),mem)
                        np.testing.assert_array_equal(actor.act(obs),action)
                        np.testing.assert_array_equal(actor.memory(),next_mem)
            results.append(dict(scenario=scenario,seed=seed,policy=tag,ttt=b["ttt"],carry_ttt=carry["ttt"],
                delta_pct=100*(b["ttt"]-carry["ttt"])/carry["ttt"],sha256=sha(file),experience_sha256=sha(xp),
                fallbacks=sum(r["source"]=="reference_fallback" for r in rows)))
    report=dict(status="authenticated_neural_training_pilot",rows=results,transitions=600,
        model_sha256=plan["model_sha256"],canonical_eligible=False,
        reason="One fresh training seed per scenario; five-seed admission and gain gates unchanged")
    with (WAVE / "analysis.json").open("x",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
