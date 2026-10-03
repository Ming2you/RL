"""Verify the current numerical runtime reproduces a saved carry interval."""
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results/sdmpc_rl_p15_20261002"
BASE = REPO / "results/sdmpc_rl_p13_20261002/pilot1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = ROOT / "recovery_preflight_20261003.json"
    if target.exists():
        raise FileExistsError("Preserve recovery preflight")
    if any((p / "STOP").exists() for p in (REPO,ROOT,ROOT / "wave1",BASE,BASE.parent,
            REPO / "results/sdmpc_rl_balanced_goal_20260930",REPO / "results/sdmpc_rl_machine_b_20260930")):
        raise RuntimeError("STOP")
    audit = read(ROOT / "recovery_audit_20261003.json")
    assert audit["status"] == "authenticated_interrupted_p15"
    for rel,digest in read(ROOT / "wave1/plan.json")["sources"].items():
        assert sha(REPO / rel) == digest
    for path in HERE.glob("*.py"):
        compile(path.read_text(encoding="utf-8"),str(path),"exec")
    sys.path.insert(0,str(REPO / "work/sdmpc_rl_multi_20260929"))
    from budget_runtime import boot,DEFAULT_SNAPSHOT
    rt=boot(DEFAULT_SNAPSHOT,1)
    import numpy as np
    import scipy
    import torch
    from budget_env import BudgetEnv
    sys.path.insert(0,str(REPO / "work/sdmpc_rl_probe_b_20260930"))
    from probe import compact
    torch.set_num_threads(1)
    assert tuple(sys.version_info[:3]) == (3,12,14)
    assert (torch.__version__,np.__version__,scipy.__version__) == ("2.14.0+cpu","2.3.5","1.16.3")
    env=BudgetEnv(rt,scenario="sweet_190_w",training_seed=9205,guard_mode="physical")
    try:
        checkpoint=BASE / "cache/sweet_190_w_s9205/k16.pt"
        env.restore(torch.load(checkpoint,map_location="cpu",weights_only=False))
        _,reward,terminal,row=env.step(np.zeros(2,np.float32),"rl",0.,actor_cpu_seconds=0.)
        current=compact(row)
        expected=read(BASE / "sweet_190_w_s9205/branches/k16_carry.json")["rows"][0]
        compared=[k for k in expected if k != "decision_wall_seconds"]
        for key in compared:
            assert current[key] == expected[key], key
        assert not terminal and reward == -current["interval_ttt"]/100.
        rt["verify"]()
        result=dict(status="passed",scope="existing carry first interval reproduced exactly; completed branches separately authenticated",
            compared_fields=compared,checkpoint_sha256=sha(checkpoint),snapshot=rt["snapshot_identity"],
            python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,torch_threads=torch.get_num_threads(),
            original_preflight_sha256=sha(ROOT / "preflight.json"),audit_sha256=sha(ROOT / "recovery_audit_20261003.json"),
            sources={str(p.relative_to(REPO)):sha(p) for p in HERE.glob("*.py")})
        with target.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2)
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
