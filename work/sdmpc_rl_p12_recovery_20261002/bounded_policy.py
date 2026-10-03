"""Two observation-only action bounds around the frozen P11 neural policy."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO / "work/sdmpc_rl_p11_20261002"))
from neural_policy import NeuralPolicy

FORMAT="sdmpc-p12-action-bounds-v1"
MODES=("nuf_nonnegative","fitting_bounds")


def project(action, obs, anchor, indices, step, mode):
    if mode not in MODES:
        raise ValueError("Unknown P12 bound")
    action=action.copy()
    if 16<=step<=30:
        action[1]=max(0.,float(action[1]))
        if mode=="fitting_bounds":
            gap=(anchor.astype(float)-obs[indices].astype(float))*[20.,10.]
            action[1]=min(action[1],max(0.,min(1.,gap[1])))
            if step<=25:
                action[0]=min(0.,float(action[0]))
            elif action[0]>0:
                action[0]=min(action[0],max(0.,min(1.,gap[0])))
    return action.astype(np.float32)


class BoundedPolicy:
    def __init__(self, base, mode):
        if mode not in MODES:
            raise ValueError("Unknown P12 bound")
        self.base,self.mode=base,mode

    @classmethod
    def load(cls,path,names):
        spec=json.loads(Path(path).read_text(encoding="utf-8"))
        if set(spec)!={"format","mode","base_spec_path","base_spec_sha256"} or spec["format"]!=FORMAT:
            raise ValueError("Invalid P12 policy spec")
        base_path=REPO / spec["base_spec_path"]
        if hashlib.sha256(base_path.read_bytes()).hexdigest()!=spec["base_spec_sha256"]:
            raise ValueError("Frozen P11 base spec changed")
        return cls(NeuralPolicy.load(base_path,names),spec["mode"])

    def memory(self):
        return self.base.memory()

    def act(self,obs):
        action=self.base.act(obs)
        return project(action,obs,self.base.anchor,self.base.anchors,self.base.previous_step,self.mode)
