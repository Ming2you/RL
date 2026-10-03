"""Shared action bounds permit early NP relaxation and learned late return/hold."""
import hashlib
import json
from pathlib import Path
import numpy as np
from neural_policy import NeuralPolicy

REPO=Path(__file__).resolve().parents[2]
FORMAT="sdmpc-elite-bounds-p17-v2"


def project(action,obs,anchor,indices,step,nuf_bounds):
    action=action.copy()
    if step<16:return action
    gap=(anchor.astype(float)-obs[indices].astype(float))*[20.,10.]
    for i in (range(2) if step>30 else ()):
        action[i]=np.clip(action[i],max(-1.,min(0.,gap[i])),min(1.,max(0.,gap[i])))
    if step<=30:action[1]=np.clip(action[1],*nuf_bounds)
    if step<=30 and action[0]>0.:
        room=(float(anchor[0])+.1-float(obs[indices[0]]))*20.
        action[0]=min(float(action[0]),max(0.,min(1.,room)))
    return action.astype(np.float32)


class BoundedPolicy:
    def __init__(self,base,nuf_bounds):
        if len(nuf_bounds)!=2 or not -1.<=nuf_bounds[0]<=0.<=nuf_bounds[1]<=1.:
            raise ValueError("Invalid fitting NUF envelope")
        self.base,self.nuf_bounds=base,tuple(nuf_bounds)

    @classmethod
    def load(cls,path,names):
        spec=json.loads(Path(path).read_text(encoding="utf-8"))
        if (set(spec)!={"format","mode","base_spec_path","base_spec_sha256","nuf_window_action_bounds"}
                or spec["format"]!=FORMAT or spec["mode"]!="learned_return"):
            raise ValueError("Invalid P17 bound spec")
        base_path=REPO/spec["base_spec_path"]
        if hashlib.sha256(base_path.read_bytes()).hexdigest()!=spec["base_spec_sha256"]:
            raise ValueError("Frozen neural spec changed")
        return cls(NeuralPolicy.load(base_path,names),spec["nuf_window_action_bounds"])

    def memory(self):return self.base.memory()

    def act(self,obs):
        action=self.base.act(obs)
        return project(action,obs,self.base.anchor,self.base.anchors,self.base.previous_step,self.nuf_bounds)
