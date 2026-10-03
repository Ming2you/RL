"""Shared neutral/positive NP exploration around the frozen P13 model."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p13_20261002"))
from bounded_policy import BoundedPolicy

FORMAT = "sdmpc-np-direction-p16-v1"
MODES = ("np_neutral", "np_relax_100")


class DirectionPolicy:
    def __init__(self, base, mode):
        if mode not in MODES:
            raise ValueError("Unknown NP direction mode")
        self.base, self.mode = base, mode

    @classmethod
    def load(cls, path, names):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if (set(spec) != {"format", "mode", "base_spec_path", "base_spec_sha256"}
                or spec["format"] != FORMAT):
            raise ValueError("Invalid P16 spec")
        base_path = REPO / spec["base_spec_path"]
        if hashlib.sha256(base_path.read_bytes()).hexdigest() != spec["base_spec_sha256"]:
            raise ValueError("Frozen P13 spec changed")
        return cls(BoundedPolicy.load(base_path, names), spec["mode"])

    def memory(self):
        return self.base.memory()

    def act(self, obs):
        action = self.base.act(obs)
        neural = self.base.base
        if 16 <= neural.previous_step <= 30:
            if self.mode == "np_neutral":
                action[0] = np.float32(0.)
            else:
                room = (float(neural.anchor[0]) + .1 - float(obs[neural.anchors[0]])) * 20.
                action[0] = np.float32(min(abs(float(action[0])), np.clip(room, 0., 1.)))
        return action
