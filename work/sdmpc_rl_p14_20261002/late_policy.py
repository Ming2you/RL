"""P14 paired changes only to P13 decisions 26-30; shared across profiles."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p13_20261002"))
from bounded_policy import BoundedPolicy

FORMAT = "sdmpc-late-window-p14-v1"
MODES = ("return_both_w25", "hold_np_after25")


class LatePolicy:
    def __init__(self, base, mode):
        if mode not in MODES:
            raise ValueError("Unknown late-window policy")
        self.base, self.mode = base, mode

    @classmethod
    def load(cls, path, names):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if (set(spec) != {"format", "mode", "base_spec_path", "base_spec_sha256"}
                or spec["format"] != FORMAT):
            raise ValueError("Invalid P14 spec")
        base_path = REPO / spec["base_spec_path"]
        if hashlib.sha256(base_path.read_bytes()).hexdigest() != spec["base_spec_sha256"]:
            raise ValueError("Frozen P13 spec changed")
        return cls(BoundedPolicy.load(base_path, names), spec["mode"])

    def memory(self):
        return self.base.memory()

    def act(self, obs):
        action = self.base.act(obs)
        neural = self.base.base
        if 26 <= neural.previous_step <= 30:
            if self.mode == "return_both_w25":
                action = np.clip((neural.anchor.astype(float) - obs[neural.anchors].astype(float))
                                 * [20., 10.], -1., 1.).astype(np.float32)
            else:
                action[0] = max(0., float(action[0]))
        return action
