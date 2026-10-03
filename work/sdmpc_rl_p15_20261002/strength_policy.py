"""P15 factorial action changes around the frozen shared P13 neural model."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p13_20261002"))
from bounded_policy import BoundedPolicy

FORMAT = "sdmpc-strength-recovery-p15-v1"
MODES = ("half_np", "restore_nuf", "half_np_restore_nuf")


class StrengthPolicy:
    def __init__(self, base, mode):
        if mode not in MODES:
            raise ValueError("Unknown strength/recovery policy")
        self.base, self.mode = base, mode

    @classmethod
    def load(cls, path, names):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        if (set(spec) != {"format", "mode", "base_spec_path", "base_spec_sha256"}
                or spec["format"] != FORMAT):
            raise ValueError("Invalid P15 spec")
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
            if self.mode in ("half_np", "half_np_restore_nuf") and action[0] < 0.:
                action[0] *= np.float32(.5)
            if self.mode in ("restore_nuf", "half_np_restore_nuf"):
                gap = (float(neural.anchor[1]) - float(obs[neural.anchors[1]])) * 10.
                action[1] = np.float32(np.clip(gap, 0., 1.))
        return action
