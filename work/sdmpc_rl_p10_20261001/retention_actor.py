"""P10 observation/history policy: isolate NUF retention from NP window length."""
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_p9_20261001"))
from actor import RecoveryActor, spec_sha256


class RetentionActor:
    def __init__(self, spec, names):
        if set(spec) != {"format", "recovery", "retain_nuf_during_bind"}:
            raise ValueError("Unexpected P10 spec keys")
        if spec["format"] != "sdmpc-retention-p10-v1" or type(spec["retain_nuf_during_bind"]) is not bool:
            raise ValueError("Invalid P10 spec")
        self.policy = RecoveryActor(spec["recovery"], names)
        if not self.policy.restore_nuf or self.policy.abort_on_fallback:
            raise ValueError("P10 requires both-budget return without fallback abort")
        self.retain = spec["retain_nuf_during_bind"]
        self.spec, self.sha256 = spec, spec_sha256(spec)

    @classmethod
    def load(cls, path, names):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), names)

    def memory(self):
        p = self.policy
        return np.array([float(p.latched_anchor is not None),
                         0. if p.latched_anchor is None else p.latched_anchor / 1000.,
                         0. if p.nuf_anchor is None else p.nuf_anchor / 10000.,
                         float(p.aborted)], dtype=np.float32)

    def act(self, obs):
        action = self.policy.act(obs)
        p = self.policy
        step = p.control_step(obs)
        if self.retain and p.start_step <= step <= p.end_step and p.bind_start <= step <= p.bind_end:
            action[1] = np.clip((p.nuf_anchor - 10000. * float(obs[p.i_nuf])) / 1000., -1., 1.)
        return action
