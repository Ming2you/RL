"""Observation-only training candidates; preserve the registered P8 implementation."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "work/sdmpc_rl_perimeter_b_20261001"))
from perimeter_actor import PerimeterActor, spec_sha256


class RecoveryActor(PerimeterActor):
    """Latch both anchors; optionally end binding after its first observed fallback.

    This changes the proposed action only. The frozen physical feasibility guard,
    solver, action projection and scoring are untouched. No simulator access.
    """

    def __init__(self, spec, names):
        if set(spec) != {"format", "base", "restore_nuf", "abort_on_fallback"}:
            raise ValueError("Invalid recovery spec keys")
        if spec["format"] != "sdmpc-recovery-actor-p9-v1":
            raise ValueError("Invalid recovery spec format")
        if any(type(spec[k]) is not bool for k in ("restore_nuf", "abort_on_fallback")):
            raise ValueError("Recovery switches must be booleans")
        super().__init__(spec["base"], names)
        if self.pre_mode != "hold" or self.post_mode != "return":
            raise ValueError("P9 requires hold then latched return")
        self.restore_nuf = spec["restore_nuf"]
        self.abort_on_fallback = spec["abort_on_fallback"]
        self.i_nuf = list(names).index("memory/action_anchor/1")
        self.i_fallback = list(names).index("memory/fallback/0")
        self.nuf_anchor = None
        self.aborted = False
        self.previous_step = None
        self.spec, self.sha256 = spec, spec_sha256(spec)

    @classmethod
    def load(cls, path, names):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), names)

    def act(self, obs):
        obs = np.asarray(obs)
        if obs.dtype != np.float32 or obs.shape != (2367,) or not np.isfinite(obs).all():
            raise ValueError("Finite float32 observation required")
        step = self.control_step(obs)
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("Use a fresh actor per episode and sequential observations")
        inside = self.bind_start <= step <= self.bind_end
        if inside and self.nuf_anchor is None:
            self.nuf_anchor = 10000. * float(obs[self.i_nuf])
        # The fallback before the binding window was caused by carry, not this policy.
        if (self.abort_on_fallback and inside and self.previous_step is not None
                and self.previous_step >= self.bind_start and obs[self.i_fallback] > .5):
            self.aborted = True
        action = super().act(obs)
        returning = self.aborted or step > self.bind_end
        if self.start_step <= step <= self.end_step and returning:
            if self.latched_anchor is not None:
                action[0] = np.clip((self.latched_anchor - 1000. * float(obs[self.i_anchor])) / 50., -1., 1.)
            if self.restore_nuf and self.nuf_anchor is not None:
                action[1] = np.clip((self.nuf_anchor - 10000. * float(obs[self.i_nuf])) / 1000., -1., 1.)
        self.previous_step = step
        return action
