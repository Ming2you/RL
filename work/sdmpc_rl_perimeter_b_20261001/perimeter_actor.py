"""Stateless structured budget actor: perimeter feedback on the NP budget (machine B).

The actor is a pure function of the 2367-dimensional coordination observation (no hidden memory,
no randomness, no simulator access), shared by all five scenarios. Its parameters, including any
static network layout it needs, live in one JSON spec whose SHA-256 is the model identity.

Rule (NP only; the NUF action is always 0, i.e. the NUF budget is carried):
    load      = "all":       total urban movement queue (veh), 1000 * sum(state/urban_movement_queue/*)
                "protected": protected accumulation N_P as defined by the frozen model
                             (State.protected_accumulation_veh): link in-transit occupancy
                             (capacity - available storage, off-ramp storage excluded) plus the
                             internal/boundary_out/off_ramp movement queues; boundary_in gate and
                             on-ramp access queues are excluded, so perimeter gating does not feed
                             its own trigger
    anchor    = carried NP budget (veh over H3)          = 1000 * obs[memory/action_anchor/0]
    achieved  = last executed minus last slack          = 1000 * (obs[previous_executed/0] - obs[previous_slack/0])
                (at control step 1, before any executed interval, achieved = anchor)
    step < bind_start:  pre_mode  "relax": target = max(anchor, achieved + margin) | "hold": target = anchor
    step > bind_end:    post_mode "relax": target = max(anchor, achieved + post_margin)
                                  "level": target = max(anchor, post_level)          | "hold": target = anchor
    inside the window:  if load >= on:  target = achieved - min(delta + gain * (load - on), delta_max)  (bind)
                        elif load <= off: target = max(anchor, achieved + margin)   (keep feasible)
                        else:             target = anchor                           (hold)
    action_NP = clip((target - anchor) / 50, -1, 1)
The rule is active for control steps start_step..end_step (from memory/remaining) and outputs zero
otherwise. The env projects the float32 action exactly as for any other policy.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCALE_NP = 50.
FORMAT = "sdmpc-perimeter-feedback-actor-v1"
REQUIRED = ("uq_on", "uq_off", "delta", "margin")
OPTIONAL = dict(gain=0., delta_max=None, start_step=1, end_step=75, bind_start=1, bind_end=75,
                queue_scope="all", layout=None, pre_mode="relax", post_mode="relax",
                post_margin=None, post_level=None)
MODES = ("relax", "hold", "level", "return")


def spec_sha256(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _number(value, label, minimum=None):
    if type(value) not in (int, float) or not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"Invalid actor parameter {label}: {value!r}")
    return float(value)


def _step(value, label):
    if type(value) is not int or not 1 <= value <= 75:
        raise ValueError(f"Invalid actor parameter {label}: {value!r}")
    return value


class PerimeterActor:
    def __init__(self, spec, names):
        if spec.get("format") != FORMAT or set(spec) != {"format", "params"}:
            raise ValueError("Unknown actor format or extra spec fields")
        params = spec["params"]
        unknown = set(params) - set(REQUIRED) - set(OPTIONAL)
        if unknown or any(k not in params for k in REQUIRED):
            raise ValueError("Actor parameters incomplete or unknown: " + str(sorted(unknown)))
        self.spec, self.sha256 = spec, spec_sha256(spec)
        p = dict(OPTIONAL, **params)
        self.uq_on, self.uq_off = _number(p["uq_on"], "uq_on"), _number(p["uq_off"], "uq_off")
        self.delta = _number(p["delta"], "delta", 0.)
        self.gain = _number(p["gain"], "gain", 0.)
        self.margin = _number(p["margin"], "margin", 0.)
        self.delta_max = None if p["delta_max"] is None else _number(p["delta_max"], "delta_max", 0.)
        self.start_step, self.end_step = _step(p["start_step"], "start_step"), _step(p["end_step"], "end_step")
        self.bind_start, self.bind_end = _step(p["bind_start"], "bind_start"), _step(p["bind_end"], "bind_end")
        self.pre_mode, self.post_mode = p["pre_mode"], p["post_mode"]
        if self.pre_mode not in ("relax", "hold") or self.post_mode not in MODES:
            raise ValueError("Invalid pre/post mode")
        self.post_margin = self.margin if p["post_margin"] is None else _number(p["post_margin"], "post_margin", 0.)
        if (self.post_mode == "level") != (p["post_level"] is not None):
            raise ValueError("post_level is required exactly for post_mode 'level'")
        self.post_level = None if p["post_level"] is None else _number(p["post_level"], "post_level")
        # "return" is the only mode with memory: the anchor seen at the first decision inside the
        # window is latched and targeted after the window. It is a deterministic function of the
        # observation history (a fresh actor per episode; replay of the same observations is exact).
        self.latched_anchor = None
        if not (self.uq_off <= self.uq_on and self.start_step <= self.end_step and self.bind_start <= self.bind_end):
            raise ValueError("Invalid actor parameters")
        names = list(names)
        if len(names) != 2367:
            raise ValueError("Observation contract must have 2367 entries")
        self.scope = p["queue_scope"]
        if self.scope == "all":
            if p["layout"] is not None:
                raise ValueError("Layout only applies to the protected scope")
            self.i_queue = [i for i, n in enumerate(names) if n.startswith("state/urban_movement_queue/")]
            if len(self.i_queue) != 78:
                raise ValueError("Unexpected urban movement queue layout")
            self.i_storage, self.capacity = [], np.zeros(0)
        elif self.scope == "protected":
            layout = p["layout"]
            if not isinstance(layout, dict) or set(layout) != {"queue_names", "link_capacity"}:
                raise ValueError("Protected scope requires a queue/link layout")
            self.i_queue = [names.index("state/urban_movement_queue/" + m) for m in layout["queue_names"]]
            links = sorted(layout["link_capacity"])
            self.i_storage = [names.index("state/urban_link_storage/" + link) for link in links]
            self.capacity = np.array([_number(layout["link_capacity"][link], "capacity", 0.) for link in links])
            if not self.i_queue or not self.i_storage:
                raise ValueError("Empty protected layout")
        else:
            raise ValueError("Unknown queue scope")
        self.i_anchor = names.index("memory/action_anchor/0")
        self.i_exec = names.index("memory/previous_executed/0")
        self.i_slack = names.index("memory/previous_slack/0")
        self.i_remaining = names.index("memory/remaining/0")

    @classmethod
    def load(cls, path, names):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), names)

    def control_step(self, obs):
        # memory/remaining = (80 - k) / 75 with k = control_step + 4 at decision time
        return int(round(76. - 75. * float(obs[self.i_remaining])))

    def load_veh(self, obs):
        queues = np.maximum(obs[self.i_queue].astype(np.float64), 0.)
        total = float(np.sum(queues))
        if self.scope == "protected":
            available = obs[self.i_storage].astype(np.float64) * 1000.
            total = 1000. * total + float(np.sum(np.maximum(self.capacity - available, 0.)))
            return total
        return 1000. * total

    def act(self, obs):
        obs = np.asarray(obs)
        if obs.dtype != np.float32 or obs.shape != (2367,) or not np.isfinite(obs).all():
            raise ValueError("Actor requires a finite float32 observation of length 2367")
        step = self.control_step(obs)
        if not self.start_step <= step <= self.end_step:
            return np.zeros(2, dtype=np.float32)
        load = self.load_veh(obs)
        anchor = 1000. * float(obs[self.i_anchor])
        # Before the first controlled interval there is no executed budget (the memory is zero):
        # the carried anchor is the PFO-achieved budget then, so use it as the achieved level.
        achieved = anchor if step == 1 else 1000. * (float(obs[self.i_exec]) - float(obs[self.i_slack]))
        if step < self.bind_start:          # before the binding window
            target = max(anchor, achieved + self.margin) if self.pre_mode == "relax" else anchor
        elif step > self.bind_end:          # after the binding window
            if self.post_mode == "return":  # steer back to the latched pre-window anchor
                target = anchor if self.latched_anchor is None else self.latched_anchor
            elif self.post_mode == "relax":
                target = max(anchor, achieved + self.post_margin)
            elif self.post_mode == "level":
                target = max(anchor, self.post_level)
            else:
                target = anchor
        else:                               # inside the window
            if self.latched_anchor is None:
                self.latched_anchor = anchor  # first decision inside the window
            if load >= self.uq_on:          # bind while loaded
                delta = self.delta + self.gain * (load - self.uq_on)
                if self.delta_max is not None:
                    delta = min(delta, self.delta_max)
                target = achieved - delta
            elif load <= self.uq_off:
                target = max(anchor, achieved + self.margin)
            else:
                target = anchor
        a = float(np.clip((target - anchor) / SCALE_NP, -1., 1.))
        return np.array([a, 0.], dtype=np.float32)
