"""Shared return-weighted cloned policy; observation/history inputs only."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
FORMAT = "sdmpc-rwbc-p11-v1"


def network(inputs=2369):
    net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64),
                        nn.Tanh(), nn.Linear(64, 2), nn.Tanh())
    nn.init.zeros_(net[-2].weight)
    nn.init.zeros_(net[-2].bias)
    return net


class NeuralPolicy:
    def __init__(self, payload, names):
        if payload["format"] != FORMAT or list(names) != payload["observation_names"]:
            raise ValueError("Neural policy observation contract mismatch")
        if len(names) != 2367 or len(set(names)) != 2367:
            raise ValueError("Unexpected observation schema")
        torch.set_num_threads(1)
        self.names = list(names)
        self.anchors = [self.names.index("memory/action_anchor/0"),
                        self.names.index("memory/action_anchor/1")]
        self.remaining = self.names.index("memory/remaining/0")
        self.mean = payload["mean"].detach().clone()
        self.scale = payload["scale"].detach().clone()
        if (self.mean.shape != (2369,) or self.scale.shape != (2369,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or (self.scale < .05).any()):
            raise ValueError("Invalid fitting normalization")
        self.net = network().eval()
        self.net.load_state_dict(payload["state_dict"], strict=True)
        if any(not torch.isfinite(v).all() for v in self.net.parameters()):
            raise ValueError("Nonfinite policy weights")
        self.net.requires_grad_(False)
        self.anchor = None
        self.previous_step = None

    @classmethod
    def load(cls, spec_path, names):
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        if set(spec) != {"format", "model_path", "model_sha256", "window", "after"}:
            raise ValueError("Unknown policy spec fields")
        if spec["format"] != FORMAT or spec["window"] != [16, 30] or spec["after"] != "return_both":
            raise ValueError("Policy scope mismatch")
        path = REPO / spec["model_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["model_sha256"]:
            raise ValueError("Frozen neural model hash mismatch")
        return cls(torch.load(path, map_location="cpu", weights_only=True), names)

    def memory(self):
        return np.array([float(self.anchor is not None),
            0. if self.anchor is None else self.anchor[0],
            0. if self.anchor is None else self.anchor[1], 0.], dtype=np.float32)

    @torch.no_grad()
    def act(self, obs):
        if not isinstance(obs, np.ndarray) or obs.dtype != np.float32 or obs.shape != (2367,) or not np.isfinite(obs).all():
            raise ValueError("Finite float32 observation required")
        step = int(round(76. - 75. * float(obs[self.remaining])))
        if not 1 <= step <= 75 or (self.previous_step is not None and step != self.previous_step + 1):
            raise ValueError("Sequential decisions and fresh actor required")
        if self.previous_step is None and step not in (1, 16):
            raise ValueError("Actor must start at decision 1 or 16")
        self.previous_step = step
        if step < 16:
            return np.zeros(2, np.float32)
        if self.anchor is None:
            self.anchor = obs[self.anchors].copy()
        if step > 30:
            return np.clip((self.anchor.astype(float) - obs[self.anchors].astype(float))
                           * np.array([20., 10.]), -1., 1.).astype(np.float32)
        features = torch.from_numpy(np.concatenate([obs, self.anchor]))
        x = ((features - self.mean) / self.scale).clamp(-10., 10.)
        action = self.net(x).numpy().copy()
        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.:
            raise ValueError("Invalid neural action")
        return action
