# RL leader torch networks(2026-07-22) — SAC Actor(tanh-squash Gaussian) + twin Critic
"""torch가 C:/torchlib(짧은경로 --target 설치, Windows 긴경로 회피)에 있으므로 부트스트랩."""
from __future__ import annotations
import sys
from pathlib import Path
if Path(r"C:/torchlib").is_dir() and r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")   # Windows 긴경로 우회 설치일 때만(HANDOFF §7.2)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(din, dh, dout, act=nn.ReLU):
    return nn.Sequential(nn.Linear(din, dh), act(), nn.Linear(dh, dh), act(), nn.Linear(dh, dout))


class Actor(nn.Module):
    """obs → tanh-squash Gaussian over action∈[-1,1]^A. obs 정규화 버퍼 내장."""
    def __init__(self, obs_dim, act_dim, dh=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, dh), nn.ReLU(), nn.Linear(dh, dh), nn.ReLU())
        self.mu = nn.Linear(dh, act_dim)
        self.log_std = nn.Linear(dh, act_dim)
        self.register_buffer("obs_mu", torch.zeros(obs_dim))
        self.register_buffer("obs_sd", torch.ones(obs_dim))

    def _norm(self, obs):
        return (obs - self.obs_mu) / self.obs_sd

    def forward(self, obs):
        h = self.trunk(self._norm(obs))
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre = mu + std * eps
        a = torch.tanh(pre)
        # tanh-squash log-prob 보정
        logp = (-0.5 * (eps ** 2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp -= torch.log(1 - a ** 2 + 1e-6).sum(-1, keepdim=True)
        return a, logp

    @torch.no_grad()
    def act(self, obs_np, deterministic=True):
        obs = torch.as_tensor(np.asarray(obs_np, np.float32)).unsqueeze(0)
        if deterministic:
            mu, _ = self.forward(obs)
            return torch.tanh(mu).squeeze(0).numpy()
        a, _ = self.sample(obs)
        return a.squeeze(0).numpy()

    def set_obs_norm(self, mu, sd):
        self.obs_mu.copy_(torch.as_tensor(np.asarray(mu, np.float32)))
        self.obs_sd.copy_(torch.as_tensor(np.asarray(sd, np.float32)).clamp_min(1e-6))


class Critic(nn.Module):
    """twin Q(obs, action)."""
    def __init__(self, obs_dim, act_dim, dh=128):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, dh, 1)
        self.q2 = mlp(obs_dim + act_dim, dh, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)
