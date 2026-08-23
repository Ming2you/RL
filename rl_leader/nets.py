# RL leader torch networks(2026-07-22) — SAC Actor(tanh-squash Gaussian) + twin Critic
from __future__ import annotations

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


class UnifiedCoordinationActor(nn.Module):
    """One trunk with parameter-shared urban and freeway block heads."""

    def __init__(
        self, obs_dim, urban_blocks, freeway_blocks, vsl_blocks=0, certificate_blocks=0,
        block_dim=5, vsl_block_dim=2, dh=128, embed_dim=8,
    ):
        super().__init__()
        self.urban_blocks = int(urban_blocks)
        self.freeway_blocks = int(freeway_blocks)
        self.vsl_blocks = int(vsl_blocks)
        self.certificate_blocks = int(certificate_blocks)
        self.block_dim = int(block_dim)
        self.vsl_block_dim = int(vsl_block_dim)
        self.action_dim = (
            2
            + block_dim * (self.urban_blocks + self.freeway_blocks)
            + vsl_block_dim * self.vsl_blocks
            + self.certificate_blocks
        )
        self.trunk = nn.Sequential(nn.Linear(obs_dim, dh), nn.ReLU(), nn.Linear(dh, dh), nn.ReLU())
        self.budget_mu = nn.Linear(dh, 2)
        self.budget_log_std = nn.Linear(dh, 2)
        self.urban_embedding = nn.Embedding(max(self.urban_blocks, 1), embed_dim)
        self.freeway_embedding = nn.Embedding(max(self.freeway_blocks, 1), embed_dim)
        self.vsl_embedding = nn.Embedding(max(self.vsl_blocks, 1), embed_dim)
        self.urban_head = mlp(dh + embed_dim, dh, 2 * block_dim)
        self.freeway_head = mlp(dh + embed_dim, dh, 2 * block_dim)
        self.vsl_head = mlp(dh + embed_dim, dh, 2 * vsl_block_dim)
        self.certificate_head = mlp(dh + embed_dim, dh, 2)
        self.register_buffer("obs_mu", torch.zeros(obs_dim))
        self.register_buffer("obs_sd", torch.ones(obs_dim))

    def _family(self, h, count, embedding, head, output_dim):
        if count <= 0:
            empty = h.new_zeros((h.shape[0], 0))
            return empty, empty
        index = torch.arange(count, device=h.device)
        emb = embedding(index).unsqueeze(0).expand(h.shape[0], -1, -1)
        repeated = h.unsqueeze(1).expand(-1, count, -1)
        output = head(torch.cat((repeated, emb), dim=-1))
        mu, log_std = output.split(output_dim, dim=-1)
        return mu.reshape(h.shape[0], -1), log_std.reshape(h.shape[0], -1)

    def forward(self, obs):
        h = self.trunk((obs - self.obs_mu) / self.obs_sd)
        urban_mu, urban_log = self._family(
            h, self.urban_blocks, self.urban_embedding, self.urban_head, self.block_dim,
        )
        freeway_mu, freeway_log = self._family(
            h, self.freeway_blocks, self.freeway_embedding, self.freeway_head, self.block_dim,
        )
        vsl_mu, vsl_log = self._family(
            h, self.vsl_blocks, self.vsl_embedding, self.vsl_head, self.vsl_block_dim,
        )
        certificate_mu, certificate_log = self._family(
            h, self.certificate_blocks, self.freeway_embedding, self.certificate_head, 1,
        )
        mu = torch.cat(
            (self.budget_mu(h), urban_mu, freeway_mu, vsl_mu, certificate_mu), dim=-1,
        )
        log_std = torch.cat(
            (self.budget_log_std(h), urban_log, freeway_log, vsl_log, certificate_log), dim=-1,
        )
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        action = torch.tanh(mu + std * eps)
        logp = (-0.5 * eps.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp -= torch.log(1 - action.pow(2) + 1.0e-6).sum(-1, keepdim=True)
        return action, logp

    @torch.no_grad()
    def act(self, obs_np, deterministic=True):
        obs = torch.as_tensor(np.asarray(obs_np, np.float32)).unsqueeze(0)
        if deterministic:
            mu, _ = self.forward(obs)
            return torch.tanh(mu).squeeze(0).cpu().numpy()
        action, _ = self.sample(obs)
        return action.squeeze(0).cpu().numpy()

    def set_obs_norm(self, mu, sd):
        self.obs_mu.copy_(torch.as_tensor(np.asarray(mu, np.float32)))
        self.obs_sd.copy_(torch.as_tensor(np.asarray(sd, np.float32)).clamp_min(1.0e-6))
