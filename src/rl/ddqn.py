# Stackelberg 다중에이전트 DDQN 학습 코어: leader 1 + follower N을 각각 독립 Double DQN(PyTorch)으로 학습
"""Spec 19의 leader+follower RL을 실제로 학습시키는 트레이너 코어.

설계.
- 에이전트별 독립 Double DQN(online/target Q-net, 리플레이, Adam). leader 1개 + follower N개.
- Stackelberg 순서: 매 스텝 leader가 먼저 target(N_P*,N_UF*)을 ε-greedy로 고른 뒤, 그 target을
  조건으로 follower 관측을 만들어 follower가 고른다(env.follower_observations_for_leader).
- transition의 next_obs는 *다음 스텝* 관측으로 한 스텝 지연 완성한다. follower next_obs가 다음 스텝
  leader action 조건으로 잡혀 bootstrap 분포 불일치를 피한다(neutral-leader 근사 대신).
- 보상은 그대로 두면(leader ~ -1000 규모) Q값이 발산하기 쉬워 reward_scale로 축소.

학습 외 로깅/체크포인트는 scripts/train_rl_ddqn.py(thin CLI)가 담당하고, 여기서는 순수 학습 루프와
save/load만 노출한다(테스트 용이).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.rl.env import StackelbergRLEnvironment


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    """obs 벡터 -> 각 이산 action의 Q값을 내는 작은 MLP."""

    def __init__(self, input_dim: int, n_actions: int, hidden: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.ReLU()]
            dim = width
        layers += [nn.Linear(dim, n_actions)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """고정 용량 ring-buffer 리플레이. numpy 배열로 보관해 샘플링이 빠르다."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, obs, action, reward, next_obs, done) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.idx = (i + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def sample(self, batch: int, rng: np.random.Generator):
        idxs = rng.integers(0, len(self), size=batch)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs],
        )


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1.0e-3
    batch_size: int = 64
    buffer_capacity: int = 50_000
    hidden: Tuple[int, ...] = (128, 128)
    target_sync_every: int = 500  # gradient step 기준
    min_buffer: int = 500
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000
    reward_scale: float = 0.01


class DQNAgent:
    """단일 에이전트(leader 또는 follower 1개)의 Double DQN."""

    def __init__(self, agent_id, obs_dim, n_actions, cfg: DQNConfig, device, neutral_index=0):
        self.agent_id = str(agent_id)
        self.n_actions = int(n_actions)
        self.cfg = cfg
        self.device = device
        self.neutral_index = int(neutral_index)
        self.online = QNetwork(obs_dim, n_actions, cfg.hidden).to(device)
        self.target = QNetwork(obs_dim, n_actions, cfg.hidden).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_capacity, obs_dim)
        self.train_steps = 0

    def epsilon(self, global_step: int) -> float:
        c = self.cfg
        frac = min(1.0, global_step / max(1, c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def select(self, obs_vec, global_step: int, rng: np.random.Generator, greedy: bool = False) -> int:
        if not greedy and rng.random() < self.epsilon(global_step):
            return int(rng.integers(0, self.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(t).argmax(1).item())

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward * self.cfg.reward_scale, next_obs, done)

    def train_batch(self, rng: np.random.Generator):
        c = self.cfg
        if len(self.buffer) < max(c.min_buffer, c.batch_size):
            return None
        obs, act, rew, nobs, done = self.buffer.sample(c.batch_size, rng)
        dev = self.device
        obs = torch.as_tensor(obs, device=dev)
        nobs = torch.as_tensor(nobs, device=dev)
        act = torch.as_tensor(act, device=dev)
        rew = torch.as_tensor(rew, device=dev)
        done = torch.as_tensor(done, device=dev)

        q = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: action은 online으로 고르고 값은 target으로 평가.
            next_actions = self.online(nobs).argmax(1)
            next_q = self.target(nobs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rew + c.gamma * (1.0 - done) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.grad_clip)
        self.opt.step()
        self.train_steps += 1
        if self.train_steps % c.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def state_dict(self) -> Dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "train_steps": self.train_steps,
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.online.load_state_dict(sd["online"])
        self.target.load_state_dict(sd["target"])
        self.opt.load_state_dict(sd["opt"])
        self.train_steps = int(sd.get("train_steps", 0))


@dataclass
class EpisodeStats:
    episode: int
    steps: int
    leader_return: float
    mean_follower_return: float
    leader_loss: float
    mean_follower_loss: float
    epsilon: float
    episode_ttt: float
    follower_returns: Dict[str, float] = field(default_factory=dict)


class StackelbergDDQNTrainer:
    """env를 굴리며 leader+follower 독립 DDQN을 동시에 학습한다."""

    def __init__(self, env: StackelbergRLEnvironment, cfg: DQNConfig, device: str = "cpu", seed: int = 0):
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.global_step = 0

        probe = env.observe()
        self.leader = DQNAgent(
            "__leader__",
            int(probe.leader.features.shape[0]),
            env.leader_action_space.size,
            cfg,
            self.device,
            env.leader_action_space.neutral_index(),
        )
        neutral_followers = env.follower_observations_for_leader(env.leader_action_space.neutral_index())
        self.followers: Dict[str, DQNAgent] = {}
        for agent_id, space in env.follower_action_spaces.items():
            self.followers[agent_id] = DQNAgent(
                agent_id,
                int(neutral_followers[agent_id].features.shape[0]),
                space.size,
                cfg,
                self.device,
                space.neutral_index(),
            )

    def _push(self, prev: Dict, next_leader_obs, next_follower_obs: Dict, done: bool) -> None:
        self.leader.push(prev["leader_obs"], prev["leader_a"], prev["leader_r"], next_leader_obs, done)
        for agent_id, agent in self.followers.items():
            agent.push(
                prev["follower_obs"][agent_id],
                prev["follower_a"][agent_id],
                prev["follower_r"][agent_id],
                next_follower_obs[agent_id],
                done,
            )

    def run_episode(self, episode_idx: int, train: bool = True, greedy: bool = False) -> EpisodeStats:
        env = self.env
        env.reset()
        n_steps = env.cfg.simulation.n_control_steps

        leader_return = 0.0
        follower_return = {aid: 0.0 for aid in self.followers}
        leader_losses: list[float] = []
        follower_losses: list[float] = []
        episode_ttt = 0.0
        last_eps = self.leader.epsilon(self.global_step)
        prev: Dict | None = None

        for _ in range(n_steps):
            leader_obs = env.observe().leader.features
            leader_a = self.leader.select(leader_obs, self.global_step, self.rng, greedy)
            follower_obs_map = env.follower_observations_for_leader(leader_a)
            follower_feat = {aid: follower_obs_map[aid].features for aid in self.followers}

            # 이전 스텝 transition을 현재 관측으로 완성(한 스텝 지연 -> 조건부 next_obs 정합).
            if prev is not None and train:
                self._push(prev, leader_obs, follower_feat, done=False)

            follower_a = {
                aid: agent.select(follower_feat[aid], self.global_step, self.rng, greedy)
                for aid, agent in self.followers.items()
            }
            step = env.step(leader_a, follower_a)

            leader_return += step.leader_reward
            for aid in follower_return:
                follower_return[aid] += step.follower_rewards.get(aid, 0.0)
            episode_ttt += float(step.record.global_step_ttt)
            last_eps = self.leader.epsilon(self.global_step)

            prev = {
                "leader_obs": leader_obs,
                "leader_a": leader_a,
                "leader_r": step.leader_reward,
                "follower_obs": follower_feat,
                "follower_a": follower_a,
                "follower_r": {aid: step.follower_rewards.get(aid, 0.0) for aid in self.followers},
            }
            if train:
                self.global_step += 1

            if train:
                lloss = self.leader.train_batch(self.rng)
                if lloss is not None:
                    leader_losses.append(lloss)
                for agent in self.followers.values():
                    floss = agent.train_batch(self.rng)
                    if floss is not None:
                        follower_losses.append(floss)

            if step.done:
                if prev is not None and train:
                    # 종료 transition: next_obs는 done 마스크로 무시되므로 현재 관측 재사용.
                    self._push(prev, leader_obs, follower_feat, done=True)
                break

        n_follow = max(1, len(self.followers))
        return EpisodeStats(
            episode=episode_idx,
            steps=n_steps,
            leader_return=leader_return,
            mean_follower_return=sum(follower_return.values()) / n_follow,
            leader_loss=float(np.mean(leader_losses)) if leader_losses else float("nan"),
            mean_follower_loss=float(np.mean(follower_losses)) if follower_losses else float("nan"),
            epsilon=last_eps,
            episode_ttt=episode_ttt,
            follower_returns=follower_return,
        )

    def save(self, path) -> None:
        payload = {
            "leader": self.leader.state_dict(),
            "followers": {aid: ag.state_dict() for aid, ag in self.followers.items()},
            "global_step": self.global_step,
        }
        torch.save(payload, path)

    def load(self, path) -> None:
        sd = torch.load(path, map_location=self.device)
        self.leader.load_state_dict(sd["leader"])
        for aid, ag in self.followers.items():
            if aid in sd["followers"]:
                ag.load_state_dict(sd["followers"][aid])
        self.global_step = int(sd.get("global_step", 0))
