from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping


@dataclass(frozen=True)
class NashProbeResult:
    """One-step approximate follower exploitability diagnostic."""

    epsilon_Nash_hat: float
    best_deviating_agent_id: str | None
    realized_reward: float
    best_unilateral_reward: float
    best_unilateral_action_index: int | None
    realized_action_index: int | None
    candidate_rewards: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "epsilon_Nash_hat": float(self.epsilon_Nash_hat),
            "best_deviating_agent_id": self.best_deviating_agent_id,
            "realized_reward": float(self.realized_reward),
            "best_unilateral_reward": float(self.best_unilateral_reward),
            "best_unilateral_action_index": self.best_unilateral_action_index,
            "realized_action_index": self.realized_action_index,
            "candidate_rewards": {
                agent_id: {int(index): float(reward) for index, reward in rewards.items()}
                for agent_id, rewards in self.candidate_rewards.items()
            },
        }


def probe_follower_nash_residual(
    env,
    leader_action_index: int,
    follower_action_indices: Mapping[str, int] | None = None,
) -> NashProbeResult:
    """Estimate one-step follower Nash residual with unilateral local deviations.

    The probe keeps the current leader target and other follower actions fixed,
    varies one follower's discrete local action, applies the usual physical
    projection and simulator step, and reads that follower's one-step local
    reward. The environment is restored before returning.
    """

    base_sim = env.sim.copy()
    base_previous_control = env.previous_control.copy() if env.previous_control is not None else None
    base_step_index = int(env.step_index)
    base_records = list(env.records)
    completed_indices = env._complete_follower_indices(follower_action_indices or {})
    candidate_rewards: Dict[str, Dict[int, float]] = {}
    best_agent_id: str | None = None
    best_epsilon = 0.0
    best_reward = 0.0
    best_action_index: int | None = None
    realized_reward_for_best = 0.0
    realized_action_for_best: int | None = None

    try:
        for agent_id, action_space in env.follower_action_spaces.items():
            realized_action_index = int(completed_indices[agent_id])
            realized_reward = None
            candidate_rewards[agent_id] = {}
            for action_index in range(action_space.size):
                probe_indices = dict(completed_indices)
                probe_indices[agent_id] = int(action_index)
                _restore_env(env, base_sim, base_previous_control, base_step_index, base_records)
                step = env.step(int(leader_action_index), probe_indices)
                reward = float(step.follower_rewards[agent_id])
                candidate_rewards[agent_id][int(action_index)] = reward
                if action_index == realized_action_index:
                    realized_reward = reward
            if realized_reward is None:
                raise RuntimeError(f"realized action {realized_action_index} was not evaluated for {agent_id}")
            agent_best_action, agent_best_reward = max(
                candidate_rewards[agent_id].items(),
                key=lambda item: item[1],
            )
            agent_epsilon = max(0.0, float(agent_best_reward - realized_reward))
            if best_agent_id is None or agent_epsilon > best_epsilon:
                best_agent_id = agent_id
                best_epsilon = agent_epsilon
                best_reward = float(agent_best_reward)
                best_action_index = int(agent_best_action)
                realized_reward_for_best = float(realized_reward)
                realized_action_for_best = int(realized_action_index)
    finally:
        _restore_env(env, base_sim, base_previous_control, base_step_index, base_records)

    if best_agent_id is None:
        return NashProbeResult(
            epsilon_Nash_hat=0.0,
            best_deviating_agent_id=None,
            realized_reward=0.0,
            best_unilateral_reward=0.0,
            best_unilateral_action_index=None,
            realized_action_index=None,
            candidate_rewards={},
        )
    return NashProbeResult(
        epsilon_Nash_hat=float(best_epsilon),
        best_deviating_agent_id=best_agent_id,
        realized_reward=realized_reward_for_best,
        best_unilateral_reward=best_reward,
        best_unilateral_action_index=best_action_index,
        realized_action_index=realized_action_for_best,
        candidate_rewards=candidate_rewards,
    )


def _restore_env(env, sim, previous_control, step_index: int, records) -> None:
    env.sim = sim.copy()
    env.previous_control = previous_control.copy() if previous_control is not None else None
    env.step_index = int(step_index)
    env.records = list(records)
