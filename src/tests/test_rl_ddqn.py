# DDQN 트레이너 코어 스모크 테스트: 학습 루프가 돌고 리플레이가 채워지고 체크포인트 save/load가 일관한지 검증
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from src.models.demand import load_scenarios
from src.models.state import ExperimentConfig
from src.rl.ddqn import DQNConfig, StackelbergDDQNTrainer, set_global_seed
from src.rl.env import StackelbergRLEnvironment


def _build_trainer():
    set_global_seed(0)
    cfg = ExperimentConfig.from_file("src/config/default.yaml")
    scenario = load_scenarios("src/config/scenarios.yaml")["peak_demand"]
    env = StackelbergRLEnvironment(cfg, scenario, seed=0)
    # 작고 빠른 설정: 즉시 학습이 시작되도록 min_buffer/batch를 작게.
    dqn_cfg = DQNConfig(
        hidden=(16,),
        batch_size=8,
        buffer_capacity=2000,
        min_buffer=8,
        target_sync_every=10,
        eps_decay_steps=100,
    )
    return StackelbergDDQNTrainer(env, dqn_cfg, device="cpu", seed=0)


class TestDDQNTrainer(unittest.TestCase):
    def test_agents_match_action_spaces(self):
        trainer = _build_trainer()
        self.assertEqual(trainer.leader.n_actions, trainer.env.leader_action_space.size)
        self.assertEqual(set(trainer.followers), set(trainer.env.follower_action_spaces))
        for aid, agent in trainer.followers.items():
            self.assertEqual(agent.n_actions, trainer.env.follower_action_spaces[aid].size)

    def test_run_episode_fills_buffer_and_finite_stats(self):
        trainer = _build_trainer()
        stats = trainer.run_episode(0, train=True)
        n = trainer.env.cfg.simulation.n_control_steps
        self.assertEqual(stats.steps, n)
        self.assertTrue(math.isfinite(stats.leader_return))
        self.assertTrue(math.isfinite(stats.mean_follower_return))
        # 리플레이가 채워졌어야 한다(에피소드 길이만큼 transition).
        self.assertGreater(len(trainer.leader.buffer), 0)
        for agent in trainer.followers.values():
            self.assertGreater(len(agent.buffer), 0)

    def test_training_steps_advance(self):
        trainer = _build_trainer()
        trainer.run_episode(0, train=True)
        # min_buffer가 작아 학습이 실제로 수행됐어야 한다.
        self.assertGreater(trainer.leader.train_steps, 0)

    def test_greedy_eval_does_not_advance_global_step(self):
        trainer = _build_trainer()
        trainer.run_episode(0, train=True)
        global_step = trainer.global_step
        leader_train_steps = trainer.leader.train_steps

        stats = trainer.run_episode(1, train=False, greedy=True)

        self.assertTrue(math.isfinite(stats.episode_ttt))
        self.assertEqual(trainer.global_step, global_step)
        self.assertEqual(trainer.leader.train_steps, leader_train_steps)

    def test_checkpoint_save_load_roundtrip(self):
        trainer = _build_trainer()
        trainer.run_episode(0, train=True)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pt"
            trainer.save(path)
            gs = trainer.global_step
            trainer2 = _build_trainer()
            trainer2.load(path)
            self.assertEqual(trainer2.global_step, gs)
            # online 가중치가 일치해야 한다.
            import torch

            for p1, p2 in zip(trainer.leader.online.parameters(), trainer2.leader.online.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()
