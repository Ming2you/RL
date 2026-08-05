# BC 정책 폐루프 평가(Phase 1 게이트, 2026-07-22) — BC정책 env 롤아웃 → optimizer와 TTT 비교
"""학습된 BC 정책(numpy MLP)을 RLLeaderEnv에서 굴려 windowed/whole TTT를 측정.
Phase 1 검증: BC정책 TTT ≈ optimizer-leader TTT 이면 게이트 통과.

usage: python rl_leader/bc_eval.py <policy.npz> <scenario> [--optimizer]
  --optimizer: 같은 시나리오를 optimizer-leader로도 굴려 대조.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv


def load_policy(path):
    d = np.load(path)
    W = {k: d[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    mu, sd = d["obs_mu"], d["obs_sd"]

    def act(obs):
        x = (np.asarray(obs, float) - mu) / sd
        a1 = np.maximum(0.0, x @ W["W1"] + W["b1"])
        a2 = np.maximum(0.0, a1 @ W["W2"] + W["b2"])
        return np.tanh(a2 @ W["W3"] + W["b3"])
    return act


def rollout_policy(scenario, policy):
    env = RLLeaderEnv(scenario_name=scenario)
    obs = env.reset()
    done = False
    while not done:
        obs, r, done, info = env.step(policy(obs))
    return float(env.sim.total_ttt), float(env.sim.urban_ttt), float(env.sim.freeway_ttt)


def rollout_optimizer(scenario):
    from src.controllers.stackelberg_wu_metered import StackelbergWuMeteredController
    env = RLLeaderEnv(scenario_name=scenario)
    teacher = StackelbergWuMeteredController(env.cfg)
    env.reset()
    done = False
    while not done:
        forecast = env._forecast()
        prev = env.previous if env.previous is not None else env._fixed_prev()
        control = teacher.decide(env.sim.state.copy(), forecast, prev)
        _, _, done = env.step_with_control(control)
    return float(env.sim.total_ttt), float(env.sim.urban_ttt), float(env.sim.freeway_ttt)


if __name__ == "__main__":
    policy_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints" / "bc_policy.npz")
    scen = sys.argv[2] if len(sys.argv) > 2 else "sweet_170_w60"
    pol = load_policy(policy_path)
    bt, bu, bf = rollout_policy(scen, pol)
    print(f"[BC]        {scen}: total_ttt={bt:.1f} (urban {bu:.1f} / freeway {bf:.1f})")
    if "--optimizer" in sys.argv:
        ot, ou, of = rollout_optimizer(scen)
        print(f"[optimizer] {scen}: total_ttt={ot:.1f} (urban {ou:.1f} / freeway {of:.1f})")
        print(f"  BC/optimizer TTT ratio = {bt/ot:.3f}  (1.0 근접 = Phase1 게이트 통과)")
