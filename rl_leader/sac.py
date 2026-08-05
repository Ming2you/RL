# SAC 학습(Phase 2, 2026-07-22) — RL leader budget 정책, BC warm-start
"""RLLeaderEnv에서 SAC로 leader budget 정책 학습. actor를 BC 체크포인트로 init.
reward=−ΔTTT(스케일링). env.step은 follower만이라 ~1.4s → 학습 가능.

usage: python rl_leader/sac.py [--bc actor_bc.pt] [--scenario ...] [--steps N] [--out actor_sac.pt]
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
if Path(r"C:/torchlib").is_dir() and r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")   # Windows 긴경로 우회 설치일 때만(HANDOFF §7.2)

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv, make_random_scenario
from rl_leader.nets import Actor, Critic


class Replay:
    def __init__(self, cap, od, ad):
        self.o = np.zeros((cap, od), np.float32); self.a = np.zeros((cap, ad), np.float32)
        self.r = np.zeros((cap, 1), np.float32); self.o2 = np.zeros((cap, od), np.float32)
        self.d = np.zeros((cap, 1), np.float32); self.cap = cap; self.i = 0; self.n = 0
    def add(self, o, a, r, o2, d):
        j = self.i; self.o[j] = o; self.a[j] = a; self.r[j] = r; self.o2[j] = o2; self.d[j] = d
        self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)
    def sample(self, b):
        k = np.random.randint(0, self.n, size=b)
        t = lambda x: torch.as_tensor(x[k])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # continuous: 에피소드마다 연속 수요분포(155~240)+랜덤 stressor에서 새 env 샘플(도메인 랜덤화).
    # hold-out: stressor 활성 시 수요≤holdout_demand → 190+stressor는 학습에 없음(test 전용).
    if args.continuous:
        def new_env():
            scen = make_random_scenario(rng, args.holdout_demand)
            return RLLeaderEnv(scenario_dict=scen, T_total=args.T)
        env = new_env(); cur = "random"; envs = None
        print(f"continuous SAC: demand~U[1.55,2.40], stressor∈{{none,skew,incident}}, "
              f"holdout_demand={args.holdout_demand}", flush=True)
    else:
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        envs = {s: RLLeaderEnv(scenario_name=s, T_total=args.T) for s in scenarios}
        cur = scenarios[0]; env = envs[cur]
        print(f"multi-scenario SAC: {scenarios}", flush=True)
    od, ad = env.obs_dim, env.action_dim
    actor = Actor(od, ad)
    if args.bc and Path(args.bc).exists():
        actor.load_state_dict(torch.load(args.bc)); print(f"actor init ← BC {args.bc}")
    critic = Critic(od, ad); ctarg = Critic(od, ad); ctarg.load_state_dict(critic.state_dict())
    a_opt = torch.optim.Adam(actor.parameters(), 3e-4)
    c_opt = torch.optim.Adam(critic.parameters(), 3e-4)
    log_alpha = torch.zeros(1, requires_grad=True); al_opt = torch.optim.Adam([log_alpha], 3e-4)
    targ_H = -float(ad); gamma, tau, rscale = 0.99, 0.005, args.rscale
    rb = Replay(200_000, od, ad)
    obs = env.reset(); ep_r = 0.0; ep_i = 0; best = None; t0 = time.time()
    for step in range(1, args.steps + 1):
        if rb.n < args.warmup_rand:
            a = np.random.uniform(-1, 1, ad).astype(np.float32)
        else:
            a = actor.act(obs, deterministic=False)
        o2, r, done, info = env.step(a)
        rb.add(obs, a, r * rscale, o2, float(done)); obs = o2; ep_r += r
        if done:
            ep_i += 1
            print(f"  ep {ep_i} [{cur}]: return={ep_r:.0f} cum_ttt={info['cum_ttt']:.0f} "
                  f"step={step} elapsed={time.time()-t0:.0f}s", flush=True)
            if args.continuous:
                env = new_env()
                cur = f"d{env.scenario.urban_scale:.2f}"
            else:
                cur = scenarios[np.random.randint(len(scenarios))]; env = envs[cur]
            obs = env.reset(); ep_r = 0.0
        if rb.n >= max(args.batch, args.warmup_rand):
            o, ac, rw, o2b, db = rb.sample(args.batch)
            with torch.no_grad():
                a2, lp2 = actor.sample(o2b); q1t, q2t = ctarg(o2b, a2)
                qt = torch.min(q1t, q2t) - log_alpha.exp() * lp2
                y = rw + gamma * (1 - db) * qt
            q1, q2 = critic(o, ac)
            c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            c_opt.zero_grad(); c_loss.backward(); c_opt.step()
            ap, lp = actor.sample(o); q1p, q2p = critic(o, ap)
            a_loss = (log_alpha.exp().detach() * lp - torch.min(q1p, q2p)).mean()
            a_opt.zero_grad(); a_loss.backward(); a_opt.step()
            al_loss = -(log_alpha * (lp.detach() + targ_H)).mean()
            al_opt.zero_grad(); al_loss.backward(); al_opt.step()
            for p, pt in zip(critic.parameters(), ctarg.parameters()):
                pt.data.mul_(1 - tau).add_(tau * p.data)
        if step % args.save_every == 0:
            torch.save(actor.state_dict(), args.out)
    torch.save(actor.state_dict(), args.out)
    print(f"saved actor(SAC) → {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bc", default=str(ROOT / "checkpoints" / "actor_bc.pt"))
    p.add_argument("--scenarios", default="sweet_155_w60,sweet_170_w60,sweet_170_skew15_w60,sweet_170_incident_w60,sweet_190_w60")
    p.add_argument("--continuous", action="store_true", help="연속 수요분포 도메인 랜덤화(hold-out 프로토콜)")
    p.add_argument("--holdout_demand", type=float, default=1.80, help="stressor 활성 시 수요 상한(190은 test 전용)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=float, default=14400.0)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--warmup_rand", type=int, default=500)
    p.add_argument("--rscale", type=float, default=0.01)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--out", default=str(ROOT / "checkpoints" / "actor_sac.pt"))
    train(p.parse_args())
