# offline RL(IQL) 학습기(2026-07-30) — 병렬 수집 데이터로 budget 정책 학습(env 롤아웃 없음)
"""online SAC가 발산한 이유(비싼 env·375샘플·untrained critic이 BC 파괴)를 우회한다.
IQL은 데이터셋 안에서만 학습(환경 접촉 0) → env 비용과 무관, 발산 위험 낮음.

  V 손실  : expectile 회귀  E[|τ − 1(Q−V<0)|·(Q−V)²]      (τ=0.7)
  Q 손실  : (r + γ(1−d)·V(s') − Q(s,a))²
  정책    : advantage-weighted regression, w=exp(β·(Q−V)) 를 데이터 행동에 가중 MSE

행동 다양성(sweet/uniform/reactive)이 있어야 advantage가 의미를 갖는다 → collect_parallel.py 참조.
저장된 actor는 eval_all.py / eval_holdout.py와 호환(obs_mu 버퍼 포함).

usage: python rl_leader/iql.py --data "rl_leader/data/w*.npz" --steps 20000 --out rl_leader/actor_iql.pt
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

if r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.nets import Actor, Critic, mlp


def load_data(pattern):
    O, A, R, O2, D = [], [], [], [], []
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"데이터 없음: {pattern}")
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            if int(d["obs"].shape[0]) == 0:
                continue
            O.append(d["obs"]); A.append(d["act"]); R.append(d["rew"])
            O2.append(d["next_obs"]); D.append(d["done"])
        except Exception as e:
            print(f"  skip {f}: {e}", flush=True)
    obs = np.concatenate(O).astype(np.float32)
    act = np.concatenate(A).astype(np.float32)
    rew = np.concatenate(R).astype(np.float32).reshape(-1, 1)
    nobs = np.concatenate(O2).astype(np.float32)
    done = np.concatenate(D).astype(np.float32).reshape(-1, 1)
    print(f"데이터 {len(files)}파일 → {obs.shape[0]} 전이, obs_dim={obs.shape[1]}, act_dim={act.shape[1]}", flush=True)
    return obs, act, rew, nobs, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "rl_leader" / "data" / "w*.npz"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau_exp", type=float, default=0.7, help="expectile")
    ap.add_argument("--beta", type=float, default=3.0, help="AWR 온도")
    ap.add_argument("--rscale", type=float, default=0.01)
    ap.add_argument("--polyak", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "rl_leader" / "actor_iql.pt"))
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    obs, act, rew, nobs, done = load_data(a.data)
    od, ad = obs.shape[1], act.shape[1]

    mu, sd = obs.mean(0), obs.std(0) + 1e-6
    nrm = lambda x: torch.as_tensor((x - mu) / sd)
    # critic/value는 정규화된 obs를, actor는 raw obs를 받는다(내부에서 스스로 정규화).
    O, A = nrm(obs), torch.as_tensor(act)
    O_raw = torch.as_tensor(obs)
    R = torch.as_tensor(rew * a.rscale)
    O2, Dn = nrm(nobs), torch.as_tensor(done)

    actor = Actor(od, ad); actor.set_obs_norm(mu, sd)   # actor는 내부 정규화
    critic, ctarg = Critic(od, ad), Critic(od, ad)
    ctarg.load_state_dict(critic.state_dict())
    vnet = mlp(od, 128, 1)

    a_opt = torch.optim.Adam(actor.parameters(), 3e-4)
    c_opt = torch.optim.Adam(critic.parameters(), 3e-4)
    v_opt = torch.optim.Adam(vnet.parameters(), 3e-4)
    N = O.shape[0]

    for step in range(1, a.steps + 1):
        i = torch.as_tensor(np.random.randint(0, N, a.batch))
        o, ac, r, o2, dn = O[i], A[i], R[i], O2[i], Dn[i]

        # --- V: expectile 회귀 ---
        with torch.no_grad():
            q1t, q2t = ctarg(o, ac)
            q = torch.min(q1t, q2t)
        v = vnet(o)
        diff = q - v
        w_exp = torch.where(diff < 0, 1.0 - a.tau_exp, a.tau_exp)
        v_loss = (w_exp * diff.pow(2)).mean()
        v_opt.zero_grad(); v_loss.backward(); v_opt.step()

        # --- Q: TD with V(s') ---
        with torch.no_grad():
            y = r + a.gamma * (1 - dn) * vnet(o2)
        q1, q2 = critic(o, ac)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        c_opt.zero_grad(); c_loss.backward(); c_opt.step()

        # --- 정책: advantage-weighted regression ---
        with torch.no_grad():
            q1p, q2p = critic(o, ac)
            adv = torch.min(q1p, q2p) - vnet(o)
            wgt = torch.clamp(torch.exp(a.beta * adv), max=100.0)
        mu_a, _ = actor(O_raw[i])     # ★ raw obs (actor가 내부에서 정규화)
        a_loss = (wgt * (torch.tanh(mu_a) - ac).pow(2).mean(-1, keepdim=True)).mean()
        a_opt.zero_grad(); a_loss.backward(); a_opt.step()

        with torch.no_grad():
            for p, pt in zip(critic.parameters(), ctarg.parameters()):
                pt.data.mul_(1 - a.polyak).add_(a.polyak * p.data)

        if step % 2000 == 0 or step == 1:
            print(f"  step {step:6d}  V={v_loss.item():.4f} Q={c_loss.item():.4f} "
                  f"pi={a_loss.item():.4f}  adv[min/mean/max]="
                  f"{adv.min().item():+.3f}/{adv.mean().item():+.3f}/{adv.max().item():+.3f}", flush=True)
            torch.save(actor.state_dict(), a.out)

    torch.save(actor.state_dict(), a.out)
    print(f"saved actor(IQL) → {a.out}", flush=True)


if __name__ == "__main__":
    main()
