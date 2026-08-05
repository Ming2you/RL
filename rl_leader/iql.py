# offline RL(IQL) 학습기(2026-07-30) — 병렬 수집 데이터로 budget 정책 학습(env 롤아웃 없음)
"""online SAC가 발산한 이유(비싼 env·375샘플·untrained critic이 BC 파괴)를 우회한다.
IQL은 데이터셋 안에서만 학습(환경 접촉 0) → env 비용과 무관, 발산 위험 낮음.

  V 손실  : expectile 회귀  E[|τ − 1(Q−V<0)|·(Q−V)²]      (τ=0.7)
  Q 손실  : (r + γ(1−d)·V(s') − Q(s,a))²
  정책    : advantage-weighted regression, w=exp(β·(Q−V)) 를 데이터 행동에 가중 MSE

행동 다양성(sweet/uniform/reactive)이 있어야 advantage가 의미를 갖는다 → collect_parallel.py 참조.
저장된 actor는 eval_all.py / eval_holdout.py와 호환(obs_mu 버퍼 포함).

usage: python rl_leader/iql.py --data "data/rl_dataset/w*.npz" --steps 20000 --out checkpoints/actor_iql.pt
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

if Path(r"C:/torchlib").is_dir() and r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")   # Windows 긴경로 우회 설치일 때만(HANDOFF §7.2)

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
    ap.add_argument("--data", default=str(ROOT / "data" / "rl_dataset" / "w*.npz"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau_exp", type=float, default=0.7, help="expectile")
    ap.add_argument("--beta", type=float, default=3.0, help="AWR 온도")
    ap.add_argument("--rscale", type=float, default=0.01)
    ap.add_argument("--polyak", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=1.0,
                    help="데이터 비율(데이터-스케일링 진단용). <1이면 시드 고정 무작위 부분집합.")
    ap.add_argument("--shape", default="none",
                    choices=["none", "over", "rho", "accum", "mix", "link", "linkmix", "hinge",
                             "farx", "hingefar"],
                    help="potential-based shaping. r' = r + γΦ(s') − Φ(s), Φ=−w·(혼잡 포텐셜). "
                         "최적 정책 불변(Ng et al.)이면서 신호만 조밀해진다 — N_UF는 즉시 TTT "
                         "효과가 ≈0(±300에 0.02)이라 학습 신호가 전부 V(s')로만 흐르는 문제 대응.")
    ap.add_argument("--shape-w", type=float, default=100.0,
                    help="shaping 가중치. rscale 적용 전 원단위(step TTT ~150 규모) 기준.")
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "actor_iql.pt"))
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    obs, act, rew, nobs, done = load_data(a.data)
    if a.frac < 1.0:
        # 데이터-스케일링 진단: 같은 부분집합을 시드 무관하게 뽑아야 frac 간 비교가 성립.
        n_all = obs.shape[0]
        idx = np.random.default_rng(12345).permutation(n_all)[:max(1, int(n_all * a.frac))]
        obs, act, rew, nobs, done = obs[idx], act[idx], rew[idx], nobs[idx], done[idx]
        print(f"frac={a.frac} → {obs.shape[0]}/{n_all} 전이 사용", flush=True)
    if a.shape != "none":
        # Φ(s) — 관측에서 만드는 혼잡 포텐셜(값이 클수록 나쁨). obs 규약(env._observe):
        #   [1]=fveh/1000  [2]=rampq/100  [5]=rho_max(ρ_crit 정규화)  [6]=over/10
        # TTT는 차량 위치를 구분하지 않는다(램프 큐 1대 = 정체 본선 1대). 그래서 방류를
        # 흔들어도 즉시 TTT는 '이전'일 뿐 0에 가깝다. Φ는 '본선이 붕괴 영역에 얼마나
        #들어가 있나'를 재서 그 차이를 신호로 만든다.
        def phi(o):
            if a.shape == "over":     # ρ_crit 초과 세그먼트 수 — MFD 붕괴 직접 지표
                return o[:, 6]
            if a.shape == "rho":      # 임계 초과분(초과했을 때만)
                return np.maximum(0.0, o[:, 5] - 1.0)
            if a.shape == "accum":    # 본선 누적
                return o[:, 1]
            if a.shape in ("farx", "hingefar"):
                # ★far 근사 — probe_far_approx.py 실측: 진짜 far(_mfd_far_cost_to_go)는
                # urban 차량 누적과 corr=+0.881로 지배적 상관(freeway 지표는 |r|<0.16).
                # far는 '지금 얼마나 나쁜가'(hinge)가 아니라 '도시에 쌓인 저장량의 미래
                # 비용'을 재고 있다 — §12.6의 "P-CENT는 urban에서 지불"과 정합.
                # hinge와 far는 corr=0.81이나 성격이 달라(hinge는 진동, far는 단조 증가)
                # 함께 넣으면 서로 다른 신호를 준다.
                fx = o[:, 0]
                if a.shape == "farx":
                    return fx
                return o[:, 6] * np.maximum(0.0, o[:, 5] - 1.0) + 0.3 * fx
            if a.shape == "hinge":
                # ★hinge 근사 — 진짜 hinge는 Σ max(0, ρ_i − ρ_crit)·L·λ 인데 세그먼트별
                # 밀도가 관측에 없다(집계량만). "몇 개가 넘었나 × 얼마나 깊이 넘었나"로
                # 상수배 비례하는 형태를 만든다. over 단독은 초과량을 무시(1만큼 넘든
                # 50만큼 넘든 동일)하므로 정보가 얇다.
                return o[:, 6] * np.maximum(0.0, o[:, 5] - 1.0)
            if a.shape in ("link", "linkmix"):
                # ★공간 분해 Φ — obs[17],[18] = FW_W / FW_E 의 link_rho_max(ρ_crit 정규화).
                # 스칼라 Φ(over/rho_max)는 inc(국소 사고)엔 맞지만 skew(서-동 비대칭)엔
                # 오도한다. 링크별 초과분을 각각 더하면 어느 쪽이 나쁜지 구분된다.
                if o.shape[1] <= 18:
                    raise SystemExit("link shaping은 23차원 obs(Phase 4+)가 필요하다")
                lk = (np.maximum(0.0, o[:, 17] - 1.0) + np.maximum(0.0, o[:, 18] - 1.0))
                return lk if a.shape == "link" else lk + o[:, 6]
            return o[:, 6] + np.maximum(0.0, o[:, 5] - 1.0) + 0.5 * o[:, 1]   # mix
        p_s, p_s2 = phi(obs), phi(nobs)
        # 종료 전이는 s'가 흡수상태 — 부트스트랩이 끊기므로 Φ(s')를 0으로 둔다.
        p_s2 = p_s2 * (1.0 - done.ravel())
        shaped = a.gamma * (-a.shape_w * p_s2) - (-a.shape_w * p_s)
        rew = rew + shaped.reshape(-1, 1).astype(np.float32)
        print(f"shaping={a.shape} w={a.shape_w}: Φ mean={p_s.mean():.3f} "
              f"보정항 mean={shaped.mean():+.2f} sd={shaped.std():.2f} "
              f"(원 보상 mean={float(rew.mean()):+.1f})", flush=True)
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
