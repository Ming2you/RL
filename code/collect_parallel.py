# 병렬 offline RL 데이터 수집(2026-07-30) — 도메인 랜덤화 + 행동 다양성 + 증분 저장
"""375샘플 병목 해소용. RLLeaderEnv(리더 탐색 우회, budget 직접 주입)로 다양한 budget을
굴려 (obs, action, reward, next_obs) 전이를 모은다. offline RL(IQL/AWAC)이 teacher를
넘으려면 teacher 궤적만으론 안 되고 행동 대비(counterfactual)가 필요하다.

행동 정책(에피소드마다 모드 선택 + 스텝 노이즈):
  sweet    — 스윗스팟(N_UF~5254) 근방. 좋은 영역 밀도 확보.
  uniform  — U(3500,6000) 광역 탐색. 가치 지형 학습용.
  reactive — 밀도 반응형(혼잡↑ → 조임). ★미검증 가설(상태-반응형 스케줄링) 시연.

교훈 반영: (1) 에피소드마다 증분 저장 — 죽어도 데이터 보존.
          (2) wall-clock 가드 — congested 폭증(26h 사례) 시 에피소드 중단.

usage: python rl_leader/collect_parallel.py --seed 0 --episodes 20 --out rl_leader/data/w00.npz
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv, make_random_scenario

NP_LO, NP_HI = 0.0, 2200.0
NUF_LO, NUF_HI = 0.0, 6000.0
SWEET_NUF, SWEET_NP = 5254.0, 1161.0


def pick_mode(rng):
    return str(rng.choice(["sweet", "uniform", "reactive"], p=[0.35, 0.35, 0.30]))


def behavior_action(mode, obs, rng):
    """모드별 (N_P, N_UF). obs[5]=rho_max/rho_crit (env._observe 규약)."""
    rho_max = float(obs[5]) if len(obs) > 5 else 1.0
    if mode == "sweet":
        n_uf = np.clip(rng.normal(SWEET_NUF, 500.0), 4200.0, NUF_HI)
        n_p = np.clip(rng.normal(SWEET_NP, 400.0), NP_LO, NP_HI)
    elif mode == "uniform":
        n_uf = rng.uniform(3500.0, NUF_HI)
        n_p = rng.uniform(NP_LO, NP_HI)
    else:  # reactive — 혼잡 커지면 유입 조임(가설 시연)
        tight = float(np.clip((rho_max - 0.80) / 0.60, 0.0, 1.0))
        n_uf = np.clip(6000.0 - 1800.0 * tight + rng.normal(0, 250.0), 3500.0, NUF_HI)
        n_p = np.clip(SWEET_NP + rng.normal(0, 350.0), NP_LO, NP_HI)
    return float(n_p), float(n_uf)


def save(out: Path, buf: dict, meta: list):
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             obs=np.array(buf["obs"], np.float32),
             act=np.array(buf["act"], np.float32),        # 정규화 [-1,1]^2
             budget=np.array(buf["budget"], np.float32),  # raw (N_P, N_UF)
             rew=np.array(buf["rew"], np.float32),
             next_obs=np.array(buf["next_obs"], np.float32),
             done=np.array(buf["done"], np.float32),
             ep=np.array(buf["ep"], np.int32),
             meta=np.array(meta, dtype=object) if meta else np.array([]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ep-sec", type=float, default=1800.0, help="에피소드 wall-clock 가드")
    ap.add_argument("--T", type=float, default=14400.0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    buf = {k: [] for k in ["obs", "act", "budget", "rew", "next_obs", "done", "ep"]}
    meta = []
    t_start = time.time()

    for ep in range(a.episodes):
        scen = make_random_scenario(rng)
        mode = pick_mode(rng)
        try:
            env = RLLeaderEnv(scenario_dict=scen, T_total=a.T)
        except Exception as e:
            print(f"[ep{ep}] env 생성 실패: {e}", flush=True)
            continue
        obs = env.reset()
        t_ep = time.time()
        n_step, aborted = 0, False
        while True:
            n_p, n_uf = behavior_action(mode, obs, rng)
            act = env.budget_to_action(n_p, n_uf)
            try:
                nobs, rew, done, info = env.step(act)
            except Exception as e:
                print(f"[ep{ep}] step 실패: {e}", flush=True)
                break
            buf["obs"].append(obs.tolist()); buf["act"].append(act.tolist())
            buf["budget"].append([n_p, n_uf]); buf["rew"].append(float(rew))
            buf["next_obs"].append(nobs.tolist()); buf["done"].append(float(done))
            buf["ep"].append(ep)
            obs = nobs; n_step += 1
            if done:
                break
            if time.time() - t_ep > a.max_ep_sec:   # ★ 폭증 가드
                aborted = True
                break
        meta.append({"ep": ep, "mode": mode, "demand": float(scen.get("urban_scale", 0)),
                     "stressor": ("incident" if "freeway_lane_closures" in scen
                                  else "skew" if "urban_west_east_ratio" in scen else "none"),
                     "steps": n_step, "aborted": aborted,
                     "cum_ttt": float(env.sim.total_ttt), "sec": round(time.time() - t_ep, 1)})
        save(out, buf, meta)   # ★ 에피소드마다 증분 저장
        print(f"[ep{ep}] mode={mode} d={scen.get('urban_scale',0):.2f} steps={n_step}"
              f"{' ABORT' if aborted else ''} ttt={env.sim.total_ttt:.0f} "
              f"{time.time()-t_ep:.0f}s | 누적샘플={len(buf['obs'])} 총{time.time()-t_start:.0f}s", flush=True)

    print(f"DONE seed={a.seed} 샘플={len(buf['obs'])} → {out}", flush=True)


if __name__ == "__main__":
    main()
