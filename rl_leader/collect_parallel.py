# 병렬 offline RL 데이터 수집(2026-07-30, Phase 4 가격 확장 2026-07-31) — 도메인 랜덤화 + 행동 다양성 + 증분 저장
"""375샘플 병목 해소용. RLLeaderEnv(리더 탐색 우회, budget+가격 직접 주입)로 다양한 행동을
굴려 (obs, action, reward, next_obs) 전이를 모은다. offline RL(IQL/AWAC)이 teacher를
넘으려면 teacher 궤적만으론 안 되고 행동 대비(counterfactual)가 필요하다.

행동 정책(에피소드마다 budget 모드 × price 모드 독립 선택 + 스텝 노이즈):
  budget — sweet(스윗스팟 근방) / uniform(광역, N_P 음수 포함) / reactive(밀도 반응형)
  price  — zero(budget-only 앵커) / iid(광역) / contrast(merge 내 반대부호 쌍 = 재분배 축)

교훈 반영: (1) 에피소드마다 증분 저장 — 죽어도 데이터 보존.
          (2) wall-clock 가드 — congested 폭증(26h 사례) 시 에피소드 중단.

Phase 4 데이터는 act 6차원·obs 23차원 → 기존 data/rl_dataset/(2차원)과 섞지 말 것.
usage: python rl_leader/collect_parallel.py --seed 100 --episodes 40 --out data/rl_dataset_p6/w00.npz
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import (RLLeaderEnv, make_random_scenario, RAMPS, PRICE_MAG,
                           PRICE_LEVEL_MAG, OMEGA_LO, OMEGA_HI, SIGNALS, GREEN_PRICE_MAG)

NP_LO, NP_HI = -1000.0, 2200.0    # 기본(구 Phase 4). --np-wide면 env 박스에서 다시 읽는다.
NUF_LO, NUF_HI = 0.0, 6000.0
SWEET_NUF, SWEET_NP = 5254.0, 1161.0


HOLD_MIN, HOLD_MAX = 8, 20        # P-CENT는 낮은 총량을 19스텝 연속 유지한다
CLAMP_LO, CLAMP_HI = 4400.0, 5300.0   # 지지가 0이던 밴드(P-CENT 운영점 부근)


ONPOL_SIGMAS = (0.05, 0.15, 0.30)   # 탐색 노이즈 층화 — σ=0이면 대비가 없어 advantage 소멸(§4.1)


def load_policy(path):
    """체크포인트 → act(obs) 함수. torch는 이 모드에서만 필요하므로 지연 import."""
    import sys as _sys
    from pathlib import Path as _P
    if _P(r"C:/torchlib").is_dir() and r"C:/torchlib" not in _sys.path:
        _sys.path.insert(0, r"C:/torchlib")
    import torch
    from rl_leader.nets import Actor
    sd = torch.load(path)
    actor = Actor(sd["obs_mu"].shape[0], sd["mu.bias"].shape[0])
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def pick_mode(rng):
    """★2026-08-01 재설계. 이전 i.i.d. 재추출은 '낮고 평평한 clamp'를 데이터에 한 번도
    만들지 못했다(peak std<218 에피소드 0/560, [4550,4900] 연속 최장 6스텝 vs P-CENT 19스텝)
    → offline RL이 P-CENT식 정책을 원리적으로 재현 불가 = SATURATED 진단의 원인.
    이제 전 모드가 piecewise-constant 유지(hold 8~20스텝)를 기본으로 한다."""
    return str(rng.choice(["clamp", "hold_wide", "reactive"], p=[0.40, 0.35, 0.25]))


def pick_price_mode(rng):
    """zero = 앵커(가격 없는 지지). hold = 시간상관 레벨가격. contrast = merge 내 반대부호."""
    return str(rng.choice(["zero", "hold", "contrast"], p=[0.25, 0.45, 0.30]))


def _segments(rng, n, sampler, hold_lo=HOLD_MIN, hold_hi=HOLD_MAX):
    """piecewise-constant 스케줄 생성 — 각 구간은 sampler()가 뽑은 값을 hold 스텝 유지."""
    out = []
    while len(out) < n:
        out += [sampler()] * int(rng.integers(hold_lo, hold_hi + 1))
    return out[:n]


def make_budget_schedule(mode, rng, n, np_lo=NP_LO, np_hi=NP_HI):
    """에피소드 전체의 (N_P, N_UF) 스케줄. reactive만 런타임 상태 의존(런타임에 덮어씀).
    np_lo/np_hi는 env의 실제 박스를 받는다 — 모듈 상수를 쓰면 --np-wide로 넓힌 영역을
    수집기가 영영 안 뽑아 지지가 0이 된다(액션 축 확장 때 반복된 실패 패턴)."""
    if mode == "clamp":     # ★지지 결손 밴드를 길게 유지 — P-CENT 운영점 재현용
        nuf = _segments(rng, n, lambda: float(rng.uniform(CLAMP_LO, CLAMP_HI)))
    elif mode == "hold_wide":
        nuf = _segments(rng, n, lambda: float(rng.uniform(3500.0, NUF_HI)))
    else:                   # reactive: 베이스라인만 잡고 런타임에 밀도로 보정
        nuf = _segments(rng, n, lambda: float(rng.uniform(4500.0, NUF_HI)))
    n_p = _segments(rng, n, lambda: float(rng.uniform(np_lo, np_hi)))
    return np.array(n_p), np.array(nuf)


def make_price_schedule(price_mode, rng, n, mag):
    """per-ramp 가격 스케줄(veh/h, RAMPS 순서). 레벨가격은 시간상관이 필수 —
    매 스텝 부호가 바뀌면 follower 입장에선 잡음이라 총량 궤적이 안 만들어진다."""
    if price_mode == "zero":
        return np.zeros((n, 4))
    if price_mode == "hold":     # 4램프 공통 레벨 + 램프별 소폭 차등(총량 채널 위주)
        base = _segments(rng, n, lambda: float(rng.uniform(-mag, mag)))
        jitter = _segments(rng, n, lambda: rng.normal(0, mag * 0.15, size=4))
        return np.clip(np.array(base)[:, None] + np.array(jitter), -mag, mag)
    # contrast: merge 내 반대부호 쌍(배분 채널) — 레벨 채널과 분리해 귀속 가능하게
    def _pair():
        p_w, p_e = rng.uniform(-mag, mag), rng.uniform(-mag, mag)
        return np.array([p_w, -p_w, p_e, -p_e])
    return np.clip(np.array(_segments(rng, n, _pair)), -mag, mag)


def make_omega_schedule(rng, n):
    """링크 배분 스케줄 — [0,1] 정규화 위치(0=가용창 하한, 1=상한).
    창이 N_UF에 따라 좁아지므로(cap 누수) 절대 ω가 아니라 위치로 뽑아야 예산을 안 버린다.
    기존 데이터의 ω 분산은 정확히 0이라 이 축은 지지가 전무했다."""
    return np.array(_segments(rng, n, lambda: float(rng.uniform(0.0, 1.0))))


def pick_green_mode(rng):
    """zero = green 가격 없는 지지(기존 분포와 겹침). uniform = 전 신호 공통(도시 전체 조임/풂).
    diff = 신호별 차등(공간 타겟팅). probe상 개별 감도가 제각각이라 두 형태를 모두 깔아둔다."""
    return str(rng.choice(["zero", "uniform", "diff"], p=[0.30, 0.35, 0.35]))


def make_green_schedule(green_mode, rng, n, mag):
    """per-signal green 가격 스케줄. 매 스텝 부호가 바뀌면 신호가 진동만 하므로 hold 필수."""
    k = len(SIGNALS)
    if green_mode == "zero":
        return np.zeros((n, k))
    if green_mode == "uniform":
        base = _segments(rng, n, lambda: float(rng.uniform(-mag, mag)))
        return np.clip(np.repeat(np.array(base)[:, None], k, axis=1), -mag, mag)
    return np.clip(np.array(_segments(rng, n, lambda: rng.uniform(-mag, mag, size=k))), -mag, mag)


def reactive_nuf(base, obs, rng):
    """혼잡 반응 보정(clamp/hold의 유지 성질은 깨지 않도록 base 주변 ±)."""
    rho_max = float(obs[5]) if len(obs) > 5 else 1.0
    tight = float(np.clip((rho_max - 0.80) / 0.60, 0.0, 1.0))
    return float(np.clip(base - 1200.0 * tight + rng.normal(0, 80.0), 3500.0, NUF_HI))


def save(out: Path, buf: dict, meta: list):
    """원자적 증분 저장 — tmp에 쓰고 os.replace로 교체. 직접 덮어쓰면 저장 도중 kill될 때
    zip이 잘려 그 워커의 누적분 전체가 날아간다(watchdog kill/Ctrl-C 실존 위험)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp,
             obs=np.array(buf["obs"], np.float32),
             act=np.array(buf["act"], np.float32),        # 정규화 [-1,1]^action_dim
             budget=np.array(buf["budget"], np.float32),  # raw (N_P, N_UF)
             prices=np.array(buf["prices"], np.float32),  # raw per-ramp 가격(veh/h, RAMPS 순서)
             omega=np.array(buf["omega"], np.float32),    # raw ω 위치(가용창 내 [0,1])
             green=np.array(buf["green"], np.float32),    # raw per-signal green 가격
             rew=np.array(buf["rew"], np.float32),
             next_obs=np.array(buf["next_obs"], np.float32),
             done=np.array(buf["done"], np.float32),
             ep=np.array(buf["ep"], np.int32),
             meta=np.array(meta, dtype=object) if meta else np.array([]))
    os.replace(tmp, out)   # 원자적 교체(같은 볼륨이라 rename 보장)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ep-sec", type=float, default=1800.0, help="에피소드 wall-clock 가드")
    ap.add_argument("--T", type=float, default=14400.0)
    ap.add_argument("--policy", default=None,
                    help="on-policy 수집: 이 체크포인트를 굴리고 탐색 노이즈를 얹는다. "
                         "SATURATED(같은 분포에서 더 뽑아도 무익)의 정석 대응 — 학습된 정책이 "
                         "실제 방문하는 상태가 데이터에 없던 분포 이동 문제를 메운다.")
    ap.add_argument("--onpol-frac", type=float, default=0.7,
                    help="정책 롤아웃 에피소드 비율. 나머지는 기존 스케줄 행동정책(광역 커버리지 유지)")
    ap.add_argument("--np-wide", action="store_true",
                    help="N_P 박스를 [-3500, 2500]으로 확장. P-Stack leader의 실제 탐색범위와 "
                         "맞춘다(intent −3315까지, 그 스텝에서 λ_P 포화). 액션 차원은 2 그대로.")
    ap.add_argument("--price", action="store_true",
                    help="per-ramp 가격 4차원을 액션에 포함(obs도 23차원으로 확장). "
                         "기본 OFF — 가격 축은 4구성 실측에서 전부 기각됐다(HANDOFF §12.9). "
                         "Phase 4~7 재현 시에만 켤 것.")
    ap.add_argument("--omega", action="store_true", help="링크 배분 ω를 액션에 포함")
    ap.add_argument("--green", action="store_true", help="per-signal urban green 가격을 액션에 포함")
    ap.add_argument("--price-level", action="store_true",
                    help="가격이 총량을 정하는 모드(follower.metering_price_split=False). "
                         "이게 없으면 가격은 merge 내부 배분만 바꾼다(총량 변동 0.0 실측)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    buf = {k: [] for k in ["obs", "act", "budget", "prices", "omega", "green", "rew",
                           "next_obs", "done", "ep"]}
    meta = []
    t_start = time.time()
    mag = PRICE_LEVEL_MAG if a.price_level else PRICE_MAG
    actor = load_policy(a.policy) if a.policy else None
    if actor is not None:
        print(f"on-policy 수집: {a.policy}  비율={a.onpol_frac}  σ층화={ONPOL_SIGMAS}", flush=True)

    for ep in range(a.episodes):
        scen = make_random_scenario(rng)
        mode = pick_mode(rng)
        price_mode = pick_price_mode(rng)
        green_mode = pick_green_mode(rng)
        try:
            env = RLLeaderEnv(scenario_dict=scen, T_total=a.T, price_action=a.price,
                              price_level=a.price_level, omega_action=a.omega,
                              green_action=a.green, np_wide=a.np_wide)
        except Exception as e:
            print(f"[ep{ep}] env 생성 실패: {e}", flush=True)
            continue
        obs = env.reset()
        on_pol = actor is not None and float(rng.random()) < a.onpol_frac
        sigma = float(rng.choice(ONPOL_SIGMAS)) if on_pol else 0.0
        n_plan = env.n_steps + 4
        np_sched, nuf_sched = make_budget_schedule(mode, rng, n_plan, env.np_lo, env.np_hi)
        pr_sched = make_price_schedule(price_mode, rng, n_plan, mag)
        om_sched = make_omega_schedule(rng, n_plan)
        gr_sched = make_green_schedule(green_mode, rng, n_plan, GREEN_PRICE_MAG)
        t_ep = time.time()
        n_step, aborted = 0, False
        while True:
            k = min(n_step, n_plan - 1)
            n_p, n_uf = float(np_sched[k]), float(nuf_sched[k])
            if mode == "reactive":
                n_uf = reactive_nuf(n_uf, obs, rng)
            prices, om, gp = pr_sched[k], float(om_sched[k]), gr_sched[k]
            if on_pol:
                # 정책 출력 + 층화된 탐색 노이즈. budget은 정책이 정하고 나머지 축은
                # (활성일 때) 스케줄을 유지해 그 축들의 대비를 잃지 않는다.
                act = np.clip(actor.act(obs, deterministic=True)
                              + rng.normal(0.0, sigma, size=env.action_dim), -1.0, 1.0)
                n_p = env.np_lo + (act[0] + 1.0) * 0.5 * (env.np_hi - env.np_lo)
                n_uf = env.nuf_lo + (act[1] + 1.0) * 0.5 * (env.nuf_hi - env.nuf_lo)
            else:
                act = env.compose_action(n_p, n_uf, prices, om, gp)
            try:
                nobs, rew, done, info = env.step(act)
            except Exception as e:
                print(f"[ep{ep}] step 실패: {e}", flush=True)
                break
            buf["obs"].append(obs.tolist()); buf["act"].append(act.tolist())
            buf["budget"].append([n_p, n_uf]); buf["prices"].append(list(prices))
            buf["omega"].append(om); buf["green"].append(list(gp))
            buf["rew"].append(float(rew))
            buf["next_obs"].append(nobs.tolist()); buf["done"].append(float(done))
            buf["ep"].append(ep)
            obs = nobs; n_step += 1
            if done:
                break
            if time.time() - t_ep > a.max_ep_sec:   # ★ 폭증 가드
                aborted = True
                break
        meta.append({"ep": ep, "mode": ("onpol" if on_pol else mode), "sigma": sigma,
                     "price_mode": price_mode, "green_mode": green_mode,
                     "demand": float(scen.get("urban_scale", 0)),
                     "stressor": ("incident" if "freeway_lane_closures" in scen
                                  else "skew" if "urban_west_east_ratio" in scen else "none"),
                     "steps": n_step, "aborted": aborted,
                     "cum_ttt": float(env.sim.total_ttt), "sec": round(time.time() - t_ep, 1)})
        save(out, buf, meta)   # ★ 에피소드마다 증분 저장
        _md = f"onpol(s={sigma:.2f})" if on_pol else mode
        print(f"[ep{ep}] mode={_md}/{price_mode} d={scen.get('urban_scale',0):.2f} steps={n_step}"
              f"{' ABORT' if aborted else ''} ttt={env.sim.total_ttt:.0f} "
              f"{time.time()-t_ep:.0f}s | 누적샘플={len(buf['obs'])} 총{time.time()-t_start:.0f}s", flush=True)

    print(f"DONE seed={a.seed} 샘플={len(buf['obs'])} → {out}", flush=True)


if __name__ == "__main__":
    main()
