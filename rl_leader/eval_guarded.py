# 학습 정책 held-out 평가(2026-07-30) — wall-clock 가드 + budget 스케줄 진단
"""IQL 정책을 190-skew/190-incident(학습분포 밖)에서 평가. 나쁜 정책은 혼잡 폭증으로
롤아웃이 수시간 걸리므로 **시간 가드** 필수(절단 시 비교불가로 명시 — 조용히 숫자 내지 않음).

TTT 외에 정책이 고른 N_UF 궤적을 함께 보고한다. 정적 budget 천장(6325)은 이미 닫혔으므로,
이득이 나온다면 **상태-반응형 스케줄링**에서 와야 한다 — 그 증거를 직접 확인한다.

usage: python rl_leader/eval_guarded.py rl_leader/actor_iql.pt [--max-sec 3600]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

if Path(r"C:/torchlib").is_dir() and r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")   # Windows 긴경로 우회 설치일 때만(HANDOFF §7.2)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv, RAMPS, SIGNALS


def env_signals():
    return SIGNALS
from rl_leader.eval_holdout import windowed_from_csv, WARM
from rl_leader.nets import Actor

CELLS = [("190-skew", "sweet_190_skew15_w60", "skew"),
         ("190-incident", "sweet_190_incident_w60", "inc")]
BASE = [("NC", "ho_nc"), ("PFO", "ho_pfo"), ("P-Stack", "ho_pstack"), ("P-CENT", "ho_pcent")]


def baselines(tag):
    out = {}
    for label, pref in BASE:
        p = ROOT / "data" / "holdout" / f"{pref}_{tag}.csv"
        out[label] = windowed_from_csv(p)
    return out


def rollout(actor, scen, max_sec, price_action=False, zero_price=False,
            price_level=False, omega_action=False, fix_omega=False, price_nonneg=False,
            green_action=False, zero_green=False):
    """zero_price: 가격 성분만 0으로 강제(ablation). fix_omega: ω를 0.5로 강제(ablation).
    두 ablation으로 '가격 축'과 'ω 축'의 기여를 각각 귀속할 수 있다."""
    env = RLLeaderEnv(scenario_name=scen, price_action=price_action,
                      price_level=price_level, omega_action=omega_action,
                      green_action=green_action)
    obs = env.reset()
    n_green = len(env_signals()) if green_action else 0
    warm = float(env.sim.total_ttt)
    t0 = time.time()
    tr = {k: [] for k in ["t", "N_P", "N_UF", "step_ttt", "cum_ttt", "urban_ttt", "fw_ttt",
                          "rho_mean", "rho_max", "ramp_q", "urban_veh", "fw_veh"]}
    if price_action:      # Phase 4: per-ramp 가격·실현 release도 트레이스(공간 분배 분석용)
        for rp in RAMPS:
            tr[f"price_{rp}"] = []
            tr[f"rel_{rp}"] = []
    done, truncated, n = False, False, 0
    while not done:
        a = actor.act(obs, deterministic=True)
        if zero_price and price_action:
            a = np.concatenate([a[:2], np.zeros(4, dtype=a.dtype), a[6:]])
        if price_nonneg and price_action:
            # 음수 가격 = budget 위로 방류 증가. price_level에선 budget이 soft anchor라
            # 이게 방류를 cap까지 밀어올려 제어가 소멸한다(실측: peak 방류 6000 = NC 수준).
            # 가격을 억제 방향으로만 제한하면 N_UF가 상한으로서 권위를 회복한다.
            a = np.concatenate([a[:2], np.clip(a[2:6], 0.0, 1.0), a[6:]])
        if fix_omega and omega_action:
            oi = 2 + (4 if price_action else 0)
            a = np.concatenate([a[:oi], np.zeros(1, dtype=a.dtype), a[oi + 1:]])  # ω=중앙
        if zero_green and green_action:
            a = np.concatenate([a[:len(a) - n_green], np.zeros(n_green, dtype=a.dtype)])
        t_sec = env.step_idx * env.dt
        obs, _, done, info = env.step(a)
        if price_action:
            for rp in RAMPS:
                tr[f"price_{rp}"].append(info["prices"][rp])
                tr[f"rel_{rp}"].append(info["ramp_release"][rp])
        try:
            rq = float(sum(env.sim.state.ramp_queue.values()))
        except Exception:
            rq = float("nan")
        tr["t"].append(t_sec); tr["N_P"].append(info["N_P"]); tr["N_UF"].append(info["N_UF"])
        tr["step_ttt"].append(info["step_ttt"]); tr["cum_ttt"].append(info["cum_ttt"])
        tr["urban_ttt"].append(info.get("urban_ttt", float("nan")))
        tr["fw_ttt"].append(info.get("freeway_ttt", float("nan")))
        tr["rho_mean"].append(float(obs[4])); tr["rho_max"].append(float(obs[5]))
        tr["ramp_q"].append(rq)
        tr["urban_veh"].append(float(obs[0]) * 1000.0); tr["fw_veh"].append(float(obs[1]) * 1000.0)
        n += 1
        if time.time() - t0 > max_sec:
            truncated = True
            break
    out = {k: np.array(v, np.float32) for k, v in tr.items()}
    out.update(ttt=float(env.sim.total_ttt) - warm, steps=n, truncated=truncated,
               sec=time.time() - t0, nuf=out["N_UF"], rho=out["rho_max"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--max-sec", type=float, default=3600.0)
    ap.add_argument("--trace-dir", default=None, help="per-step 트레이스 npz 저장(메커니즘 분석용)")
    ap.add_argument("--tag", default="", help="트레이스 파일 접두(시드 구분)")
    ap.add_argument("--cells", default="", help="평가할 셀 tag 쉼표구분(skew,inc). 생략=전부. "
                                               "롤아웃은 단일 스레드라 셀별로 쪼개 병렬 실행해도 안전")
    ap.add_argument("--zero-price", action="store_true",
                    help="가격 성분을 0으로 강제(ablation) — 가격 축의 기여 귀속")
    ap.add_argument("--fix-omega", action="store_true",
                    help="ω를 0.5로 강제(ablation) — 링크 배분 축의 기여 귀속")
    ap.add_argument("--zero-green", action="store_true",
                    help="green 가격을 0으로 강제(ablation) — urban 채널 기여 귀속")
    ap.add_argument("--price-nonneg", action="store_true",
                    help="가격을 억제 방향(>=0)으로만 제한 — budget이 상한 권위를 되찾나 검증")
    ap.add_argument("--price-level", action="store_true",
                    help="가격이 총량을 정하는 모드로 롤아웃(수집 때와 일치시킬 것)")
    a = ap.parse_args()

    cells = CELLS
    if a.cells:
        want = {c.strip() for c in a.cells.split(",") if c.strip()}
        unknown = want - {t for _, _, t in CELLS}
        if unknown:
            raise SystemExit(f"알 수 없는 셀 tag: {sorted(unknown)} (가능: {[t for _, _, t in CELLS]})")
        cells = [c for c in CELLS if c[2] in want]

    sd = torch.load(a.policy)
    od, ad = sd["obs_mu"].shape[0], sd["mu.bias"].shape[0]
    actor = Actor(od, ad)
    actor.load_state_dict(sd); actor.eval()
    # 액션 차원으로 모드 역추론: 2=budget, 6=+가격, 7=+ω, 11=+green(ω 없음), 12=+ω+green
    price = ad >= 6
    green = ad in (11, 12)
    omega = ad in (7, 12)
    print(f"=== held-out 평가: {Path(a.policy).name} (가드 {a.max_sec:.0f}s/셀, "
          f"셀 {[t for _, _, t in cells]}, act_dim={ad} price={price} omega={omega} "
          f"level={a.price_level} | ablation zero_price={a.zero_price} fix_omega={a.fix_omega}) ===",
          flush=True)

    for disp, scen, tag in cells:
        b = baselines(tag)
        print(f"\n--- {disp} ---", flush=True)
        print("  기준: " + " | ".join(f"{k} {v:.0f}" for k, v in b.items() if v), flush=True)
        try:
            r = rollout(actor, scen, a.max_sec, price_action=price, zero_price=a.zero_price,
                        price_level=a.price_level, omega_action=omega, fix_omega=a.fix_omega,
                        price_nonneg=a.price_nonneg, green_action=green, zero_green=a.zero_green)
        except Exception as e:
            print(f"  ROLLOUT FAIL: {e}", flush=True)
            continue
        if r["truncated"]:
            print(f"  ⚠️ 시간가드 절단 ({r['steps']}스텝, {r['sec']:.0f}s) — TTT 비교불가."
                  f" 부분 TTT={r['ttt']:.0f} (정책이 혼잡을 유발한 신호)", flush=True)
        else:
            ps, pc = b.get("P-Stack"), b.get("P-CENT")
            print(f"  ✅ 완주 {r['steps']}스텝 {r['sec']:.0f}s → windowed TTT = {r['ttt']:.1f}", flush=True)
            if ps: print(f"     vs P-Stack {ps:.0f}: {r['ttt']-ps:+.1f}", flush=True)
            if pc: print(f"     vs P-CENT  {pc:.0f}: {r['ttt']-pc:+.1f}", flush=True)
        if a.trace_dir:   # 메커니즘 분석용 per-step 트레이스
            td = Path(a.trace_dir); td.mkdir(parents=True, exist_ok=True)
            keys = [k for k in r if isinstance(r[k], np.ndarray) and k not in ("nuf", "rho")]
            np.savez(td / f"trace{('_' + a.tag) if a.tag else ''}_{tag}.npz",
                     **{k: r[k] for k in keys}, ttt=r["ttt"], truncated=float(r["truncated"]))
            print(f"     트레이스 저장 → {td.name}/trace{('_'+a.tag) if a.tag else ''}_{tag}.npz", flush=True)
        # budget 스케줄 진단 — 상태-반응형인가(정적이면 std≈0, rho와 상관 없음)
        nuf, rho = r["nuf"], r["rho"]
        if len(nuf) > 3:
            c = float(np.corrcoef(nuf, rho)[0, 1]) if np.std(nuf) > 1e-6 and np.std(rho) > 1e-6 else float("nan")
            print(f"     N_UF: mean={nuf.mean():.0f} std={nuf.std():.0f} "
                  f"min={nuf.min():.0f} max={nuf.max():.0f} | corr(N_UF, rho_max)={c:+.3f}", flush=True)
            print(f"     → {'상태-반응형(std 큼)' if nuf.std() > 200 else '거의 정적(std 작음)'}"
                  f", {'혼잡↑에 조임(음의 상관)' if c < -0.2 else '혼잡과 무관/역방향' if c == c else ''}", flush=True)


if __name__ == "__main__":
    main()
