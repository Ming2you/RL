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

if r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv
from rl_leader.eval_holdout import windowed_from_csv, WARM
from rl_leader.nets import Actor

CELLS = [("190-skew", "sweet_190_skew15_w60", "skew"),
         ("190-incident", "sweet_190_incident_w60", "inc")]
BASE = [("NC", "ho_nc"), ("PFO", "ho_pfo"), ("P-Stack", "ho_pstack"), ("P-CENT", "ho_pcent")]
CTRL = {"ho_nc": "NO-CONTROL", "ho_pfo": "WU-FAITHFUL-FOLLOWER",
        "ho_pstack": "P-STACK-WU-FAITHFUL-ALLPRICE-JOINT", "ho_pcent": "P-CENT"}


def baselines(tag):
    out = {}
    for label, pref in BASE:
        p = ROOT / "outputs" / "_wang3" / f"{pref}_{tag}" / CTRL[pref] / "run_log.csv"
        out[label] = windowed_from_csv(p)
    return out


def rollout(actor, scen, max_sec):
    env = RLLeaderEnv(scenario_name=scen)
    obs = env.reset()
    warm = float(env.sim.total_ttt)
    t0 = time.time()
    tr = {k: [] for k in ["t", "N_P", "N_UF", "step_ttt", "cum_ttt", "urban_ttt", "fw_ttt",
                          "rho_mean", "rho_max", "ramp_q", "urban_veh", "fw_veh"]}
    done, truncated, n = False, False, 0
    while not done:
        a = actor.act(obs, deterministic=True)
        t_sec = env.step_idx * env.dt
        obs, _, done, info = env.step(a)
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
    a = ap.parse_args()

    sd = torch.load(a.policy)
    actor = Actor(sd["obs_mu"].shape[0], sd["mu.bias"].shape[0])
    actor.load_state_dict(sd); actor.eval()
    print(f"=== held-out 평가: {Path(a.policy).name} (가드 {a.max_sec:.0f}s/셀) ===", flush=True)

    for disp, scen, tag in CELLS:
        b = baselines(tag)
        print(f"\n--- {disp} ---", flush=True)
        print("  기준: " + " | ".join(f"{k} {v:.0f}" for k, v in b.items() if v), flush=True)
        try:
            r = rollout(actor, scen, a.max_sec)
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
            keys = ["t", "N_P", "N_UF", "step_ttt", "cum_ttt", "urban_ttt", "fw_ttt",
                    "rho_mean", "rho_max", "ramp_q", "urban_veh", "fw_veh"]
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
