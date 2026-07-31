# IQL 이득 귀속 분석(2026-07-31) — 구간(peak/recovery) × 서브시스템(urban/freeway) 분해
"""IQL이 P-Stack을 이긴 이유를 데이터로 특정한다. corr(N_UF, rho)는 수요 프로파일에
교란돼 결론을 못 내므로, **어느 구간·어느 서브시스템에서 TTT가 줄었는지**를 직접 본다.

비교: IQL 트레이스(eval_guarded --trace-dir) vs P-Stack run_log(동일 셀).
창: warmup [0,900) 제외, peak [900,5220), recovery [5220,14400].

usage: python rl_leader/analyze_mechanism.py --trace-dir rl_leader/traces --tag s0
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = [("190-skew", "skew", "ho_pstack_skew"), ("190-incident", "inc", "ho_pstack_inc")]
CTRL = "P-STACK-WU-FAITHFUL-ALLPRICE-JOINT"
WARM_T, PEAK_END, T_END = 900.0, 5220.0, 14400.0


def pstack_series(pref):
    p = ROOT / "outputs" / "_wang3" / pref / CTRL / "run_log.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8", errors="ignore")))
    g = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    return dict(
        t=np.array([g(r, "time_sec") for r in rows]),
        urban=np.array([g(r, "step_urban_ttt") for r in rows]),
        fw=np.array([g(r, "step_freeway_ttt") for r in rows]),
        nuf=np.array([g(r, "leader_realized_N_UF_star") for r in rows]),
        rampq=np.array([g(r, "total_ramp_queue_end_veh") for r in rows]),
    )


def win_sum(t, v, lo, hi):
    m = (t >= lo) & (t < hi)
    return float(np.nansum(v[m])) if m.any() else float("nan")


def win_mean(t, v, lo, hi):
    m = (t >= lo) & (t < hi)
    return float(np.nanmean(v[m])) if m.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=str(ROOT / "rl_leader" / "traces"))
    ap.add_argument("--tag", default="s0")
    a = ap.parse_args()

    print("=== IQL 이득 귀속 분석 (IQL − P-Stack, 음수 = IQL 개선) ===", flush=True)
    for disp, tag, pref in CELLS:
        f = Path(a.trace_dir) / f"trace_{a.tag}_{tag}.npz"
        ps = pstack_series(pref)
        if not f.exists():
            print(f"\n--- {disp}: 트레이스 없음({f.name}) ---"); continue
        if ps is None:
            print(f"\n--- {disp}: P-Stack run_log 없음 ---"); continue
        d = np.load(f)
        t_r, u_r, w_r = d["t"], d["urban_ttt"], d["fw_ttt"]
        nuf_r, q_r = d["N_UF"], d["ramp_q"]

        print(f"\n--- {disp} ---", flush=True)
        print(f"  {'구간':10}{'urban Δ':>11}{'freeway Δ':>11}{'합 Δ':>10}", flush=True)
        tot = 0.0
        for lab, lo, hi in [("peak", WARM_T, PEAK_END), ("recovery", PEAK_END, T_END)]:
            du = win_sum(t_r, u_r, lo, hi) - win_sum(ps["t"], ps["urban"], lo, hi)
            dw = win_sum(t_r, w_r, lo, hi) - win_sum(ps["t"], ps["fw"], lo, hi)
            tot += du + dw
            print(f"  {lab:10}{du:+11.1f}{dw:+11.1f}{du+dw:+10.1f}", flush=True)
        print(f"  {'합계':10}{'':>11}{'':>11}{tot:+10.1f}", flush=True)

        print(f"  [제어] {'구간':10}{'N_UF(IQL)':>11}{'N_UF(PS)':>10}{'Δ':>9}"
              f"{'rampQ(IQL)':>12}{'rampQ(PS)':>11}", flush=True)
        for lab, lo, hi in [("peak", WARM_T, PEAK_END), ("recovery", PEAK_END, T_END)]:
            n1 = win_mean(t_r, nuf_r, lo, hi); n0 = win_mean(ps["t"], ps["nuf"], lo, hi)
            q1 = win_mean(t_r, q_r, lo, hi); q0 = win_mean(ps["t"], ps["rampq"], lo, hi)
            print(f"        {lab:10}{n1:11.0f}{n0:10.0f}{n1-n0:+9.0f}{q1:12.1f}{q0:11.1f}", flush=True)

    print("\n해석 지침: 이득이 특정 구간·서브시스템에 몰려 있으면 그것이 메커니즘.", flush=True)
    print("           N_UF Δ의 부호가 구간마다 다르면 = 시간 배분(스케줄링) 효과.", flush=True)


if __name__ == "__main__":
    main()
