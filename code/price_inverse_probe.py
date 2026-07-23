# price 정적 역산 probe(2026-07-23) — follower가 per-ramp 가격에 반응/선택하나 + P-CENT 목표 실현가능성
"""핵심 질문(realizability): budget 고정 상태에서 per-ramp metering 가격을 내리면
follower의 per-ramp metering이 (a) 반응하나 (b) 선택적인가(해당 램프만) (c) trust_frac이
얼마나 조이나 (d) P-CENT의 per-ramp split에 닿을 수 있나.

반응하면 → price 축에 헤드룸 실재 → 학습 가치. 안 하면 → 가격 impotent(구조적 한계).

peak 상태를 RLLeaderEnv로 잡고(budget=sweet spot), follower.solve에 가격을 주입해 응답곡선을 잰다.
usage: python rl_leader/price_inverse_probe.py
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rl_leader.env import RLLeaderEnv
from src.controllers.leader import LeaderAction

RAMPS = ["R_D_W", "R_F_W", "R_D_E", "R_F_E"]
NP, NUF = 1161.0, 5254.0          # sweet-spot budget 고정


def pcent_target(step):
    """P-CENT skew 런의 per-ramp metering release(해당 step)."""
    p = ROOT / "outputs/_wang3/ho_pcent_skew/P-CENT/run_log.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, newline="", encoding="utf-8", errors="ignore")))
    for r in rows:
        if int(float(r["step"])) == step:
            return {rp: float(r.get(f"ramp_metering_release_actual_{rp}_veh", "nan")) for rp in RAMPS}
    return None


def solve_metering(env, state, prev, forecast, la, prices, ref, trust):
    f = env.follower
    f.metering_marginal_price = prices          # dict or None
    f.metering_marginal_price_ref = ref or {}
    f.metering_marginal_price_trust_frac = trust
    ctrl = f.solve(state.copy(), la, forecast, prev).control
    f.metering_marginal_price = None            # 오염 방지
    return {rp: float(ctrl.ramp_metering.get(rp, float("nan"))) for rp in RAMPS}


def probe_at(env, peak_step):
    """env를 peak_step까지 전진 후 가격 응답 측정."""
    env.reset()
    a = env.budget_to_action(NP, NUF)
    while env.step_idx < peak_step:
        env.step(a)
    state = env.sim.state
    prev = env.previous
    forecast = env._forecast()
    la = LeaderAction(NP, NUF)
    op = {rp: float(prev.ramp_metering.get(rp, 0.0)) for rp in RAMPS}
    base = solve_metering(env, state, prev, forecast, la, None, None, None)
    print(f"\n=== step {env.step_idx} (t={env.step_idx*env.dt:.0f}s) ===", flush=True)
    tgt = pcent_target(env.step_idx)
    print(f"  기준 metering(가격無): " + "  ".join(f"{rp}={base[rp]:.1f}" for rp in RAMPS), flush=True)
    if tgt:
        print(f"  P-CENT 목표      : " + "  ".join(f"{rp}={tgt[rp]:.1f}" for rp in RAMPS), flush=True)
    # 응답곡선: R_D_W 가격만 sweep, trust 2종
    for trust in [0.20, 0.60]:
        print(f"  -- R_D_W 가격 sweep (trust_frac={trust}) [자기응답 + 교차] --", flush=True)
        for pv in [-2000.0, -500.0, 0.0, 500.0, 2000.0]:
            m = solve_metering(env, state, prev, forecast, la,
                               {"R_D_W": pv, "R_F_W": 0.0, "R_D_E": 0.0, "R_F_E": 0.0}, op, trust)
            d = {rp: m[rp] - base[rp] for rp in RAMPS}
            print(f"     p_RDW={pv:+7.0f}  R_D_W={m['R_D_W']:6.1f}(Δ{d['R_D_W']:+5.1f})  "
                  f"타램프Δ: RFW{d['R_F_W']:+.1f} RDE{d['R_D_E']:+.1f} RFE{d['R_F_E']:+.1f}", flush=True)


def main():
    env = RLLeaderEnv(scenario_name="sweet_190_skew15_w60")
    print("=== price 정적 역산 probe: sweet_190_skew15_w60 (budget 고정 N_UF=5254) ===", flush=True)
    print("질문: per-ramp 가격 → follower metering 반응/선택성/trust 영향/P-CENT 실현가능성", flush=True)
    for st in [12, 20]:      # far-gate ON 구간(step12~39)
        probe_at(env, st)
    print("\n판정: 자기응답Δ가 크고 교차Δ가 작으면 → 선택적 가격 가능(헤드룸 실재).", flush=True)
    print("      trust 0.6서 응답이 커지면 → trust_frac이 병목(튜닝으로 완화 가능).", flush=True)


if __name__ == "__main__":
    main()
