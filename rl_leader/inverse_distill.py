# P-CENT distillation(2026-07-23) — P-CENT를 목표로 (budget+역산가격) 폐루프 추종 + teacher 수집
"""RL을 P-CENT 기반으로 짓는 첫 조각. P-CENT의 (총유입, per-ramp split)을 목표로
follower에 budget(총량) + per-ramp 가격(분배)을 내려 재현시킨다. 동시에:
  (a) realizability — windowed TTT가 P-CENT(5757)에 얼마나 닿나 + 램프큐 폭발 여부.
  (b) teacher 데이터 — (obs 13-dim → budget[N_P,N_UF] + price[4 ramp]) 수집 → npz.

수집된 데이터가 좋은 teacher가 되려면 follower가 실제로 P-CENT에 근접해야(폭발 X).
그 판정을 이 스크립트가 내린다.

usage: python rl_leader/inverse_distill.py [cell=skew|inc] [--noprice]
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csv
import numpy as np
from rl_leader.env import RLLeaderEnv

RAMPS = ["R_D_W", "R_F_W", "R_D_E", "R_F_E"]
MERGES = {"W": ["R_D_W", "R_F_W"], "E": ["R_D_E", "R_F_E"]}
NP = 1161.0
PRICE_MAG = 1500.0
BASE = {"skew": {"P-CENT": 5757, "PFO": 6299, "P-Stack": 6379, "NC": 6882},
        "inc":  {"P-CENT": 8016, "PFO": 9230, "P-Stack": 8386, "NC": 8556}}
SCEN = {"skew": "sweet_190_skew15_w60", "inc": "sweet_190_incident_w60"}


def load_pcent(cell):
    p = ROOT / "data" / "pcent_teacher" / cell / "run_log.csv"
    r = list(csv.DictReader(open(p, newline="", encoding="utf-8", errors="ignore")))
    tot, split = {}, {}
    for x in r:
        k = int(float(x["step"]))
        v = {rp: float(x.get(f"ramp_metering_release_actual_{rp}_veh", 0) or 0) * 20.0 for rp in RAMPS}
        split[k] = v
        tot[k] = sum(v.values())
    return tot, split


def ramp_queue(state):
    try:
        return float(sum(state.ramp_queue.values()))
    except Exception:
        return 0.0


def run(cell, use_price=True):
    scen = SCEN[cell]
    pcent_tot, pcent_split = load_pcent(cell)
    env = RLLeaderEnv(scenario_name=scen)
    env.reset()
    warm = float(env.sim.total_ttt)
    OBS, BUD, PRC = [], [], []
    maxq = 0.0
    done = False
    while not done:
        k = env.step_idx
        obs = env._observe()
        nuf = max(0.0, min(6000.0, float(pcent_tot.get(k, 5254.0))))
        prices = {rp: 0.0 for rp in RAMPS}
        if use_price:
            op = {rp: float(env.previous.ramp_metering.get(rp, 0.0)) for rp in RAMPS} if env.previous else {}
            tgt = pcent_split.get(k, {})
            if len(tgt) == 4:
                for m, rps in MERGES.items():
                    harder = min(rps, key=lambda rp: tgt.get(rp, 1e9))  # P-CENT가 더 조인 램프
                    prices[harder] = PRICE_MAG
            env.follower.metering_marginal_price = prices
            env.follower.metering_marginal_price_ref = op
            env.follower.metering_marginal_price_trust_frac = 0.20
        else:
            env.follower.metering_marginal_price = None
        OBS.append(obs.tolist()); BUD.append([NP, nuf]); PRC.append([prices[rp] for rp in RAMPS])
        env.step(env.budget_to_action(NP, nuf))
        env.follower.metering_marginal_price = None
        q = ramp_queue(env.sim.state); maxq = max(maxq, q)
        import os as _o
        if _o.environ.get("DISTILL_TRACE"):
            print(f"  step {k:2d}  N_UF={nuf:6.0f}  ramp_q={q:8.1f}  cum_ttt={env.sim.total_ttt:9.1f}", flush=True)
            if _o.environ.get("DISTILL_MAXSTEP") and env.step_idx >= int(_o.environ["DISTILL_MAXSTEP"]):
                print("  (MAXSTEP 도달 — 조기 종료)", flush=True); break
    ttt = float(env.sim.total_ttt) - warm
    tag = "price" if use_price else "budgetonly"
    out = ROOT / "data" / "bc" / f"distill_{cell}_{tag}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, obs=np.array(OBS, np.float32), budget=np.array(BUD, np.float32),
             price=np.array(PRC, np.float32), ttt=ttt)
    return ttt, maxq, str(out)


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "skew"
    use_price = "--noprice" not in sys.argv
    b = BASE[cell]
    print(f"=== P-CENT distillation: {cell} (price={use_price}) ===", flush=True)
    print(f"기준: " + " | ".join(f"{k} {v}" for k, v in b.items()), flush=True)
    ttt, maxq, out = run(cell, use_price)
    print(f"\nwindowed TTT = {ttt:.1f}  (P-CENT {b['P-CENT']} 대비 {ttt-b['P-CENT']:+.1f})", flush=True)
    print(f"max 램프큐 = {maxq:.1f} veh  ({'폭발' if maxq > 2000 else 'OK'})", flush=True)
    print(f"teacher 데이터 저장 → {out}", flush=True)
    if ttt <= b["PFO"]:
        print("→ PFO 이하 도달: distillation 유효, teacher 데이터 양호.", flush=True)
    elif maxq > 2000:
        print("→ 램프큐 폭발: 이 budget/price로 재현 실패(realizability 부정적).", flush=True)
    else:
        print("→ P-Stack~PFO 사이: 부분 성공, 가격 정교화 여지.", flush=True)


if __name__ == "__main__":
    main()
