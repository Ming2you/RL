# P-CENT 추종 테스트(2026-07-23) — budget(총량)+가격(분배)로 P-CENT 재현 가능성
"""핵심: P-CENT의 skew 우위는 '낮은 총유입을 큐 안 터지게 분배'. 이걸 follower가
(A) budget만 P-CENT 총량 추종 vs (B) +metering 가격으로 P-CENT per-ramp 분배 흉내
로 재현할 수 있나. B가 A의 폭발을 살리면 → budget총량+가격분배가 헤드룸의 열쇠.

usage: python rl_leader/pcent_follow_test.py
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv
from src.controllers.leader import LeaderAction

RAMPS = ["R_D_W", "R_F_W", "R_D_E", "R_F_E"]
MERGES = {"W": ["R_D_W", "R_F_W"], "E": ["R_D_E", "R_F_E"]}
NP = 1161.0
PRICE_MAG = 1500.0   # probe: +500이면 이미 swap 포화


def load_pcent():
    r = list(csv.DictReader(open(ROOT / "outputs/_wang3/ho_pcent_skew/P-CENT/run_log.csv",
                                 newline="", encoding="utf-8", errors="ignore")))
    tot, split = {}, {}
    for x in r:
        k = int(float(x["step"]))
        v = {rp: float(x.get(f"ramp_metering_release_actual_{rp}_veh", 0) or 0) * 20.0 for rp in RAMPS}
        split[k] = v
        tot[k] = sum(v.values())
    return tot, split


def rollout(scen, pcent_tot, pcent_split, use_price):
    env = RLLeaderEnv(scenario_name=scen)
    env.reset()
    warm = float(env.sim.total_ttt)
    done = False
    while not done:
        k = env.step_idx
        nuf = float(pcent_tot.get(k, 5254.0))
        nuf = max(0.0, min(6000.0, nuf))
        if use_price:
            # 각 merge에서 P-CENT가 더 조인(release 낮은) 램프에 +가격 → follower swap 유도
            op = {rp: float(env.previous.ramp_metering.get(rp, 0.0)) for rp in RAMPS} if env.previous else {}
            prices = {rp: 0.0 for rp in RAMPS}
            tgt = pcent_split.get(k, {})
            for m, rps in MERGES.items():
                if len(tgt) == 4:
                    harder = min(rps, key=lambda rp: tgt.get(rp, 1e9))  # P-CENT가 더 조인 램프
                    prices[harder] = PRICE_MAG
            env.follower.metering_marginal_price = prices
            env.follower.metering_marginal_price_ref = op
            env.follower.metering_marginal_price_trust_frac = 0.20
        else:
            env.follower.metering_marginal_price = None
        env.step(env.budget_to_action(NP, nuf))
        env.follower.metering_marginal_price = None
    return float(env.sim.total_ttt) - warm


def main():
    scen = "sweet_190_skew15_w60"
    tot, split = load_pcent()
    print(f"=== P-CENT 추종 테스트: {scen} ===", flush=True)
    print(f"기준: P-CENT 5757 | PFO 6299 | P-Stack 6379 | NC 6882 | 정적sweep최선 6325", flush=True)
    print(f"P-CENT 총유입 peak ~{tot.get(15,0):.0f} veh/h (follower 스윗스팟 5254보다 낮음)\n", flush=True)
    a = rollout(scen, tot, split, use_price=False)
    print(f"(A) budget만 P-CENT총량 추종 (가격無): windowed TTT = {a:.1f}", flush=True)
    b = rollout(scen, tot, split, use_price=True)
    print(f"(B) + metering 가격으로 P-CENT 분배 흉내: windowed TTT = {b:.1f}", flush=True)
    print(f"\n판정: B가 A보다 크게 낮고 P-CENT(5757)에 근접하면 → budget총량+가격분배가 열쇠.", flush=True)
    print(f"      B도 폭발/부족이면 → 이 instrument로도 P-CENT 구조적 미도달(분산 한계).", flush=True)


if __name__ == "__main__":
    main()
