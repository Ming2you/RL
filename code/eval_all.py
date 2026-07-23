# 정책 5셀 평가(Phase 2, 2026-07-22) — RL 정책 롤아웃 → whole-sim TTT vs NC/optimizer
"""usage: python rl_leader/eval_all.py <actor.pt> [tag]
NC baseline = _wang whole-sim total. optimizer baseline = collect log 최종 cum_ttt.
출력: 시나리오별 NC / optimizer / policy TTT + policy 개선율(vs NC), 그리고 policy/optimizer 비율.
"""
from __future__ import annotations
import sys, re
from pathlib import Path
if r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv
from rl_leader.nets import Actor

CELLS = [("155", "sweet_155_w60"), ("170", "sweet_170_w60"), ("skew15", "sweet_170_skew15_w60"),
         ("incident", "sweet_170_incident_w60"), ("190", "sweet_190_w60")]
NC = {"155": 3170.4, "170": 4667.3, "skew15": 4692.1, "incident": 6086.9, "190": 7006.9}  # _wang whole-sim


def opt_ttt(cell):
    log = ROOT / "rl_leader" / "logs" / f"collect_{cell}.log"
    if not log.exists():
        return None
    vals = re.findall(r"cum_ttt=([0-9.]+)", log.read_text(encoding="utf-8", errors="ignore"))
    return float(vals[-1]) if vals else None


def rollout(actor, scenario):
    env = RLLeaderEnv(scenario_name=scenario)
    obs = env.reset(); done = False
    while not done:
        obs, r, done, info = env.step(actor.act(obs, deterministic=True))
    return float(env.sim.total_ttt)


def main(policy_path, tag):
    sd = torch.load(policy_path)
    od = sd["obs_mu"].shape[0]; ad = sd["mu.bias"].shape[0]
    actor = Actor(od, ad); actor.load_state_dict(sd); actor.eval()
    print(f"\n=== [{tag}] 5셀 평가 (policy={Path(policy_path).name}) ===")
    print(f"{'cell':10}{'NC':>9}{'optim':>9}{'policy':>9}{'impr%':>8}{'pol/opt':>9}")
    rows = []
    for cell, scen in CELLS:
        try:
            pt = rollout(actor, scen)
        except Exception as e:
            print(f"{cell:10}  ROLLOUT FAIL {e}"); continue
        nc = NC[cell]; ot = opt_ttt(cell)
        impr = (nc - pt) / nc * 100.0
        ratio = pt / ot if ot else float('nan')
        print(f"{cell:10}{nc:9.0f}{(ot or 0):9.0f}{pt:9.0f}{impr:8.1f}{ratio:9.3f}")
        rows.append((cell, nc, ot, pt, impr, ratio))
    return rows


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "rl_leader" / "actor_bc.pt")
    tag = sys.argv[2] if len(sys.argv) > 2 else "policy"
    main(p, tag)
