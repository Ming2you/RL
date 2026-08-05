# held-out 일반화 평가(Phase 4, 2026-07-22) — RL 정책 vs NC/PFO/P-Stack/P-CENT, windowed TTT
"""연속분포로 학습한 RL leader를 학습에 없던 190+stressor(190-incident/190-skew)에서 평가.
지표 = windowed TTT(warmup 5스텝 제외) = cum_total_ttt[last] − cum_total_ttt[step==WARM-1].
baseline은 data/holdout/ho_{ctrl}_{tag}.csv에서 동일 방식으로 계산.
RL은 RLLeaderEnv 롤아웃(deterministic) 후 (final − warmup) total_ttt.

usage: python rl_leader/eval_holdout.py <actor.pt> [tag]
"""
from __future__ import annotations
import sys, csv
from pathlib import Path
if Path(r"C:/torchlib").is_dir() and r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")   # Windows 긴경로 우회 설치일 때만(HANDOFF §7.2)

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.env import RLLeaderEnv
from rl_leader.nets import Actor

WARM = 5  # WARMUP_NC_STEPS (baseline·env 동일)
# (표시명, 시나리오, tag, {ctrl_dir_key: run_log 하위 컨트롤러 폴더명})
HOLDOUT = [
    ("190-incident", "sweet_190_incident_w60", "inc"),
    ("190-skew",     "sweet_190_skew15_w60",   "skew"),
]
BASE = [  # (라벨, ho 접두)
    ("NC",      "ho_nc"),
    ("PFO",     "ho_pfo"),
    ("P-Stack", "ho_pstack"),
    ("P-CENT",  "ho_pcent"),
]


def windowed_from_csv(path: Path):
    """run_log.csv → windowed TTT. 없으면 None."""
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    try:
        cum = [float(r["cumulative_total_ttt"]) for r in rows]
        steps = [int(float(r["step"])) for r in rows]
    except (KeyError, ValueError):
        return None
    # step==WARM-1 인 행의 cumulative를 기준선으로 차감
    base_idx = next((i for i, s in enumerate(steps) if s == WARM - 1), None)
    base = cum[base_idx] if base_idx is not None else 0.0
    return cum[-1] - base


def rollout_windowed(actor, scenario):
    env = RLLeaderEnv(scenario_name=scenario)
    env.reset()
    warm_ttt = float(env.sim.total_ttt)   # warmup(0..WARM-1) 누적
    done = False
    obs = env._observe()
    while not done:
        obs, _, done, info = env.step(actor.act(obs, deterministic=True))
    return float(env.sim.total_ttt) - warm_ttt


def main(policy_path, tag):
    sd = torch.load(policy_path)
    od = sd["obs_mu"].shape[0]; ad = sd["mu.bias"].shape[0]
    actor = Actor(od, ad); actor.load_state_dict(sd); actor.eval()
    print(f"\n=== [{tag}] held-out 평가 (policy={Path(policy_path).name}, windowed TTT) ===")
    for disp, scen, htag in HOLDOUT:
        base_vals = {}
        for label, pref in BASE:
            p = ROOT / "data" / "holdout" / f"{pref}_{htag}.csv"
            base_vals[label] = windowed_from_csv(p)
        try:
            rl = rollout_windowed(actor, scen)
        except Exception as e:
            print(f"  [{disp}] RL ROLLOUT FAIL: {e}"); rl = None
        print(f"\n  --- {disp} ({scen}) ---")
        hdr = f"    {'controller':10}{'windowed TTT':>14}{'vs NC':>9}"
        print(hdr)
        nc = base_vals.get("NC")
        def impr(v):
            return f"{(nc - v) / nc * 100:+7.1f}%" if (nc and v is not None) else "     - "
        for label in ["NC", "PFO", "P-Stack", "P-CENT"]:
            v = base_vals.get(label)
            vs = f"{v:14.1f}" if v is not None else f"{'(running)':>14}"
            print(f"    {label:10}{vs}{impr(v):>9}")
        vs = f"{rl:14.1f}" if rl is not None else f"{'FAIL':>14}"
        print(f"    {'RL':10}{vs}{impr(rl):>9}")
        # RL vs P-Stack(최강 optimizer) 비율
        ps = base_vals.get("P-Stack")
        if rl is not None and ps:
            print(f"    → RL/P-Stack ratio = {rl/ps:.3f}  (1.0 미만이면 RL 우세)")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "checkpoints" / "actor_iql.pt")
    t = sys.argv[2] if len(sys.argv) > 2 else "RL-continuous"
    main(p, t)
