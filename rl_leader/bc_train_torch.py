# BC 사전학습 torch(2026-07-22) — actor mean을 optimizer 결정에 MSE 회귀
"""usage: python rl_leader/bc_train_torch.py <d1.npz> [d2 ...] --out actor_bc.pt
저장: Actor state_dict(+obs 정규화 버퍼) → SAC actor init에 로드.
"""
from __future__ import annotations
import sys
from pathlib import Path
if Path(r"C:/torchlib").is_dir() and r"C:/torchlib" not in sys.path:
    sys.path.insert(0, r"C:/torchlib")   # Windows 긴경로 우회 설치일 때만(HANDOFF §7.2)

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_leader.nets import Actor


def main(paths, out, epochs=1500, dh=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    Xs, Ys = [], []
    for p in paths:
        d = np.load(p); Xs.append(d["X"]); Ys.append(d["Y"])
    X = np.concatenate(Xs).astype(np.float32); Y = np.concatenate(Ys).astype(np.float32)
    print(f"data: X={X.shape} Y={Y.shape}")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    rng = np.random.default_rng(seed); idx = rng.permutation(len(X))
    ntr = int(0.85 * len(X)); tr, va = idx[:ntr], idx[ntr:]
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    actor = Actor(X.shape[1], Y.shape[1], dh)
    actor.set_obs_norm(mu, sd)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    for e in range(epochs):
        actor.train()
        m, _ = actor.forward(Xt[tr]); pred = torch.tanh(m)
        loss = nn.functional.mse_loss(pred, Yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        if (e + 1) % 300 == 0 or e == 0:
            actor.eval()
            with torch.no_grad():
                vp = torch.tanh(actor.forward(Xt[va])[0])
                vmse = nn.functional.mse_loss(vp, Yt[va]).item()
            print(f"  epoch {e+1:4d}  train_mse={loss.item():.4f}  val_mse={vmse:.4f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), out)
    print(f"saved actor(BC) → {out}  val_mse={vmse:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]; out = str(ROOT / "checkpoints" / "actor_bc.pt")
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    paths = args or [str(ROOT / "data" / "bc" / "170.npz")]
    main(paths, out)
