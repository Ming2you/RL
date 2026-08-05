# BC 정책 학습(Phase 1, 2026-07-22) — numpy MLP로 obs→action 회귀
"""optimizer-leader 데이터((obs, budget-action))를 numpy MLP로 BC.
torch 부재 환경 → 순수 numpy(2 hidden ReLU, tanh 출력, Adam, MSE).
Phase 2 SAC는 torch 필요(별도 셋업).

usage: python rl_leader/bc_train.py <data1.npz> [data2.npz ...] --out policy.npz
저장: 가중치 + obs 표준화(mean/std) → RL 정책 초기화·배포에 사용.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def relu(z): return np.maximum(0.0, z)


class MLP:
    def __init__(self, din, dh, dout, seed=0):
        rng = np.random.default_rng(seed)
        s = lambda a, b: rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
        self.W1, self.b1 = s(din, dh), np.zeros(dh)
        self.W2, self.b2 = s(dh, dh), np.zeros(dh)
        self.W3, self.b3 = s(dh, dout) * 0.1, np.zeros(dout)
        self.params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self._m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self._t = 0

    def forward(self, X, cache=False):
        z1 = X @ self.W1 + self.b1; a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = relu(z2)
        y = np.tanh(a2 @ self.W3 + self.b3)
        if cache: self._c = (X, z1, a1, z2, a2, y)
        return y

    def backward(self, Ytrue):
        X, z1, a1, z2, a2, y = self._c
        n = X.shape[0]
        dy = (y - Ytrue) * (1.0 - y ** 2) * (2.0 / n)          # MSE + tanh
        g = {}
        g["W3"] = a2.T @ dy; g["b3"] = dy.sum(0)
        da2 = dy @ self.W3.T; dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
        return g

    def adam(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for p in self.params:
            self._m[p] = b1 * self._m[p] + (1 - b1) * g[p]
            self._v[p] = b2 * self._v[p] + (1 - b2) * g[p] ** 2
            mh = self._m[p] / (1 - b1 ** self._t); vh = self._v[p] / (1 - b2 ** self._t)
            setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + eps))


def main(paths, out, epochs=400, dh=128):
    Xs, Ys = [], []
    for p in paths:
        d = np.load(p); Xs.append(d["X"]); Ys.append(d["Y"])
    X = np.concatenate(Xs).astype(np.float64); Y = np.concatenate(Ys).astype(np.float64)
    print(f"data: X={X.shape} Y={Y.shape}")
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    rng = np.random.default_rng(0); idx = rng.permutation(len(Xn))
    ntr = int(0.85 * len(Xn)); tr, va = idx[:ntr], idx[ntr:]
    net = MLP(X.shape[1], dh, Y.shape[1])
    for e in range(epochs):
        net.forward(Xn[tr], cache=True); net.adam(net.backward(Y[tr]), lr=2e-3)
        if (e + 1) % 100 == 0 or e == 0:
            tr_mse = float(((net.forward(Xn[tr]) - Y[tr]) ** 2).mean())
            va_mse = float(((net.forward(Xn[va]) - Y[va]) ** 2).mean())
            print(f"  epoch {e+1:4d}  train_mse={tr_mse:.4f}  val_mse={va_mse:.4f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{p: getattr(net, p) for p in net.params}, obs_mu=mu, obs_sd=sd)
    print(f"saved policy → {out}  (val_mse={va_mse:.4f})")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = str(ROOT / "checkpoints" / "bc_policy.npz")
    if "--out" in args:
        i = args.index("--out"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    paths = args or [str(ROOT / "data" / "bc" / "170.npz")]
    main(paths, out)
