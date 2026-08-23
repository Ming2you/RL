#!/usr/bin/env bash
# RL 파이프라인 자동 체인(2026-07-22): 데이터대기 → torch BC → BC평가 → SAC(5셀) → SAC평가
set -u
cd /c/Users/alsrj/Desktop/Numerical-Sim-offiter
PY="C:/Users/alsrj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
D=rl_leader/bc_data

echo "### [1] 5셀 BC 데이터 대기..."
until [ -f $D/155.npz ] && [ -f $D/170.npz ] && [ -f $D/skew15.npz ] && [ -f $D/incident.npz ] && [ -f $D/190.npz ]; do sleep 60; done
echo "### 데이터 준비됨 $(date +%H:%M)"

echo "### [2] torch BC (5셀 합쳐)"
"$PY" rl_leader/bc_train_torch.py $D/155.npz $D/170.npz $D/skew15.npz $D/incident.npz $D/190.npz --out rl_leader/actor_bc.pt 2>&1 | grep -vE "경고|calibrat|WARNING|캘리"

echo "### [3] BC 정책 5셀 평가"
"$PY" rl_leader/eval_all.py rl_leader/actor_bc.pt BC 2>&1 | grep -vE "경고|calibrat|WARNING|캘리"

echo "### [4] SAC 학습 (5셀 도메인랜덤, BC init, 1500 steps) $(date +%H:%M)"
"$PY" rl_leader/sac.py --bc rl_leader/actor_bc.pt --steps 1500 --save_every 300 --out rl_leader/actor_sac.pt 2>&1 | grep -vE "경고|calibrat|WARNING|캘리"

echo "### [5] SAC 정책 5셀 평가 $(date +%H:%M)"
"$PY" rl_leader/eval_all.py rl_leader/actor_sac.pt SAC 2>&1 | grep -vE "경고|calibrat|WARNING|캘리"

echo "### PIPELINE DONE $(date +%H:%M)"
