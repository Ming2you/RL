#!/usr/bin/env bash
# 3시드 학습 대기 → 순차 held-out 평가(트레이스) → 메커니즘 귀속 분석 (2026-07-31)
# 순차 실행: 수집기 14개가 계속 도는 중이라 병렬 평가는 CPU 과다구독 → 시간가드 오작동 위험.
set -u
cd /c/Users/alsrj/Desktop/Numerical-Sim-offiter
export PYTHONIOENCODING=utf-8
PY="C:/Users/alsrj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
G='경고|calibrat|WARNING|캘리|merge|Ķ|리 '

echo "### [1] 3시드 학습 완료 대기 $(date '+%m-%d %H:%M')"
for i in $(seq 1 240); do
  n=0
  for s in 0 1 2; do grep -q "saved actor(IQL)" rl_leader/logs/iql_s$s.log 2>/dev/null && n=$((n+1)); done
  [ "$n" -ge 3 ] && { echo "### 3시드 학습 완료 $(date +%H:%M)"; break; }
  sleep 60
done

echo "### [2] 순차 held-out 평가 (시드별 2셀, 트레이스 저장)"
for s in 0 1 2; do
  if [ ! -f rl_leader/actor_iql_s$s.pt ]; then echo "  s$s 체크포인트 없음 — 건너뜀"; continue; fi
  echo "--- seed $s 평가 시작 $(date +%H:%M) ---"
  "$PY" rl_leader/eval_guarded.py rl_leader/actor_iql_s$s.pt --max-sec 3600 \
      --trace-dir rl_leader/traces --tag s$s 2>&1 | grep -vE "$G"
done

echo "### [3] 메커니즘 귀속 분석 (seed 0)"
"$PY" rl_leader/analyze_mechanism.py --trace-dir rl_leader/traces --tag s0 2>&1 | grep -vE "$G"

echo "### SEEDS_EVAL_DONE $(date '+%m-%d %H:%M')"
