#!/usr/bin/env bash
# Phase 4 자동 체인(2026-07-22): 파일럿 완료 대기 → 연속분포 SAC(도메인 랜덤화) → held-out 평가
# 트리거: 파일럿(run_rl_pipeline.sh)이 actor_sac.pt를 남기면 시작. init은 중립적인 actor_bc.pt.
# held-out = 190-incident/190-skew (학습분포에 없음: stressor 활성 시 수요≤1.80).
set -u
cd /c/Users/alsrj/Desktop/Numerical-Sim-offiter
PY="C:/Users/alsrj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
GREP='경고|calibrat|WARNING|캘리|UserWarning|merge|Ķ|리'
BC=rl_leader/actor_bc.pt
PILOT=rl_leader/actor_sac.pt
CONT=rl_leader/actor_sac_continuous.pt

echo "### [P4-1] 파일럿 완료 대기 (actor_sac.pt) ... $(date +%H:%M)"
until [ -f "$PILOT" ]; do sleep 60; done
sleep 30   # 파일럿 5셀 평가 출력 마무리 여유
echo "### 파일럿 완료 확인 $(date +%H:%M)"

if [ ! -f "$BC" ]; then echo "!!! BC 체크포인트 없음($BC) — 연속 학습 중단"; exit 1; fi

echo "### [P4-2] 연속분포 SAC 학습 (BC init, 도메인 랜덤화, 3000 steps) $(date +%H:%M)"
"$PY" rl_leader/sac.py --continuous --bc "$BC" --steps 3000 --save_every 300 \
    --out "$CONT" 2>&1 | grep -vE "$GREP"

echo "### [P4-3] held-out baseline 완료 대기 (8 run_log) ... $(date +%H:%M)"
BASE_OK=0
for _ in $(seq 1 60); do   # 최대 60분 대기
  n=$(ls outputs/_wang3/ho_{nc,pfo,pstack,pcent}_{inc,skew}/*/run_log.csv 2>/dev/null | wc -l)
  if [ "$n" -ge 8 ]; then BASE_OK=1; break; fi
  echo "  baseline $n/8 준비, 대기..."; sleep 60
done
[ "$BASE_OK" -eq 1 ] && echo "### baseline 8/8 준비됨" || echo "### baseline 일부 미완(있는 것만 비교)"

echo "### [P4-4] held-out 평가 — BC(비교군) $(date +%H:%M)"
"$PY" rl_leader/eval_holdout.py "$BC" BC-raw 2>&1 | grep -vE "$GREP"
echo "### [P4-5] held-out 평가 — SAC-continuous $(date +%H:%M)"
"$PY" rl_leader/eval_holdout.py "$CONT" SAC-continuous 2>&1 | grep -vE "$GREP"

echo "### PHASE4 DONE $(date +%H:%M)"
