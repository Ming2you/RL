#!/usr/bin/env bash
# sweep 완료 대기 → 오라클(위상별 budget) 측정 자동 체인(2026-07-23)
set -u
cd /c/Users/alsrj/Desktop/Numerical-Sim-offiter
PY="C:/Users/alsrj/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe"
export PYTHONIOENCODING=utf-8
SWEEP=rl_leader/logs/ceiling_final.log
GREP='경고|calibrat|WARNING|캘리|merge|Ķ|리 '

echo "### [oracle-1] sweep 완료 대기 (7/7) ... $(date +%H:%M)"
until [ "$(grep -cE '^ *[0-9]{3,} ' "$SWEEP" 2>/dev/null || echo 0)" -ge 7 ]; do sleep 60; done
echo "### sweep 완료 $(date +%H:%M):"
grep -E '^ *[0-9]{3,} ' "$SWEEP" | grep -vE "$GREP"

echo "### [oracle-2] 위상별 오라클 측정 (skew) $(date +%H:%M)"
"$PY" rl_leader/oracle_measure.py sweet_190_skew15_w60 2>&1 | grep -vE "$GREP" | tee rl_leader/logs/oracle_skew.log

echo "### ORACLE DONE $(date +%H:%M)"
