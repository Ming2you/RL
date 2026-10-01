#!/usr/bin/env bash
# Run one registered spec on all five canonical scenarios, one after another on a single CPU.
# Usage: run_canonical_sequential.sh <spec.json> <run_root> <cpu_mask>
set -u
SPEC=$1; ROOT=$2; MASK=$3
cd /c/Users/alsrj/Desktop/RL
export PYTHONUTF8=1
LOGS=results/sdmpc_rl_machine_b_20260930/logs
for S in sweet_170_incident_w sweet_155_w sweet_170_w sweet_170_skew15_w sweet_190_w; do
  .venv-torch/Scripts/python.exe -B -u work/sdmpc_rl_perimeter_b_20261001/canonical_eval.py --scenario $S \
     --spec "$SPEC" --output "$ROOT" --cpu-mask $MASK \
     > "$LOGS/$(basename $ROOT)_$S.stdout.log" 2> "$LOGS/$(basename $ROOT)_$S.stderr.log"
  echo "$S exit $? $(tail -1 "$LOGS/$(basename $ROOT)_$S.stdout.log")"
done
.venv-torch/Scripts/python.exe -B work/sdmpc_rl_perimeter_b_20261001/readout.py "$ROOT" > "$ROOT/readout.json"
echo "readout: $(grep -m1 '"status"' "$ROOT/readout.json")"
