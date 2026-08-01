#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in SOURCE_ROOT VLA_CKPT VLM_CKPT OPENPI_ROOT OPENPI_INFERENCE_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "[ERROR] Set ${name}." >&2; exit 2; }
done

export TASK_ID=7
export NUM_TRIALS=8
export SEED=100
export REPLAN_STEPS=5
export POST_STAGE_STEPS=30
export VLM_INTERVAL=25
export HOLD_AFTER_REQUIRED_STAGES=0
export EVALUATOR_FILE_OVERRIDE="${PACK_DIR}/evaluators/eval_counting_autonomous_guarded_d9f83ac.py"
export RUN_ID="${RUN_ID:-task7_vlm35999_hardcase500_8ep_$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-${PACK_DIR}/runs/${RUN_ID}}"

exec "${PACK_DIR}/scripts/run_autonomous_task.sh"
