#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in SOURCE_ROOT VLA_CKPT VLM_CKPT OPENPI_ROOT OPENPI_INFERENCE_ROOT; do
  [[ -n "${!name:-}" ]] || { echo "[ERROR] Set ${name}." >&2; exit 2; }
done

export TASK_ID=16
export NUM_TRIALS="${NUM_TRIALS:-29}"
export SEED="${SEED:-100}"
export VLA_POLICY_SEED=100
export REPLAN_STEPS=1
export POST_STAGE_STEPS=30
export VLM_INTERVAL=25
export HOLD_AFTER_REQUIRED_STAGES=1
export STAGE_LATCH_AUTONOMOUS_HOLD=0
export PROMPT_NO_REGRESSION=1
export POUR_RETURN_ASSIST=1
export POUR_RETURN_ASSIST_TARGET_RADIUS=0.20
export POUR_RETURN_ASSIST_ROTATION_MAGNITUDE=0.8
export POUR_RETURN_ASSIST_MAX_STEPS=24
export ORACLE_FORCE_INITIAL_PROMPT=0
export ORACLE_HOLD_RELEASE_NEXT=0
export ORACLE_STAGE_ADVANCE_NEXT=0
export ORACLE_TASK8_PICK_AFTER_PLACE_STEPS=-1
export EVALUATOR_FILE_OVERRIDE="${PACK_DIR}/evaluators/eval_counting_autonomous_pour_return_assist_d9f83ac.py"
export RUN_ID="${RUN_ID:-task16_vlm35999_pourreturnassist_${NUM_TRIALS}ep_$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-${PACK_DIR}/runs/${RUN_ID}}"

exec "${PACK_DIR}/scripts/run_autonomous_task.sh"
