#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ROBOMEMARENA_REMOTE_ROOT:?set ROBOMEMARENA_REMOTE_ROOT}"

export LOG_BASE="${OUT_ROOT:?set OUT_ROOT}"
export EVAL_PY="${PACK_DIR}/evaluators/eval_task1_qwen3_sync_endpose_hold_officialscore.py"
export TASK1_BASE_EVAL_PY="${REPO_ROOT}/evaluation_benchmark/openpi_minimal_runtime/eval_task1_qwen3_async_openpi_inference_vla_cam.py"
export ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR="${REPO_ROOT}/evaluation_benchmark/scripts"
export BASE_MODEL_DIR="${VLM_CKPT:?set VLM_CKPT}"
export VLA_POLICY="${VLA_POLICY:?set VLA_POLICY}"
export BDDL="${BDDL_PATH:?set BDDL_PATH}"
export ENDPOSE_HOLD_TARGETS_JSON="${ENDPOSE_HOLD_TARGETS_JSON:-${PACK_DIR}/config/task1_subtask_end_poses_successindex_seed100_199.json}"
export NUM_TRIALS_PER_TASK=1
export POST_HOLD_RELEASE_VLA_STEPS="${POST_HOLD_RELEASE_VLA_STEPS:-30}"
export VLM_PROMPT_PROFILE="${VLM_PROMPT_PROFILE:-task1_no_label_no_order}"
export PREVENT_SUBTASK_REGRESSION="${PREVENT_SUBTASK_REGRESSION:-1}"
export REGRESSION_GUARD_AFTER_HOLD_RELEASE="${REGRESSION_GUARD_AFTER_HOLD_RELEASE:-1}"
export TASK1_ACCEPT_RAW_VLM_OUTPUT="${TASK1_ACCEPT_RAW_VLM_OUTPUT:-1}"
export TASK1_DISABLE_OUTPUT_NORMALIZE="${TASK1_DISABLE_OUTPUT_NORMALIZE:-1}"

exec bash "${PACK_DIR}/evaluators/run_task1_officialscore.sh"
