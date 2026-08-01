#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${INFER_ROOT:?set INFER_ROOT}"
: "${TARGET_LIBERO_PATH:?set TARGET_LIBERO_PATH}"
: "${ROBOMEMARENA_REMOTE_ROOT:?set ROBOMEMARENA_REMOTE_ROOT}"
: "${TASK21_DATA_ROOT:?set TASK21_DATA_ROOT}"
: "${VLA_POLICY:?set VLA_POLICY}"
: "${VLM_CKPT:?set VLM_CKPT}"

export VLA_REPO_ID=${VLA_REPO_ID:-${PACK_DIR}/assets/norm_repo}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${PACK_DIR}/outputs}
export STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_ID=${RUN_ID:-task21_v121_seed${SEED:-104}_${STAMP}}
export OUT_ROOT=${OUT_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}
export NUM_TRIALS=1
export SEED=${SEED:-104}
export MAX_STEPS=2000
export REPLAN_STEPS=5

# v121 invariant: VLM chooses prompts; no oracle prompt injection is permitted.
export ORACLE_HOLD_RELEASE_NEXT=0
export ORACLE_FORCE_INITIAL_PROMPT=0
export ORACLE_INITIAL_STAGE_LOCK=0
export ORACLE_STAGE_ADVANCE_NEXT=0
export ORACLE_MONOTONIC_SEQUENCE_LOCK=0
export ORACLE_STAGE_LOCK_UNTIL_DONE=0

# Exact v121 top-level contract. The release-anchor HDF path is materialized
# from TASK21_DATA_ROOT locally; the policy values are unchanged.
export RUN_VERSION=v121_nopick2place_upward
export ENDPOSE_HOLD_TARGETS_JSON=${PACK_DIR}/config/task21_v121_eef_targets.json
# Keep the frozen v121 tolerance file as the default.  A versioned experiment
# may opt in to a separately tracked file without mutating the frozen default.
export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE=${TASK21_HOLD_TOLERANCES:-${PACK_DIR}/config/task21_v121_eef_tolerances.json}
export ENDPOSE_TARGET_PASSAGE_COUNTS_JSON=${PACK_DIR}/config/task21_v121_passages.json
export ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON=${PACK_DIR}/config/task21_v121_pick_directions.json
export ENDPOSE_HOLD_DIRECTION_COS_MIN=0.50
export ENDPOSE_HOLD_DIRECTION_WINDOW=3
export ENDPOSE_HOLD_DIRECTION_MIN_DISPLACEMENT=0.001
export ENDPOSE_HOLD_DIRECTION_TREND_EPS=0.03
export ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE=${PACK_DIR}/config/task21_v121_min_hold_steps.json
export POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS=50

mkdir -p "${OUT_ROOT}/runtime_config"
RELEASE_ANCHOR_TEMPLATE=${TASK21_RELEASE_ANCHOR_TEMPLATE:-${PACK_DIR}/config/task21_v121_release_anchors.template.json}
[[ -f "${RELEASE_ANCHOR_TEMPLATE}" ]] || { echo "missing release-anchor template: ${RELEASE_ANCHOR_TEMPLATE}" >&2; exit 2; }
python3 "${PACK_DIR}/scripts/materialize_task21_paths.py" \
  --template "${RELEASE_ANCHOR_TEMPLATE}" \
  --output "${OUT_ROOT}/runtime_config/task21_v121_release_anchors.json" \
  --data-root "${TASK21_DATA_ROOT}"
export SUBTASK_RELEASE_ANCHORS_JSON=${OUT_ROOT}/runtime_config/task21_v121_release_anchors.json

exec bash "${PACK_DIR}/scripts/run_task21_v110_historicalvlm_eef_pickfinish50_latest622_1ep.sh"
