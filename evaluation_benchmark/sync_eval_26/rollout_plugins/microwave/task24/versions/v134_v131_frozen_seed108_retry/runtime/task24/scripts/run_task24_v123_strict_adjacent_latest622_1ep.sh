#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This only rejects a non-adjacent jump while the current EEF hold is waiting.
# The accepted adjacent prompt still has to come from the VLM.
export STRICT_HOLD_RELEASE_NEXT=1
export ENDPOSE_HOLD_TARGETS_JSON=${PACK_DIR}/config/tasks21_23_24_pick_contact_deep_place_eef_targets_20260718.json
export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE=${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE:-${PACK_DIR}/config/tasks21_23_24_contact_pick_deep_place_tolerances_20260718.json}
export ENDPOSE_TARGET_PASSAGE_COUNTS_JSON=${PACK_DIR}/config/tasks21_23_24_pick_firstpassage_20260718.json
export ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON=${ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON:-${PACK_DIR}/config/tasks21_23_24_allpick_upward_direction_20260718.json}
export ENDPOSE_HOLD_DIRECTION_COS_MIN=0.50
export ENDPOSE_HOLD_DIRECTION_WINDOW=3
export ENDPOSE_HOLD_DIRECTION_MIN_DISPLACEMENT=0.001
export ENDPOSE_HOLD_DIRECTION_TREND_EPS=0.03
export SUBTASK_RELEASE_ANCHORS_JSON=${PACK_DIR}/config/release_anchors_t21_t23_t24_no_pick2place_robotonly_20260718.json
export POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS=50
export REPRO_ENTRY_LAUNCHER=${BASH_SOURCE[0]}

exec bash "${PACK_DIR}/scripts/run_task23_24_v112_historicalvlm_eef_pickfinish50_latest622_1ep.sh" 24
