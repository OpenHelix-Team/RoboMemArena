#!/usr/bin/env bash
set -euo pipefail

TASK_ID=${1:?usage: run_microwave_eefonly_no_object_gate.sh TASK_ID}
case "${TASK_ID}" in
  20|21|22|23|24) ;;
  *) echo "unsupported TASK_ID=${TASK_ID}; expected 20,21,22,23,24" >&2; exit 2 ;;
esac

RUNTIME_PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${RUNTIME_PACK_DIR}"

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
export MODE=vlm_free
export RUN_ID=${RUN_ID:-mw_orig35999_t${TASK_ID}_vlmfree_eefonly_noobj_nolift_${STAMP}}
export PORT=${PORT:-$((9040 + TASK_ID))}
export NUM_TRIALS=${NUM_TRIALS:-1}
export SEED=${SEED:-104}
export MAX_STEPS=${MAX_STEPS:-1000}
export REPLAN_STEPS=${REPLAN_STEPS:-10}

# VLM must output the next prompt by itself. Hold/release only controls timing.
export VLM_COMPLETED_SUBTASKS_MODE=${VLM_COMPLETED_SUBTASKS_MODE:-off}
export COMPLETED_UPDATE_FROM_OFFICIAL_STAGE=${COMPLETED_UPDATE_FROM_OFFICIAL_STAGE:-0}
export VLM_HOLD_STATE_HINT=${VLM_HOLD_STATE_HINT:-0}
export ALLOW_STAGE_DONE_RELEASE_ANCHOR=${ALLOW_STAGE_DONE_RELEASE_ANCHOR:-0}
export STRICT_HOLD_RELEASE_NEXT=${STRICT_HOLD_RELEASE_NEXT:-0}
export REQUIRE_INITIAL_VLM_SUBTASK=0

# These defaults preserve the established EEF-hold run. Dedicated no-order
# launchers may override them without changing the historical entry point.
export MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD=${MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD:-1}
export MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION=default_vla
export REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD=${REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD:-1}
export REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD=${REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD:-1}
export REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE=${REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE:-1}

# Pure EEF hold: no object-in-microwave gate, no object lift gate, no gripper gate.
export ENDPOSE_PLACE_OBJECT_GATE_JSON=
export ENDPOSE_PICK_OBJECT_LIFT_GATE=0
export ENDPOSE_PICK_OBJECT_LIFT_GATE_BY_SUBTASK_JSON='{}'
export ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR=0
export ENDPOSE_PICK_GRIPPER_GATE=0
export ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE=0

export ENDPOSE_HOLD_POS_TOL=${ENDPOSE_HOLD_POS_TOL:-0.08}
export ENDPOSE_HOLD_EEF_DEFAULT_TOL=${ENDPOSE_HOLD_EEF_DEFAULT_TOL:-0.08}
export ENDPOSE_HOLD_EEF_P95_EXTRA_TOL=${ENDPOSE_HOLD_EEF_P95_EXTRA_TOL:-0.00}
export ENDPOSE_HOLD_EEF_TOL_CAP=${ENDPOSE_HOLD_EEF_TOL_CAP:-0.24}
export ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE=${ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE:-}

export POST_HOLD_RELEASE_VLA_STEPS=50
export POST_PICK_RELEASE_KEEP_GRIPPER_STEPS=${POST_PICK_RELEASE_KEEP_GRIPPER_STEPS:-0}
export POST_PICK_RELEASE_KEEP_GRIPPER_VALUE=${POST_PICK_RELEASE_KEEP_GRIPPER_VALUE:-1.0}
export SUBTASK_RELEASE_ANCHORS_JSON=${SUBTASK_RELEASE_ANCHORS_JSON:-${RUNTIME_PACK_DIR}/config/release_anchors_empty.json}

case "${TASK_ID}" in
  20) ;;
  21)
    if [[ -z "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE:-}" && -z "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON:-}" ]]; then
      export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON='{"pick butter":0.18,"place butter":0.24,"pick chocolate":0.18,"place chocolate":0.24}'
    fi
    ;;
  23)
    if [[ -z "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE:-}" && -z "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON:-}" ]]; then
      export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON='{"pick cream":0.14,"place cream":0.18,"pick popcorn":0.18,"place popcorn":0.18}'
    fi
    ;;
  24)
    if [[ -z "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE:-}" && -z "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON:-}" ]]; then
      export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON='{"pick cookies":0.24,"place cookies":0.24,"pick popcorn":0.18,"place popcorn":0.24}'
    fi
    ;;
esac

exec bash "${RUNTIME_PACK_DIR}/scripts/launch_one_sync_hold_orig35999.sh" "${TASK_ID}"
