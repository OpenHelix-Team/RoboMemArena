#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_OFFICIAL_COMMIT=514ecdf86ba47d496ab1728a827670833107ffd3

: "${OPENPI_ROOT:?set OPENPI_ROOT}"
: "${INFER_ROOT:?set INFER_ROOT}"
: "${TARGET_LIBERO_PATH:?set TARGET_LIBERO_PATH}"
: "${ROBOMEMARENA_REMOTE_ROOT:?set ROBOMEMARENA_REMOTE_ROOT}"
: "${TASK20_DATA_ROOT:?set TASK20_DATA_ROOT}"
: "${VLA_POLICY:?set VLA_POLICY}"
: "${VLM_CKPT:?set VLM_CKPT}"

actual_commit="$(git -C "${ROBOMEMARENA_REMOTE_ROOT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_OFFICIAL_COMMIT}" ]]; then
  echo "official scorer mismatch: expected=${EXPECTED_OFFICIAL_COMMIT} actual=${actual_commit}" >&2
  exit 3
fi
test -f "${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/scripts/task2_26_reference_stage.py"

export STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
export SEED=${SEED:-104}
export NUM_TRIALS=${NUM_TRIALS:-1}
export MAX_STEPS=${MAX_STEPS:-1000}
export REPLAN_STEPS=${REPLAN_STEPS:-10}
export PORT=${PORT:-9714}
export RUN_ID=${RUN_ID:-task20_v49c6_seed${SEED}_${STAMP}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${PACK_DIR}/outputs}
export OUT_ROOT=${OUT_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}
export VLA_REPO_ID=${VLA_REPO_ID:-${PACK_DIR}/assets/norm_repo}
export VLA_CONFIG=${VLA_CONFIG:-pi05_libero_robomemarena_fullvlm_v2_noflip_dataset}
export VLA_SERVER_PY=${VLA_SERVER_PY:-${PACK_DIR}/scripts/serve_policy_custom_repo.py}

# v49c6 autonomous contract: VLM owns prompt selection. The runtime can hold
# only at EEF targets and can apply robot-only anchors after VLM already chose
# the next prompt. No oracle prompt injection or object-state anchor is used.
export MODE=vlm_free
export ORACLE_HOLD_RELEASE_NEXT=0
export ORACLE_FORCE_INITIAL_PROMPT=0
export ORACLE_INITIAL_STAGE_LOCK=0
export ORACLE_STAGE_ADVANCE_NEXT=0
export ORACLE_MONOTONIC_SEQUENCE_LOCK=0
export ORACLE_STAGE_LOCK_UNTIL_DONE=0
export VLM_COMPLETED_SUBTASKS_MODE=off
export COMPLETED_UPDATE_FROM_OFFICIAL_STAGE=0
export VLM_HOLD_STATE_HINT=0
export ENDPOSE_HOLD_SKIP_VLM_INFERENCE=0
export REQUIRE_INITIAL_VLM_SUBTASK=0
export STRICT_HOLD_RELEASE_NEXT=0
export BLOCK_FORWARD_BEFORE_FIRST_STAGE_DONE=0
export MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD=0
export MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION=default_vla
export REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD=0
export REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD=0
export PREVENT_SUBTASK_REGRESSION=0
export FORWARD_SWITCH_BLOCK_PREVIOUS=0
export REGRESSION_GUARD_AFTER_HOLD_RELEASE=0
export HOLD_RELEASE_BLOCK_PAST_SUBTASKS=0
export PREVENT_COMPLETED_STAGE_REGRESSION=1
export PREVENT_RELEASED_HOLD_REGRESSION=0
export PREVENT_HELD_SUBTASK_REGRESSION=1

# EEF-only hold. These object, lift, gripper, region, and stage gates must stay
# disabled for an autonomous result.
export ENDPOSE_PLACE_OBJECT_GATE_JSON=
export ENDPOSE_PICK_OBJECT_LIFT_GATE=0
export ENDPOSE_PICK_OBJECT_LIFT_GATE_BY_SUBTASK_JSON='{}'
export ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR=0
export ENDPOSE_PICK_GRIPPER_GATE=0
export ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE=0
export ENDPOSE_HOLD_TARGETS_JSON="${PACK_DIR}/config/task20_targets_pick_contact_f80_20260711.json"
export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE="${PACK_DIR}/config/task20_eef_open105_pickcookies03_placecookies06_pickchoc045_placechoc04_tol_20260714.json"
export ENDPOSE_TARGET_PASSAGE_COUNTS_JSON="${PACK_DIR}/config/task20_passages_pickcookies1_pickchoc2_20260714.json"
export ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE="${PACK_DIR}/config/task20_eef_runtime_pickplace_hold30.json"
export ENDPOSE_HOLD_CONSECUTIVE_BY_SUBTASK_JSON='{"open microwave":1,"place cookies":5}'
export ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR=1
export ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR_SUBTASKS="pick chocolate"
export POST_HOLD_RELEASE_VLA_STEPS=50
export POST_HOLD_RELEASE_VLA_STEPS_BY_SUBTASK_JSON='{"pick chocolate":0}'
export POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS=50
export POST_PICK_RELEASE_KEEP_GRIPPER_STEPS=0
export ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT=1
export ENDPOSE_HOLD_AUTO_RESUME_COOLDOWN_STEPS=80
export ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT_EXCLUDE_SUBTASKS="open microwave,place cookies"
export REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE=1
export MICROWAVE_REQUIRE_OPEN_EEF_HOLD_FOR_SUCCESS=1
export MICROWAVE_DEBUG_SAVE_VLM_FRAMES=1

mkdir -p "${OUT_ROOT}/runtime_config"
python3 "${PACK_DIR}/scripts/materialize_task20_paths.py" \
  --template "${PACK_DIR}/config/release_anchors_t20_open2pick_f70_place2pick_f0_robotonly.template.json" \
  --output "${OUT_ROOT}/runtime_config/release_anchors.json" \
  --data-root "${TASK20_DATA_ROOT}"
export SUBTASK_RELEASE_ANCHORS_JSON="${OUT_ROOT}/runtime_config/release_anchors.json"

export TASKS_JSON='[20]'
export TASKS2_26_BASE_EVAL_PY="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py"
export TASK_CONFIG="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json"
export ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/scripts"
export ROBOMEMARENA_OFFICIAL_BDDL_DIR="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/bddl"
export EVAL_PY="${PACK_DIR}/evaluators/eval_tasks2_26_sync_endpose_hold_officialscore.py"
export REPRO_SNAPSHOT_HELPER="${PACK_DIR}/scripts/snapshot_task20_v49c6.sh"
export REQUIRE_REPRO_SNAPSHOT=1

exec bash "${PACK_DIR}/evaluators/run_tasks2_26_sync_hold_eval.sh"
