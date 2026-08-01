#!/usr/bin/env bash
set -euo pipefail

TASK_ID=${1:?usage: launch_one_sync_hold_orig35999.sh TASK_ID}
case "${TASK_ID}" in
  20|21|22|23|24) ;;
  *) echo "unsupported TASK_ID=${TASK_ID}; expected 20,21,22,23,24" >&2; exit 2 ;;
esac

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_OFFICIAL_COMMIT=d9f83ac5182e25ad7f0a301a77a0b667f2392df1
REMOTE_ROOT=${ROBOMEMARENA_REMOTE_ROOT:?ROBOMEMARENA_REMOTE_ROOT is required}
REMOTE_EVAL_ROOT="${REMOTE_ROOT}/evaluation_benchmark"
REFERENCE_DIR="${REMOTE_EVAL_ROOT}/reference_evaluation/tasks2_26_vlm5_reference"
ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR=${ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR:-${REMOTE_EVAL_ROOT}/scripts}
ROBOMEMARENA_OFFICIAL_BDDL_DIR=${ROBOMEMARENA_OFFICIAL_BDDL_DIR:-${REMOTE_EVAL_ROOT}/bddl}
ROBOMEMARENA_ROOT_BDDL_DIR=${ROBOMEMARENA_ROOT_BDDL_DIR:-${REMOTE_ROOT}/bddl}
if [[ -e "${REMOTE_ROOT}/.git" ]]; then
  ROBOMEMARENA_OFFICIAL_COMMIT=$(git -C "${REMOTE_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)
elif [[ -f "${REMOTE_ROOT}/COMMIT" ]]; then
  ROBOMEMARENA_OFFICIAL_COMMIT=$(tr -d '[:space:]' < "${REMOTE_ROOT}/COMMIT")
else
  ROBOMEMARENA_OFFICIAL_COMMIT=unknown
fi
if [[ "${ROBOMEMARENA_OFFICIAL_COMMIT}" != "${EXPECTED_OFFICIAL_COMMIT}" ]]; then
  echo "official scorer mismatch: expected=${EXPECTED_OFFICIAL_COMMIT} actual=${ROBOMEMARENA_OFFICIAL_COMMIT}" >&2
  exit 3
fi

OPENPI_ROOT=${OPENPI_ROOT:?OPENPI_ROOT is required}
INFER_ROOT=${INFER_ROOT:?INFER_ROOT is required}
TARGET_LIBERO_PATH=${TARGET_LIBERO_PATH:?TARGET_LIBERO_PATH is required}

VLA_POLICY=${VLA_POLICY:?VLA_POLICY is required}
VLA_CONFIG=${VLA_CONFIG:?VLA_CONFIG is required}
VLA_REPO_ID=${VLA_REPO_ID:?VLA_REPO_ID is required}
VLA_SERVER_PY=${VLA_SERVER_PY:-${PACK_DIR}/scripts/serve_policy_custom_repo.py}

VLM_CKPT=${VLM_CKPT:?VLM_CKPT is required}

ENDPOSE_HOLD_POS_TOL=${ENDPOSE_HOLD_POS_TOL:-0.08}
ENDPOSE_HOLD_EEF_DEFAULT_TOL=${ENDPOSE_HOLD_EEF_DEFAULT_TOL:-0.08}
ENDPOSE_HOLD_EEF_P95_EXTRA_TOL=${ENDPOSE_HOLD_EEF_P95_EXTRA_TOL:-0.00}
ENDPOSE_HOLD_EEF_TOL_CAP=${ENDPOSE_HOLD_EEF_TOL_CAP:-0.24}

MODE=${MODE:-vlm_guarded}
case "${MODE}" in
  oracle)
    ORACLE_HOLD_RELEASE_NEXT=${ORACLE_HOLD_RELEASE_NEXT:-1}
    ORACLE_FORCE_INITIAL_PROMPT=${ORACLE_FORCE_INITIAL_PROMPT:-1}
    ORACLE_INITIAL_STAGE_LOCK=${ORACLE_INITIAL_STAGE_LOCK:-1}
    ORACLE_STAGE_ADVANCE_NEXT=${ORACLE_STAGE_ADVANCE_NEXT:-1}
    ORACLE_MONOTONIC_SEQUENCE_LOCK=${ORACLE_MONOTONIC_SEQUENCE_LOCK:-1}
    ORACLE_STAGE_LOCK_UNTIL_DONE=${ORACLE_STAGE_LOCK_UNTIL_DONE:-1}
    ;;
  vlm_free)
    ORACLE_HOLD_RELEASE_NEXT=${ORACLE_HOLD_RELEASE_NEXT:-0}
    ORACLE_FORCE_INITIAL_PROMPT=${ORACLE_FORCE_INITIAL_PROMPT:-0}
    ORACLE_INITIAL_STAGE_LOCK=${ORACLE_INITIAL_STAGE_LOCK:-0}
    ORACLE_STAGE_ADVANCE_NEXT=${ORACLE_STAGE_ADVANCE_NEXT:-0}
    ORACLE_MONOTONIC_SEQUENCE_LOCK=${ORACLE_MONOTONIC_SEQUENCE_LOCK:-0}
    ORACLE_STAGE_LOCK_UNTIL_DONE=${ORACLE_STAGE_LOCK_UNTIL_DONE:-0}
    ;;
  vlm_guarded)
    ORACLE_HOLD_RELEASE_NEXT=${ORACLE_HOLD_RELEASE_NEXT:-0}
    ORACLE_FORCE_INITIAL_PROMPT=${ORACLE_FORCE_INITIAL_PROMPT:-0}
    ORACLE_INITIAL_STAGE_LOCK=${ORACLE_INITIAL_STAGE_LOCK:-1}
    ORACLE_STAGE_ADVANCE_NEXT=${ORACLE_STAGE_ADVANCE_NEXT:-0}
    ORACLE_MONOTONIC_SEQUENCE_LOCK=${ORACLE_MONOTONIC_SEQUENCE_LOCK:-1}
    ORACLE_STAGE_LOCK_UNTIL_DONE=${ORACLE_STAGE_LOCK_UNTIL_DONE:-1}
    REQUIRE_INITIAL_VLM_SUBTASK=${REQUIRE_INITIAL_VLM_SUBTASK:-1}
    STRICT_HOLD_RELEASE_NEXT=${STRICT_HOLD_RELEASE_NEXT:-1}
    REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD=${REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD:-1}
    VLM_COMPLETED_SUBTASKS_MODE=${VLM_COMPLETED_SUBTASKS_MODE:-completed_struct}
    ;;
  *) echo "unsupported MODE=${MODE}; use oracle, vlm_free, or vlm_guarded" >&2; exit 2 ;;
esac

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_ID=${RUN_ID:-mw_orig35999_t${TASK_ID}_${MODE}_${STAMP}}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT is required}
OUT_ROOT=${OUT_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}
PORT=${PORT:-$((8800 + TASK_ID))}
NUM_TRIALS=${NUM_TRIALS:-2}
SEED=${SEED:-104}
MAX_STEPS=${MAX_STEPS:-2000}
REPLAN_STEPS=${REPLAN_STEPS:-5}
POST_HOLD_RELEASE_VLA_STEPS=${POST_HOLD_RELEASE_VLA_STEPS:-30}
POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS=${POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS:-0}
STRICT_HOLD_RELEASE_NEXT=${STRICT_HOLD_RELEASE_NEXT:-0}
REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD=${REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD:-0}
REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD_SUBTASKS=${REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD_SUBTASKS:-}
REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD=${REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD:-0}
MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD=${MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD:-0}
MICROWAVE_FORWARD_GAP_FILL_NEXT=${MICROWAVE_FORWARD_GAP_FILL_NEXT:-0}
MICROWAVE_STAGE_LOCK_UNTIL_DONE=${MICROWAVE_STAGE_LOCK_UNTIL_DONE:-0}
BLOCK_FORWARD_BEFORE_FIRST_STAGE_DONE=${BLOCK_FORWARD_BEFORE_FIRST_STAGE_DONE:-0}
PREVENT_COMPLETED_STAGE_REGRESSION=${PREVENT_COMPLETED_STAGE_REGRESSION:-0}
STOP_ON_STAGE_SUCCESS=${STOP_ON_STAGE_SUCCESS:-0}
MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION=${MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION:-dummy}
FORWARD_SWITCH_BLOCK_PREVIOUS=${FORWARD_SWITCH_BLOCK_PREVIOUS:-0}
PREVENT_SUBTASK_REGRESSION=${PREVENT_SUBTASK_REGRESSION:-1}
PREVENT_RELEASED_HOLD_REGRESSION=${PREVENT_RELEASED_HOLD_REGRESSION:-0}
PREVENT_HELD_SUBTASK_REGRESSION=${PREVENT_HELD_SUBTASK_REGRESSION:-0}
REGRESSION_GUARD_AFTER_HOLD_RELEASE=${REGRESSION_GUARD_AFTER_HOLD_RELEASE:-1}
HOLD_RELEASE_BLOCK_PAST_SUBTASKS=${HOLD_RELEASE_BLOCK_PAST_SUBTASKS:-0}
ALLOW_STAGE_DONE_RELEASE_ANCHOR=${ALLOW_STAGE_DONE_RELEASE_ANCHOR:-0}
ALLOW_AUTONOMOUS_FORWARD_RELEASE_ANCHOR=${ALLOW_AUTONOMOUS_FORWARD_RELEASE_ANCHOR:-0}
AUTONOMOUS_FORWARD_RELEASE_ANCHOR_SUBTASKS=${AUTONOMOUS_FORWARD_RELEASE_ANCHOR_SUBTASKS:-}
REQUIRE_INITIAL_VLM_SUBTASK=${REQUIRE_INITIAL_VLM_SUBTASK:-0}
VLM_COMPLETED_SUBTASKS_MODE=${VLM_COMPLETED_SUBTASKS_MODE:-off}
VLM_HOLD_STATE_HINT=${VLM_HOLD_STATE_HINT:-0}
VLM_HOLD_STATE_HINT_SUBTASKS=${VLM_HOLD_STATE_HINT_SUBTASKS:-}
VLM_HOLD_STATE_HINT_PHASE=${VLM_HOLD_STATE_HINT_PHASE:-active}
ENDPOSE_HOLD_SKIP_VLM_INFERENCE=${ENDPOSE_HOLD_SKIP_VLM_INFERENCE:-0}
COMPLETED_UPDATE_FROM_OFFICIAL_STAGE=${COMPLETED_UPDATE_FROM_OFFICIAL_STAGE:-0}
ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE=${ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE:-}
ENDPOSE_HOLD_CONSECUTIVE_BY_SUBTASK_JSON=${ENDPOSE_HOLD_CONSECUTIVE_BY_SUBTASK_JSON:-}
ENDPOSE_PLACE_HOLD_MIN_STEPS_BEFORE_RELEASE=${ENDPOSE_PLACE_HOLD_MIN_STEPS_BEFORE_RELEASE:-0}
ENDPOSE_PLACE_RELEASE_EEF_GUARD=${ENDPOSE_PLACE_RELEASE_EEF_GUARD:-0}
ENDPOSE_PLACE_RELEASE_EEF_GUARD_GRIPPER_VALUE=${ENDPOSE_PLACE_RELEASE_EEF_GUARD_GRIPPER_VALUE:-1.0}
ENDPOSE_PLACE_OBJECT_GATE_JSON=${ENDPOSE_PLACE_OBJECT_GATE_JSON:-}
SUBTASK_RELEASE_ANCHORS_JSON=${SUBTASK_RELEASE_ANCHORS_JSON:-${PACK_DIR}/config/release_anchors_empty.json}
INITIAL_SUBTASK_ANCHORS_JSON=${INITIAL_SUBTASK_ANCHORS_JSON:-}
ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE=${ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE:-0}
ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR=${ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR:-0}
REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE=${REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE:-0}
MICROWAVE_DEBUG_SAVE_VLM_FRAMES=${MICROWAVE_DEBUG_SAVE_VLM_FRAMES:-0}
MICROWAVE_REQUIRE_OPEN_EEF_HOLD_FOR_SUCCESS=${MICROWAVE_REQUIRE_OPEN_EEF_HOLD_FOR_SUCCESS:-0}
ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT_EXCLUDE_SUBTASKS=${ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT_EXCLUDE_SUBTASKS:-}
ENDPOSE_HOLD_TARGETS_JSON=${ENDPOSE_HOLD_TARGETS_JSON:-${PACK_DIR}/config/tasks2_26_endpose_targets_seed100_199.json}
ENDPOSE_TARGET_PASSAGE_COUNTS_JSON=${ENDPOSE_TARGET_PASSAGE_COUNTS_JSON:-${PACK_DIR}/config/tasks2_26_target_passage_counts_seed100_199_alltasks_tol045_20260624_074452.json}
ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE=${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE:-}
ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON=${ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON:-}

for required in \
  "${REFERENCE_DIR}/eval_tasks2_26_vlm_vla.py" \
  "${REFERENCE_DIR}/fullvlm_v2_26_memory_tasks.json" \
  "${ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR}/eval_common.py" \
  "${ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR}/task2_26_reference_stage.py" \
  "${ROBOMEMARENA_OFFICIAL_BDDL_DIR}" \
  "${ROBOMEMARENA_ROOT_BDDL_DIR}" \
  "${TARGET_LIBERO_PATH}" \
  "${VLA_POLICY}" \
  "${VLA_REPO_ID}" \
  "${VLA_SERVER_PY}" \
  "${VLM_CKPT}" \
  "${PACK_DIR}/evaluators/eval_tasks2_26_sync_endpose_hold_officialscore.py" \
  "${PACK_DIR}/evaluators/run_tasks2_26_sync_hold_eval.sh" \
  "${ENDPOSE_HOLD_TARGETS_JSON}" \
  "${ENDPOSE_TARGET_PASSAGE_COUNTS_JSON}" \
  "${SUBTASK_RELEASE_ANCHORS_JSON}"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 3; }
done
if [[ -n "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE}" && ! -e "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE}" ]]; then
  echo "missing required path: ${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE}" >&2
  exit 3
fi
if [[ -n "${ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON}" && ! -e "${ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON}" ]]; then
  echo "missing required path: ${ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON}" >&2
  exit 3
fi
if [[ -n "${INITIAL_SUBTASK_ANCHORS_JSON}" && ! -e "${INITIAL_SUBTASK_ANCHORS_JSON}" ]]; then
  echo "missing required path: ${INITIAL_SUBTASK_ANCHORS_JSON}" >&2
  exit 3
fi
if [[ -n "${ENDPOSE_PLACE_OBJECT_GATE_JSON}" && ! -e "${ENDPOSE_PLACE_OBJECT_GATE_JSON}" ]]; then
  echo "missing required path: ${ENDPOSE_PLACE_OBJECT_GATE_JSON}" >&2
  exit 3
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/videos" "${OUT_ROOT}/code_snapshot"
on_run_error() {
  local rc=$?
  trap - ERR
  {
    echo "status=failed"
    echo "mode=${MODE}"
    echo "task=${TASK_ID}"
    echo "exit_code=${rc}"
    echo "out_root=${OUT_ROOT}"
    echo "summary=${OUT_ROOT}/summary.tsv"
    echo "failed_at=$(date -Is)"
  } > "${OUT_ROOT}/LIVE_STATUS.txt"
  exit "${rc}"
}
trap on_run_error ERR
mkdir -p \
  "${OUT_ROOT}/code_snapshot/scripts" \
  "${OUT_ROOT}/code_snapshot/norm_repo" \
  "${OUT_ROOT}/code_snapshot/official_scripts" \
  "${OUT_ROOT}/code_snapshot/official_reference" \
  "${OUT_ROOT}/code_snapshot/official_bddl" \
  "${OUT_ROOT}/code_snapshot/official_root_bddl"
cp "${BASH_SOURCE[0]}" "${OUT_ROOT}/code_snapshot/"
if [[ -n "${REPRO_ENTRY_LAUNCHER:-}" && -f "${REPRO_ENTRY_LAUNCHER}" ]]; then
  cp "${REPRO_ENTRY_LAUNCHER}" "${OUT_ROOT}/code_snapshot/"
fi
cp "${PACK_DIR}/scripts/run_microwave_eefonly_no_object_gate.sh" "${OUT_ROOT}/code_snapshot/"
cp "${PACK_DIR}/scripts/serve_policy_custom_repo.py" "${OUT_ROOT}/code_snapshot/"
cp "${PACK_DIR}/README.md" "${OUT_ROOT}/code_snapshot/"
cp "${PACK_DIR}/evaluators/"*.py "${OUT_ROOT}/code_snapshot/"
cp "${PACK_DIR}/evaluators/"*.sh "${OUT_ROOT}/code_snapshot/"
cp "${PACK_DIR}/config/"*.json "${OUT_ROOT}/code_snapshot/"
for helper in build_microwave_deep_eef_targets.py build_microwave_pick_contact_eef_targets.py; do
  [[ -f "${PACK_DIR}/scripts/${helper}" ]] && cp "${PACK_DIR}/scripts/${helper}" "${OUT_ROOT}/code_snapshot/"
done
cp "${PACK_DIR}/scripts/"*.sh "${OUT_ROOT}/code_snapshot/scripts/"
cp "${PACK_DIR}/scripts/"*.py "${OUT_ROOT}/code_snapshot/scripts/"
cp "${VLA_REPO_ID}/norm_stats.json" "${OUT_ROOT}/code_snapshot/norm_repo/"
if [[ -f "${VLA_REPO_ID}/SOURCE.txt" ]]; then
  cp "${VLA_REPO_ID}/SOURCE.txt" "${OUT_ROOT}/code_snapshot/norm_repo/"
fi
cp "${ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR}/eval_common.py" "${OUT_ROOT}/code_snapshot/official_scripts/"
cp "${ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR}/task2_26_reference_stage.py" "${OUT_ROOT}/code_snapshot/official_scripts/"
cp "${REFERENCE_DIR}/eval_tasks2_26_vlm_vla.py" "${OUT_ROOT}/code_snapshot/official_reference/"
cp "${REFERENCE_DIR}/fullvlm_v2_26_memory_tasks.json" "${OUT_ROOT}/code_snapshot/official_reference/"
cp -R "${ROBOMEMARENA_OFFICIAL_BDDL_DIR}/." "${OUT_ROOT}/code_snapshot/official_bddl/"
cp -R "${ROBOMEMARENA_ROOT_BDDL_DIR}/." "${OUT_ROOT}/code_snapshot/official_root_bddl/"
printf '%s\n' "${ROBOMEMARENA_OFFICIAL_COMMIT}" > "${OUT_ROOT}/code_snapshot/official_commit.txt"

cat > "${OUT_ROOT}/run_manifest.json" <<JSON
{
  "created_at": "${STAMP}",
  "mode": "${MODE}",
  "oracle_hold_release_next": ${ORACLE_HOLD_RELEASE_NEXT},
  "oracle_force_initial_prompt": ${ORACLE_FORCE_INITIAL_PROMPT},
  "oracle_initial_stage_lock": ${ORACLE_INITIAL_STAGE_LOCK},
  "oracle_stage_advance_next": ${ORACLE_STAGE_ADVANCE_NEXT},
  "oracle_monotonic_sequence_lock": ${ORACLE_MONOTONIC_SEQUENCE_LOCK},
  "oracle_stage_lock_until_done": ${ORACLE_STAGE_LOCK_UNTIL_DONE},
  "task_id": ${TASK_ID},
  "num_trials": ${NUM_TRIALS},
  "seed": ${SEED},
  "max_steps": ${MAX_STEPS},
  "replan_steps": ${REPLAN_STEPS},
  "post_hold_release_vla_steps": ${POST_HOLD_RELEASE_VLA_STEPS},
  "post_pick_hold_release_same_prompt_steps": ${POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS},
  "post_pick_release_keep_gripper_steps": ${POST_PICK_RELEASE_KEEP_GRIPPER_STEPS:-0},
  "post_pick_release_keep_gripper_value": ${POST_PICK_RELEASE_KEEP_GRIPPER_VALUE:-1.0},
  "strict_hold_release_next": ${STRICT_HOLD_RELEASE_NEXT},
  "require_hold_release_for_pick_forward": ${REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD},
  "require_hold_release_for_pick_forward_subtasks": "${REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD_SUBTASKS}",
  "require_hold_release_for_place_forward": ${REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD},
  "microwave_forward_require_prior_hold": ${MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD},
  "microwave_forward_gap_fill_next": ${MICROWAVE_FORWARD_GAP_FILL_NEXT},
  "microwave_stage_lock_until_done": ${MICROWAVE_STAGE_LOCK_UNTIL_DONE},
  "block_forward_before_first_stage_done": ${BLOCK_FORWARD_BEFORE_FIRST_STAGE_DONE},
  "prevent_completed_stage_regression": ${PREVENT_COMPLETED_STAGE_REGRESSION},
  "stop_on_stage_success": ${STOP_ON_STAGE_SUCCESS},
  "microwave_forward_blocked_no_current_action": "${MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION}",
  "forward_switch_block_previous": ${FORWARD_SWITCH_BLOCK_PREVIOUS},
  "prevent_subtask_regression": ${PREVENT_SUBTASK_REGRESSION},
  "prevent_released_hold_regression": ${PREVENT_RELEASED_HOLD_REGRESSION},
  "prevent_held_subtask_regression": ${PREVENT_HELD_SUBTASK_REGRESSION},
  "regression_guard_after_hold_release": ${REGRESSION_GUARD_AFTER_HOLD_RELEASE},
  "hold_release_block_past_subtasks": ${HOLD_RELEASE_BLOCK_PAST_SUBTASKS},
  "allow_stage_done_release_anchor": ${ALLOW_STAGE_DONE_RELEASE_ANCHOR},
  "allow_autonomous_forward_release_anchor": ${ALLOW_AUTONOMOUS_FORWARD_RELEASE_ANCHOR},
  "autonomous_forward_release_anchor_subtasks": "${AUTONOMOUS_FORWARD_RELEASE_ANCHOR_SUBTASKS}",
  "require_initial_vlm_subtask": ${REQUIRE_INITIAL_VLM_SUBTASK},
  "require_open_microwave_endpose_hold_before_release": ${REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE},
  "microwave_debug_save_vlm_frames": ${MICROWAVE_DEBUG_SAVE_VLM_FRAMES},
  "microwave_require_open_eef_hold_for_success": ${MICROWAVE_REQUIRE_OPEN_EEF_HOLD_FOR_SUCCESS},
  "endpose_hold_auto_resume_same_prompt_exclude_subtasks": "${ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT_EXCLUDE_SUBTASKS}",
  "endpose_hold_start_after_release_anchor": ${ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR:-0},
  "endpose_hold_start_after_release_anchor_subtasks": "${ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR_SUBTASKS:-}",
  "vlm_completed_subtasks_mode": "${VLM_COMPLETED_SUBTASKS_MODE}",
  "vlm_hold_state_hint": ${VLM_HOLD_STATE_HINT},
  "vlm_hold_state_hint_subtasks": "${VLM_HOLD_STATE_HINT_SUBTASKS}",
  "vlm_hold_state_hint_phase": "${VLM_HOLD_STATE_HINT_PHASE}",
  "endpose_hold_skip_vlm_inference": ${ENDPOSE_HOLD_SKIP_VLM_INFERENCE},
  "completed_update_from_official_stage": ${COMPLETED_UPDATE_FROM_OFFICIAL_STAGE},
  "endpose_hold_release_min_steps_by_subtask_file": "${ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE}",
  "endpose_hold_targets_json": "${ENDPOSE_HOLD_TARGETS_JSON:-}",
  "endpose_hold_pos_tol_by_subtask_file": "${ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE:-}",
  "endpose_target_passage_counts_json": "${ENDPOSE_TARGET_PASSAGE_COUNTS_JSON}",
  "endpose_hold_direction_signatures_json": "${ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON}",
  "endpose_hold_direction_cos_min": ${ENDPOSE_HOLD_DIRECTION_COS_MIN:-0.50},
  "endpose_hold_direction_window": ${ENDPOSE_HOLD_DIRECTION_WINDOW:-5},
  "endpose_hold_direction_min_displacement": ${ENDPOSE_HOLD_DIRECTION_MIN_DISPLACEMENT:-0.0005},
  "endpose_hold_direction_trend_eps": ${ENDPOSE_HOLD_DIRECTION_TREND_EPS:-0.005},
  "endpose_hold_pos_tol": ${ENDPOSE_HOLD_POS_TOL},
  "endpose_hold_eef_default_tol": ${ENDPOSE_HOLD_EEF_DEFAULT_TOL},
  "endpose_hold_eef_p95_extra_tol": ${ENDPOSE_HOLD_EEF_P95_EXTRA_TOL},
  "endpose_hold_eef_tol_cap": ${ENDPOSE_HOLD_EEF_TOL_CAP},
  "endpose_hold_consecutive_by_subtask_json": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${ENDPOSE_HOLD_CONSECUTIVE_BY_SUBTASK_JSON}"),
  "endpose_place_hold_min_steps_before_release": ${ENDPOSE_PLACE_HOLD_MIN_STEPS_BEFORE_RELEASE},
  "endpose_place_release_eef_guard": ${ENDPOSE_PLACE_RELEASE_EEF_GUARD},
  "endpose_place_release_eef_guard_gripper_value": ${ENDPOSE_PLACE_RELEASE_EEF_GUARD_GRIPPER_VALUE},
  "endpose_hold_auto_resume_same_prompt": ${ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT:-0},
  "endpose_hold_auto_resume_cooldown_steps": ${ENDPOSE_HOLD_AUTO_RESUME_COOLDOWN_STEPS:-50},
  "endpose_place_object_gate_json": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${ENDPOSE_PLACE_OBJECT_GATE_JSON}"),
  "endpose_pick_gripper_gate": ${ENDPOSE_PICK_GRIPPER_GATE:-1},
  "endpose_pick_deferred_gripper_release": ${ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE},
  "endpose_pick_object_lift_gate": ${ENDPOSE_PICK_OBJECT_LIFT_GATE:-1},
  "endpose_pick_object_lift_delta": ${ENDPOSE_PICK_OBJECT_LIFT_DELTA:-0.01},
  "endpose_pick_height_require_eef_near": ${ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR},
  "vla_policy": "${VLA_POLICY}",
  "vla_config": "${VLA_CONFIG}",
  "vla_repo_id": "${VLA_REPO_ID}",
  "vlm_ckpt": "${VLM_CKPT}",
  "subtask_release_anchors_json": "${SUBTASK_RELEASE_ANCHORS_JSON}",
  "initial_subtask_anchors_json": "${INITIAL_SUBTASK_ANCHORS_JSON}",
  "robomemarena_remote_root": "${REMOTE_ROOT}",
  "robomemarena_official_commit": "${ROBOMEMARENA_OFFICIAL_COMMIT}",
  "robomemarena_official_scripts_dir": "${ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR}",
  "robomemarena_official_bddl_dir": "${ROBOMEMARENA_OFFICIAL_BDDL_DIR}",
  "robomemarena_root_bddl_dir": "${ROBOMEMARENA_ROOT_BDDL_DIR}",
  "output_root": "${OUT_ROOT}",
  "pack_dir": "${PACK_DIR}",
  "repro_entry_launcher": "${REPRO_ENTRY_LAUNCHER:-}"
}
JSON
python3 -m json.tool "${OUT_ROOT}/run_manifest.json" >/dev/null
(
  cd "${OUT_ROOT}/code_snapshot"
  find . -type f ! -name artifact_sha256.tsv -print0 \
    | sort -z \
    | xargs -0 sha256sum > artifact_sha256.tsv
)

export OPENPI_ROOT INFER_ROOT TARGET_LIBERO_PATH
export VLA_POLICY VLA_CONFIG VLA_REPO_ID VLA_SERVER_PY
export VLM_CKPT VLM_LORA_PATH=none
export TASKS_JSON="[${TASK_ID}]" NUM_TRIALS SEED MAX_STEPS REPLAN_STEPS PORT
export RUN_ID OUT_ROOT
export EVAL_PY="${PACK_DIR}/evaluators/eval_tasks2_26_sync_endpose_hold_officialscore.py"
export TASKS2_26_BASE_EVAL_PY="${REFERENCE_DIR}/eval_tasks2_26_vlm_vla.py"
export TASK_CONFIG="${REFERENCE_DIR}/fullvlm_v2_26_memory_tasks.json"
export ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR ROBOMEMARENA_OFFICIAL_BDDL_DIR
export ENDPOSE_HOLD_TARGETS_JSON
export ENDPOSE_TARGET_PASSAGE_COUNTS_JSON
export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE
export ENDPOSE_HOLD_POS_TOL ENDPOSE_HOLD_EEF_DEFAULT_TOL ENDPOSE_HOLD_EEF_P95_EXTRA_TOL ENDPOSE_HOLD_EEF_TOL_CAP
export POST_HOLD_RELEASE_VLA_STEPS POST_HOLD_RELEASE_VLA_STEPS_BY_SUBTASK_JSON
export POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS STRICT_HOLD_RELEASE_NEXT
export REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD
export REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD_SUBTASKS
export REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD
export ALLOW_STAGE_DONE_RELEASE_ANCHOR
export ALLOW_AUTONOMOUS_FORWARD_RELEASE_ANCHOR AUTONOMOUS_FORWARD_RELEASE_ANCHOR_SUBTASKS
export REQUIRE_INITIAL_VLM_SUBTASK
export VLM_HOLD_STATE_HINT
export VLM_HOLD_STATE_HINT_SUBTASKS
export VLM_HOLD_STATE_HINT_PHASE
export ENDPOSE_HOLD_SKIP_VLM_INFERENCE
export COMPLETED_UPDATE_FROM_OFFICIAL_STAGE
export ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE
export ENDPOSE_HOLD_CONSECUTIVE_BY_SUBTASK_JSON
export ENDPOSE_PLACE_HOLD_MIN_STEPS_BEFORE_RELEASE
export ENDPOSE_PLACE_RELEASE_EEF_GUARD ENDPOSE_PLACE_RELEASE_EEF_GUARD_GRIPPER_VALUE
export ENDPOSE_PLACE_OBJECT_GATE_JSON
export PREVENT_SUBTASK_REGRESSION REGRESSION_GUARD_AFTER_HOLD_RELEASE HOLD_RELEASE_BLOCK_PAST_SUBTASKS
export PREVENT_RELEASED_HOLD_REGRESSION
export PREVENT_HELD_SUBTASK_REGRESSION
export FORWARD_SWITCH_BLOCK_PREVIOUS
export ENDPOSE_PICK_GRIPPER_GATE="${ENDPOSE_PICK_GRIPPER_GATE:-1}"
export ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE="${ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE:-0}"
export ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR
export ENDPOSE_PICK_OBJECT_LIFT_GATE="${ENDPOSE_PICK_OBJECT_LIFT_GATE:-1}"
export ENDPOSE_PICK_OBJECT_LIFT_DELTA="${ENDPOSE_PICK_OBJECT_LIFT_DELTA:-0.01}"
export REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE
export MICROWAVE_DEBUG_SAVE_VLM_FRAMES
export MICROWAVE_REQUIRE_OPEN_EEF_HOLD_FOR_SUCCESS
export ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT_EXCLUDE_SUBTASKS
export DRAWER_FORWARD_ADVANCE_GUARD=0
export DISABLE_OUTPUT_NORMALIZE=1 VLM_TASK_TEXT_MODE=english_reference_no_candidate VLM_COMPLETED_SUBTASKS_MODE
export ORACLE_HOLD_RELEASE_NEXT ORACLE_FORCE_INITIAL_PROMPT ORACLE_INITIAL_STAGE_LOCK ORACLE_STAGE_ADVANCE_NEXT ORACLE_MONOTONIC_SEQUENCE_LOCK
export ORACLE_STAGE_LOCK_UNTIL_DONE
export SUBTASK_RELEASE_ANCHORS_JSON INITIAL_SUBTASK_ANCHORS_JSON
export SERVER_LOG="${OUT_ROOT}/logs/vla_server.log"
export EVAL_LOG="${OUT_ROOT}/logs/eval_tasks2_26_sync_hold.log"
export DRIVER_LOG="${OUT_ROOT}/logs/driver.log"

{
  echo "status=starting"
  echo "mode=${MODE}"
  echo "task=${TASK_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "summary=${OUT_ROOT}/summary.tsv"
  echo "started_at=$(date -Is)"
} > "${OUT_ROOT}/LIVE_STATUS.txt"

bash "${PACK_DIR}/evaluators/run_tasks2_26_sync_hold_eval.sh"

{
  echo "status=finished"
  echo "mode=${MODE}"
  echo "task=${TASK_ID}"
  echo "out_root=${OUT_ROOT}"
  echo "summary=${OUT_ROOT}/summary.tsv"
  echo "finished_at=$(date -Is)"
} > "${OUT_ROOT}/LIVE_STATUS.txt"
