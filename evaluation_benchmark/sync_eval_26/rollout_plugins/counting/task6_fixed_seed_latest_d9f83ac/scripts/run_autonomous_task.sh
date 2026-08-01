#!/usr/bin/env bash
set -euo pipefail
umask 0002

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_COMMIT="d9f83ac5182e25ad7f0a301a77a0b667f2392df1"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the frozen RoboMemArena d9f83ac checkout.}"
EVAL_DIR="${SOURCE_ROOT}/evaluation_benchmark/async_vlm26_reference"
OFFICIAL_EVALUATOR_FILE="${EVAL_DIR}/eval_fullvlm26_async_vlm_vla.py"
EVALUATOR_FILE="${EVALUATOR_FILE_OVERRIDE:-${OFFICIAL_EVALUATOR_FILE}}"
SCORER_FILE="${SOURCE_ROOT}/evaluation_benchmark/scripts/task2_26_reference_stage.py"
TASK_CONFIG_FILE="${EVAL_DIR}/fullvlm_v2_26_memory_tasks.json"
SERVER_ENTRYPOINT="${PACK_DIR}/scripts/serve_policy_selfcontained.py"

TASK_ID="${TASK_ID:?Set TASK_ID.}"
VLM_CKPT="${VLM_CKPT:?Set VLM_CKPT to a trained VLM checkpoint.}"
VLA_CKPT="${VLA_CKPT:?Set VLA_CKPT to the original fullvlm-v2 checkpoint 35999.}"
VLA_CONFIG="${VLA_CONFIG:-pi05_libero_robomemarena_fullvlm_v2_noflip_dataset}"
OPENPI_ROOT="${OPENPI_ROOT:?Set OPENPI_ROOT to the OpenPI checkout.}"
OPENPI_INFERENCE_ROOT="${OPENPI_INFERENCE_ROOT:?Set OPENPI_INFERENCE_ROOT to the OpenPI inference checkout.}"
TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH:-${SOURCE_ROOT}/evaluation_benchmark/libero_fork}"
RUNTIME_HOME="${RUNTIME_HOME:-${HOME}}"

NUM_TRIALS="${NUM_TRIALS:-2}"
SEED="${SEED:-104}"
MAX_STEPS="${MAX_STEPS:-2500}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
POST_STAGE_STEPS="${POST_STAGE_STEPS:-30}"
ASYNC_VLM="${ASYNC_VLM:-0}"
VLM_INTERVAL="${VLM_INTERVAL:-5}"
HOLD_AFTER_REQUIRED_STAGES="${HOLD_AFTER_REQUIRED_STAGES:-0}"
PORT="${PORT:-29600}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-task${TASK_ID}_autonomous_d9f83ac_vla35999_${NUM_TRIALS}ep_${STAMP}}"
OUT_ROOT="${OUT_ROOT:-${PACK_DIR}/runs_autonomous/${RUN_ID}}"

NORM_FILE="${VLA_NORM_FILE:-${VLA_CKPT}/assets/robomemarena_fullvlm_v2_noflip_dataset_v2/norm_stats.json}"
VLM_MODEL_FILE="${VLM_CKPT}/model.safetensors"
for path in "${OFFICIAL_EVALUATOR_FILE}" "${EVALUATOR_FILE}" "${SCORER_FILE}" "${TASK_CONFIG_FILE}" "${SERVER_ENTRYPOINT}" "${NORM_FILE}" "${VLM_MODEL_FILE}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] required file missing: ${path}" >&2
    exit 1
  fi
done

if [[ "${HOLD_AFTER_REQUIRED_STAGES}" == "1" && ( "${TASK_ID}" == "6" || "${TASK_ID}" == "7" ) && "${REPLAN_STEPS}" != "1" ]]; then
  echo "[ERROR] Counting-task post-Pour2 hold requires REPLAN_STEPS=1 to avoid leftover chunk actions." >&2
  exit 2
fi
for path in "${VLM_CKPT}" "${VLA_CKPT}" "${TARGET_LIBERO_PATH}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] required directory missing: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/videos"
VLM_MODEL_REALPATH="$(readlink -f "${VLM_MODEL_FILE}")"

{
  echo "run_id=${RUN_ID}"
  echo "created_at=$(date -Is)"
  echo "submitted_user=$(whoami)"
  echo "hostname=$(hostname)"
  echo "runtime_home=${RUNTIME_HOME}"
  echo "slurm_job_id=${SLURM_JOB_ID:-none}"
  echo "slurm_job_account=${SLURM_JOB_ACCOUNT:-unknown}"
  echo "slurm_job_partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "package_git_commit=$(git -C "${PACK_DIR}" rev-parse HEAD)"
  echo "package_git_status=$(git -C "${PACK_DIR}" status --short | tr '\n' ';')"
  echo "remote_commit=${REMOTE_COMMIT}"
  echo "official_scorer_sha256=$(sha256sum "${SCORER_FILE}" | awk '{print $1}')"
  echo "official_evaluator_sha256=$(sha256sum "${OFFICIAL_EVALUATOR_FILE}" | awk '{print $1}')"
  echo "evaluator_entrypoint=${EVALUATOR_FILE}"
  echo "evaluator_entrypoint_sha256=$(sha256sum "${EVALUATOR_FILE}" | awk '{print $1}')"
  echo "task_config_sha256=$(sha256sum "${TASK_CONFIG_FILE}" | awk '{print $1}')"
  echo "selfcontained_server_sha256=$(sha256sum "${SERVER_ENTRYPOINT}" | awk '{print $1}')"
  echo "task_id=${TASK_ID}"
  echo "num_trials=${NUM_TRIALS}"
  echo "seed=${SEED}"
  echo "max_steps=${MAX_STEPS}"
  echo "replan_steps=${REPLAN_STEPS}"
  echo "post_stage_steps=${POST_STAGE_STEPS}"
  echo "async_vlm=${ASYNC_VLM}"
  echo "vlm_interval=${VLM_INTERVAL}"
  echo "hold_after_required_stages=${HOLD_AFTER_REQUIRED_STAGES}"
  echo "stage_prompt_override=off"
  echo "oracle_prompt_injection=off"
  echo "vlm_ckpt=${VLM_CKPT}"
  echo "vlm_model_realpath=${VLM_MODEL_REALPATH}"
  echo "vlm_model_size=$(stat -c '%s' "${VLM_MODEL_REALPATH}")"
  echo "vla_ckpt=${VLA_CKPT}"
  echo "vla_config=${VLA_CONFIG}"
  echo "norm_sha256=$(sha256sum "${NORM_FILE}" | awk '{print $1}')"
} | tee "${OUT_ROOT}/run_manifest.txt"

export OPENPI_ROOT OPENPI_INFERENCE_ROOT TARGET_LIBERO_PATH VLM_CKPT VLA_CKPT VLA_CONFIG PORT OUT_ROOT
export TASK_CONFIG="${TASK_CONFIG_FILE}"
export VIDEO_DIR="${OUT_ROOT}/videos"
export SUMMARY_JSON="${OUT_ROOT}/summary.json"
export SUMMARY_TSV="${OUT_ROOT}/summary.tsv"
export PROMPT_TRACE_TSV="${OUT_ROOT}/prompt_trace.tsv"
export TASKS_JSON="[${TASK_ID}]"
export NUM_TRIALS SEED MAX_STEPS REPLAN_STEPS POST_STAGE_STEPS ASYNC_VLM
export NUM_STEPS_WAIT=10
export FAIL_ON_EXTRA_POUR=1
export POST_GOAL_STEPS=200
export VLM_INTERVAL HOLD_AFTER_REQUIRED_STAGES
export VLM_QUEUE_SIZE=1
export N_RECENT=5
export K_MAX=0
export D_MERGE=6
export VLM_USE_WRIST=1
export VLM_USE_KEYFRAME_MEMORY=1
export VLM_INPUT_PROFILE=fullvlm_256
export VLM_DEVICE=cuda:0
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
SLOT_DEVICE="${CUDA_VISIBLE_DEVICES%%,*}"
VLA_LOCAL_DEVICE="${VLA_LOCAL_DEVICE:-${SLOT_DEVICE}}"
VLM_LOCAL_DEVICE="${VLM_LOCAL_DEVICE:-${SLOT_DEVICE}}"
export MUJOCO_EGL_DEVICE_ID="${VLM_LOCAL_DEVICE}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${TARGET_LIBERO_PATH}:${SOURCE_ROOT}/evaluation_benchmark/openpi_minimal_runtime:${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/packages/openpi/src:${OPENPI_ROOT}:${PYTHONPATH:-}"

SERVER_PY="${OPENPI_ROOT}/.venv/bin/python3"
EVAL_PY="${OPENPI_INFERENCE_ROOT}/.venv/bin/python"
SERVER_LOG="${OUT_ROOT}/logs/serve_policy.log"
EVAL_LOG="${OUT_ROOT}/logs/eval_autonomous.log"
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

HOME="${RUNTIME_HOME}" CUDA_VISIBLE_DEVICES="${VLA_LOCAL_DEVICE}" "${SERVER_PY}" "${SERVER_ENTRYPOINT}" --port "${PORT}" \
  policy:checkpoint --policy.config="${VLA_CONFIG}" --policy.dir="${VLA_CKPT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 180); do
  sleep 2
  if "${SERVER_PY}" - <<PY >/dev/null 2>&1
import socket
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", int("${PORT}")))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
  then
    READY=1
    echo "[INFO] VLA server ready at try ${i}" | tee -a "${EVAL_LOG}"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] VLA server exited early" | tee -a "${EVAL_LOG}"
    tail -n 200 "${SERVER_LOG}" | tee -a "${EVAL_LOG}" || true
    exit 1
  fi
done
if [[ "${READY}" -ne 1 ]]; then
  echo "[ERROR] VLA server did not become ready" | tee -a "${EVAL_LOG}"
  exit 1
fi

set +e
HOME="${RUNTIME_HOME}" CUDA_VISIBLE_DEVICES="${VLM_LOCAL_DEVICE}" "${EVAL_PY}" "${EVALUATOR_FILE}" 2>&1 | tee -a "${EVAL_LOG}"
RC=${PIPESTATUS[0]}
set -e

{
  echo "exit_code=${RC}"
  echo "finished_at=$(date -Is)"
  echo "summary_tsv=${OUT_ROOT}/summary.tsv"
  echo "video_dir=${OUT_ROOT}/videos"
} | tee "${OUT_ROOT}/run_result.txt"
exit "${RC}"
