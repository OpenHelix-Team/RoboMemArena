#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ID="${TASK_ID:?Set TASK_ID.}"
VLM_CKPT="${VLM_CKPT:?Set VLM_CKPT.}"
EVALUATOR_FILE_OVERRIDE="${EVALUATOR_FILE_OVERRIDE:-}"
PARTITION="${PARTITION:-acd_u}"
NODE="${NODE:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="count_auto_t${TASK_ID}_${STAMP}"
JOB_NAME="count_auto_t${TASK_ID}_${STAMP}"
NUM_TRIALS="${NUM_TRIALS:-2}"
SEED="${SEED:-104}"
VLM_INTERVAL="${VLM_INTERVAL:-5}"
HOLD_AFTER_REQUIRED_STAGES="${HOLD_AFTER_REQUIRED_STAGES:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
POST_STAGE_STEPS="${POST_STAGE_STEPS:-30}"
PORT="${PORT:-29600}"
RUN_ID="${RUN_ID:-task${TASK_ID}_autonomous_d9f83ac_vla35999_${NUM_TRIALS}ep_${STAMP}}"
RUNS_BASE="${RUNS_BASE:-${PACK_DIR}/runs_autonomous}"
OUT_ROOT="${OUT_ROOT:-${RUNS_BASE}/${RUN_ID}}"
LOG_DIR="${LOG_DIR_OVERRIDE:-${PACK_DIR}/logs_autonomous}"
SUBMIT_LOG="${LOG_DIR}/${JOB_NAME}.log"
RUNTIME_HOME="${RUNTIME_HOME:-${HOME}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MAX_MEM_PER_CPU_MB="$(scontrol show partition "${PARTITION}" | sed -n 's/.*MaxMemPerCPU=\([0-9][0-9]*\).*/\1/p')"
if [[ -n "${MAX_MEM_PER_CPU_MB}" ]]; then
  MEM_MB="$((CPUS_PER_TASK * MAX_MEM_PER_CPU_MB))"
else
  MEM_MB="${MEM_MB:-240000}"
fi

mkdir -p "${LOG_DIR}" "${RUNS_BASE}"
NODE_ARG=""
if [[ -n "${NODE}" ]]; then
  NODE_ARG="--nodelist=${NODE}"
fi

tmux new-session -d -s "${SESSION}" \
  "bash -lc 'srun -p ${PARTITION} ${NODE_ARG} --gres=gpu:2 -c ${CPUS_PER_TASK} --mem=${MEM_MB}M --job-name=${JOB_NAME} env TASK_ID=${TASK_ID} VLM_CKPT=${VLM_CKPT} EVALUATOR_FILE_OVERRIDE=${EVALUATOR_FILE_OVERRIDE} NUM_TRIALS=${NUM_TRIALS} SEED=${SEED} REPLAN_STEPS=${REPLAN_STEPS} POST_STAGE_STEPS=${POST_STAGE_STEPS} VLM_INTERVAL=${VLM_INTERVAL} HOLD_AFTER_REQUIRED_STAGES=${HOLD_AFTER_REQUIRED_STAGES} PORT=${PORT} RUN_ID=${RUN_ID} OUT_ROOT=${OUT_ROOT} RUNTIME_HOME=${RUNTIME_HOME} bash ${PACK_DIR}/scripts/run_autonomous_task.sh 2>&1 | tee ${SUBMIT_LOG}'"

echo "session=${SESSION}"
echo "job_name=${JOB_NAME}"
echo "partition=${PARTITION}"
echo "node=${NODE:-scheduler}"
echo "run_id=${RUN_ID}"
echo "vlm_interval=${VLM_INTERVAL}"
echo "hold_after_required_stages=${HOLD_AFTER_REQUIRED_STAGES}"
echo "replan_steps=${REPLAN_STEPS}"
echo "post_stage_steps=${POST_STAGE_STEPS}"
echo "out_root=${OUT_ROOT}"
echo "submit_log=${SUBMIT_LOG}"
echo "mem_mb=${MEM_MB}"
tmux ls | grep -F "${SESSION}"
