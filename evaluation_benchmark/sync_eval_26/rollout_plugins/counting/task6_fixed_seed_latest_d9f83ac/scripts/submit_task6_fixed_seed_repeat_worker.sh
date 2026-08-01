#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_GROUP="${RUN_GROUP:?Set RUN_GROUP.}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
REPEAT_START="${REPEAT_START:?Set REPEAT_START.}"
REPEAT_COUNT="${REPEAT_COUNT:?Set REPEAT_COUNT.}"
VLM_CKPT="${VLM_CKPT:?Set VLM_CKPT.}"
PARTITION="${PARTITION:-acd_u}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-120000}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="task6_seedrepeat_w${WORKER_ID}_${STAMP}"
JOB_NAME="task6_seedrepeat_w${WORKER_ID}_${STAMP}"
RUNS_BASE="${RUNS_BASE:-${PACK_DIR}/runs_autonomous/${RUN_GROUP}/workers/worker${WORKER_ID}}"
LOG_DIR="${LOG_DIR:-${PACK_DIR}/logs_autonomous/${RUN_GROUP}/worker${WORKER_ID}}"
RUNTIME_HOME="${RUNTIME_HOME:-${HOME}}"

mkdir -p "${RUNS_BASE}" "${LOG_DIR}"
tmux new-session -d -s "${SESSION}" \
  "bash -lc 'srun -p ${PARTITION} --gres=gpu:2 -c ${CPUS_PER_TASK} --mem=${MEM_MB}M --job-name=${JOB_NAME} env RUN_GROUP=${RUN_GROUP} WORKER_ID=${WORKER_ID} REPEAT_START=${REPEAT_START} REPEAT_COUNT=${REPEAT_COUNT} FIXED_SEED=100 VLM_CKPT=${VLM_CKPT} RUNS_BASE=${RUNS_BASE} LOG_DIR=${LOG_DIR} RUNTIME_HOME=${RUNTIME_HOME} bash ${PACK_DIR}/scripts/run_task6_fixed_seed_repeat_worker.sh > ${LOG_DIR}/${JOB_NAME}.log 2>&1'"

printf 'session=%s\njob_name=%s\nrun_group=%s\nworker_id=%s\nrepeat_start=%s\nrepeat_count=%s\n' \
  "${SESSION}" "${JOB_NAME}" "${RUN_GROUP}" "${WORKER_ID}" "${REPEAT_START}" "${REPEAT_COUNT}"
