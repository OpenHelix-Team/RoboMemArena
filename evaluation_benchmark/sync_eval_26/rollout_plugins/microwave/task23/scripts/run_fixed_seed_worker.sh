#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUTS_FILE="${INPUTS_FILE:?INPUTS_FILE is required}"
WORKER_ID="${WORKER_ID:?WORKER_ID is required}"
WORKER_OUT_ROOT="${WORKER_OUT_ROOT:?WORKER_OUT_ROOT is required}"
REPEATS="${REPEATS:-4}"
FIXED_SEED="${FIXED_SEED:-105}"
PORT="${PORT:?PORT is required}"

[[ "${FIXED_SEED}" == "105" ]] || { echo "v155 is frozen to seed105" >&2; exit 2; }
[[ "${REPEATS}" == "4" ]] || { echo "v155 requires four repeats per worker" >&2; exit 2; }
[[ -f "${INPUTS_FILE}" ]] || { echo "missing ${INPUTS_FILE}" >&2; exit 2; }

mkdir -p "${WORKER_OUT_ROOT}"
printf 'version=v155\nworker_id=%s\nrepeats=%s\nfixed_seed=%s\nport=%s\ninputs_file=%s\n' \
  "${WORKER_ID}" "${REPEATS}" "${FIXED_SEED}" "${PORT}" "${INPUTS_FILE}" \
  > "${WORKER_OUT_ROOT}/worker_manifest.env"
printf 'repeat\tseed\trun_id\treturn_code\tout_root\n' > "${WORKER_OUT_ROOT}/worker_runs.tsv"

for repeat in 0 1 2 3; do
  run_id="task23_v155_fixedseed${FIXED_SEED}_worker${WORKER_ID}_repeat${repeat}"
  episode_out="${WORKER_OUT_ROOT}/repeat${repeat}"
  mkdir -p "${episode_out}"

  set +e
  INPUTS_FILE="${INPUTS_FILE}" \
  RUN_ID="${run_id}" \
  OUT_ROOT="${episode_out}" \
  PORT="${PORT}" \
  NUM_TRIALS=1 \
  SEED="${FIXED_SEED}" \
  bash "${ROOT}/run_task23_v155.sh" > "${episode_out}/worker.log" 2>&1
  rc=$?
  set -e

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${repeat}" "${FIXED_SEED}" "${run_id}" "${rc}" "${episode_out}" \
    >> "${WORKER_OUT_ROOT}/worker_runs.tsv"
done
