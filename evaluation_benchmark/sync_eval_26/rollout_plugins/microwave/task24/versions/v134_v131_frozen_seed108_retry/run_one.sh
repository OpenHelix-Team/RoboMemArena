#!/usr/bin/env bash
set -euo pipefail

VERSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${VERSION_DIR}/runtime/task24"
EXPECTED_OFFICIAL_COMMIT=d9f83ac5182e25ad7f0a301a77a0b667f2392df1

: "${SEED:?set SEED}"
: "${PORT:?set PORT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

if [[ -n "${PRIVATE_INPUTS_FILE:-}" && -r "${PRIVATE_INPUTS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${PRIVATE_INPUTS_FILE}"
fi
TARGET_LIBERO_PATH_V131="${TARGET_LIBERO_PATH_V131:-${TARGET_LIBERO_PATH:-}}"
VLA_POLICY_V131="${VLA_POLICY_V131:-${VLA_POLICY:-}}"
VLA_REPO_ID_V131="${VLA_REPO_ID_V131:-${VLA_REPO_ID:-}}"
VLM_CKPT_TASK24_V131="${VLM_CKPT_TASK24_V131:-${VLM_CKPT:-}}"
ROBOMEMARENA_FULLVLM_DATA_ROOT="${ROBOMEMARENA_FULLVLM_DATA_ROOT:-${ASSET_FULLVLM_DATA_ROOT}}"
H5DUMP_BIN="${H5DUMP_BIN:-/share/anaconda3/bin/h5dump}"
for required in \
  OPENPI_ROOT INFER_ROOT TARGET_LIBERO_PATH_V131 ROBOMEMARENA_REMOTE_ROOT \
  ROBOMEMARENA_FULLVLM_DATA_ROOT VLA_POLICY_V131 VLA_REPO_ID_V131 VLM_CKPT_TASK24_V131 H5DUMP_BIN; do
  [[ -n "${!required:-}" && -e "${!required}" ]] || {
    echo "missing required private input ${required}" >&2
    exit 2
  }
done

export VLA_CONFIG="${VLA_CONFIG:-pi05_libero_robomemarena_fullvlm_v2_noflip_dataset}"
actual_commit="$(git -C "${ROBOMEMARENA_REMOTE_ROOT}" rev-parse HEAD)"
[[ "${actual_commit}" == "${EXPECTED_OFFICIAL_COMMIT}" ]] || {
  echo "official scorer mismatch: expected=${EXPECTED_OFFICIAL_COMMIT} actual=${actual_commit}" >&2
  exit 3
}
[[ -r "${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/scripts/task2_26_reference_stage.py" ]] || {
  echo "missing task2_26_reference_stage.py" >&2
  exit 3
}

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'PREFLIGHT_OK task=24 seed=%s scorer=%s\n' "${SEED}" "${actual_commit}"
  exit 0
fi

export MODE=vlm_free
export TASK_ID=24
export NUM_TRIALS=1
export MAX_STEPS="${MAX_STEPS:-2000}"
export REPLAN_STEPS="${REPLAN_STEPS:-5}"
export VLM_CKPT="${VLM_CKPT_TASK24_V131}"
export VLA_POLICY="${VLA_POLICY_V131}"
export VLA_REPO_ID="${VLA_REPO_ID_V131}"
export TARGET_LIBERO_PATH="${TARGET_LIBERO_PATH_V131}"
export VLA_POLICY VLA_CONFIG VLA_REPO_ID
export OPENPI_ROOT INFER_ROOT TARGET_LIBERO_PATH ROBOMEMARENA_REMOTE_ROOT
export ROBOMEMARENA_FULLVLM_DATA_ROOT
export VLA_SERVER_PY="${RUNTIME_DIR}/scripts/serve_policy_custom_repo.py"
export STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID="${RUN_ID:-task24_v131frozen_seed${SEED}_${STAMP}}"
export OUT_ROOT="${OUT_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}"
export REPRO_ENTRY_LAUNCHER="${BASH_SOURCE[0]}"

exec bash "${RUNTIME_DIR}/scripts/run_task24_v130_pickpopcorn_tol007_keepdirection_latest622_1ep.sh"
