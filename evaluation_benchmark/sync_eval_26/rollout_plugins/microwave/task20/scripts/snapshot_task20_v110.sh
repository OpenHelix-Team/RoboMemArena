#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED_OFFICIAL_COMMIT=d9f83ac5182e25ad7f0a301a77a0b667f2392df1
: "${OUT_ROOT:?OUT_ROOT is required}"
: "${ROBOMEMARENA_REMOTE_ROOT:?ROBOMEMARENA_REMOTE_ROOT is required}"

SNAPSHOT_DIR="${OUT_ROOT}/code_snapshot"
mkdir -p "${SNAPSHOT_DIR}/package" "${SNAPSHOT_DIR}/official"

for name in README.md FROZEN_PROVENANCE.md paths.example.env; do
  [[ -f "${PACK_DIR}/${name}" ]] && cp -p "${PACK_DIR}/${name}" "${SNAPSHOT_DIR}/package/"
done
for name in assets config evaluators scripts versions; do
  [[ -d "${PACK_DIR}/${name}" ]] && cp -a "${PACK_DIR}/${name}" "${SNAPSHOT_DIR}/package/"
done

OFFICIAL_SCRIPTS="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/scripts"
OFFICIAL_REFERENCE="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference"
cp -p "${OFFICIAL_SCRIPTS}/eval_common.py" "${SNAPSHOT_DIR}/official/"
cp -p "${OFFICIAL_SCRIPTS}/task2_26_reference_stage.py" "${SNAPSHOT_DIR}/official/"
cp -p "${OFFICIAL_REFERENCE}/eval_tasks2_26_vlm_vla.py" "${SNAPSHOT_DIR}/official/"
cp -p "${OFFICIAL_REFERENCE}/fullvlm_v2_26_memory_tasks.json" "${SNAPSHOT_DIR}/official/"
cp -p "${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/bddl/20_"*.bddl "${SNAPSHOT_DIR}/official/"
cp -p "${ROBOMEMARENA_REMOTE_ROOT}/bddl/20_"*.bddl "${SNAPSHOT_DIR}/official/"
if actual_commit="$(git -C "${ROBOMEMARENA_REMOTE_ROOT}" rev-parse HEAD 2>/dev/null)"; then
  :
elif [[ -f "${ROBOMEMARENA_REMOTE_ROOT}/COMMIT" ]]; then
  actual_commit="$(tr -d '[:space:]' <"${ROBOMEMARENA_REMOTE_ROOT}/COMMIT")"
else
  echo "official checkout has neither git metadata nor a COMMIT marker" >&2
  exit 3
fi
if [[ "${actual_commit}" != "${EXPECTED_OFFICIAL_COMMIT}" ]]; then
  echo "official scorer mismatch: expected=${EXPECTED_OFFICIAL_COMMIT} actual=${actual_commit}" >&2
  exit 3
fi
printf '%s\n' "${actual_commit}" >"${SNAPSHOT_DIR}/official_commit.txt"
printf 'variant=%s\nvlm_variant_id=%s\n' \
  'task20_v110_placecookies11_latest622' \
  "${VLM_VARIANT_ID:-task20_mwvlm_no_completed_v49_ckpt24}" \
  >"${SNAPSHOT_DIR}/variant.env"

(
  cd "${SNAPSHOT_DIR}"
  find . -type f ! -name artifact_sha256.tsv -print0 \
    | sort -z \
    | xargs -0 sha256sum >artifact_sha256.tsv
)

echo "[REPRO_SNAPSHOT_V110_OK] ${SNAPSHOT_DIR}"
