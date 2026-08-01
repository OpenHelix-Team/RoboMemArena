#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUT_ROOT:?OUT_ROOT is required}"
: "${ROBOMEMARENA_REMOTE_ROOT:?ROBOMEMARENA_REMOTE_ROOT is required}"

SNAPSHOT_DIR="${OUT_ROOT}/code_snapshot"
mkdir -p "${SNAPSHOT_DIR}/package" "${SNAPSHOT_DIR}/official"

for name in README.md FROZEN_PROVENANCE.md paths.example.env; do
  cp -p "${PACK_DIR}/${name}" "${SNAPSHOT_DIR}/package/"
done
for name in assets config evaluators scripts; do
  cp -a "${PACK_DIR}/${name}" "${SNAPSHOT_DIR}/package/"
done

OFFICIAL_SCRIPTS="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/scripts"
OFFICIAL_REFERENCE="${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference"
cp -p "${OFFICIAL_SCRIPTS}/eval_common.py" "${SNAPSHOT_DIR}/official/"
cp -p "${OFFICIAL_SCRIPTS}/task2_26_reference_stage.py" "${SNAPSHOT_DIR}/official/"
cp -p "${OFFICIAL_REFERENCE}/eval_tasks2_26_vlm_vla.py" "${SNAPSHOT_DIR}/official/"
cp -p "${OFFICIAL_REFERENCE}/fullvlm_v2_26_memory_tasks.json" "${SNAPSHOT_DIR}/official/"
cp -p "${ROBOMEMARENA_REMOTE_ROOT}/evaluation_benchmark/bddl/20_"*.bddl "${SNAPSHOT_DIR}/official/"
cp -p "${ROBOMEMARENA_REMOTE_ROOT}/bddl/20_"*.bddl "${SNAPSHOT_DIR}/official/"
git -C "${ROBOMEMARENA_REMOTE_ROOT}" rev-parse HEAD > "${SNAPSHOT_DIR}/official_commit.txt"

(
  cd "${SNAPSHOT_DIR}"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > artifact_sha256.tsv
)

echo "[REPRO_SNAPSHOT_OK] ${SNAPSHOT_DIR}"
