#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
REMOTE_ROOT=${ROBOMEMARENA_REMOTE_ROOT:?ROBOMEMARENA_REMOTE_ROOT is required}
SNAP="${OUT_ROOT}/exact_repro_snapshot"

mkdir -p "${SNAP}"/{assets/norm_repo,config,evaluators,scripts,official_scorer,official_bddl,official_root_bddl}
cp -p "${PACK_DIR}/assets/norm_repo/norm_stats.json" "${SNAP}/assets/norm_repo/"
cp -p "${PACK_DIR}/config/"*.json "${SNAP}/config/"
if [[ -n "${SUBTASK_RELEASE_ANCHORS_JSON:-}" ]]; then
  cp -p "${SUBTASK_RELEASE_ANCHORS_JSON}" "${SNAP}/config/active_release_anchors.json"
fi
cp -p "${PACK_DIR}/evaluators/"*.py "${PACK_DIR}/evaluators/"*.sh "${SNAP}/evaluators/"
cp -p "${PACK_DIR}/scripts/"*.py "${PACK_DIR}/scripts/"*.sh "${SNAP}/scripts/"
cp -p "${REMOTE_ROOT}/evaluation_benchmark/scripts/eval_common.py" \
  "${REMOTE_ROOT}/evaluation_benchmark/scripts/task2_26_reference_stage.py" \
  "${SNAP}/official_scorer/"
cp -R "${REMOTE_ROOT}/evaluation_benchmark/bddl/." "${SNAP}/official_bddl/"
cp -R "${REMOTE_ROOT}/bddl/." "${SNAP}/official_root_bddl/"
git -C "${REMOTE_ROOT}" rev-parse HEAD >"${SNAP}/official_commit.txt"
(
  cd "${SNAP}"
  find . -type f ! -name artifact_sha256.tsv -print0 | sort -z | xargs -0 sha256sum >artifact_sha256.tsv
)
printf '[TASK21_V121_SNAPSHOT_OK] %s\n' "${SNAP}"
