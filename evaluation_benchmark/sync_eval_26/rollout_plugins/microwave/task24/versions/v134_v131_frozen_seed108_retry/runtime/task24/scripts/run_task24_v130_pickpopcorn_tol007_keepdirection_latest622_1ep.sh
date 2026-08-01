#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# v129 reached 0.06547 m from the recorded pick-popcorn EEF target while the
# applied 0.060 m tolerance prevented the autonomous place prompt release.
export ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE=${PACK_DIR}/config/task24_pickpopcorn_tol007_20260718.json
export REPRO_ENTRY_LAUNCHER=${BASH_SOURCE[0]}
exec bash "${PACK_DIR}/scripts/run_task24_v123_strict_adjacent_latest622_1ep.sh"
