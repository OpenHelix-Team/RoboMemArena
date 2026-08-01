#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep the complete v108 rollout contract. The only behavior change is that,
# after VLM autonomously asks to switch pick -> place from an active EEF hold,
# the VLA gets 50 final steps under the existing pick prompt before the switch.
export POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS=${POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS:-50}
export RUN_ID=${RUN_ID:-mw_orig35999_t21_v110_historicalvlm_eef_pickfinish50_latest622_$(date +%Y%m%d_%H%M%S)}
export REPRO_SNAPSHOT_HELPER=${REPRO_SNAPSHOT_HELPER:-${PACK_DIR}/scripts/snapshot_task21_v121.sh}
export REPRO_ENTRY_LAUNCHER=${REPRO_ENTRY_LAUNCHER:-${BASH_SOURCE[0]}}

exec bash "${PACK_DIR}/scripts/run_task21_v108_historicalvlm_eef_latest622_1ep.sh"
