#!/usr/bin/env bash
set -euo pipefail

# Canonical fixed-seed repetition entrypoint for Task7.
# It starts 8 two-GPU workers with repeat counts 3,3,3,3,2,2,2,2.
# Every rollout uses environment seed=100 and VLA policy seed=100.

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_DIR="${PACK_DIR}/../../evaluation_campaigns/all18_repo_direct_exact20_20260725"
LAUNCHER="${CAMPAIGN_DIR}/scripts/submit_task7_fixedseed100_clean20_16gpu.sh"

[[ -x "${LAUNCHER}" ]] || { echo "missing fixed-seed launcher: ${LAUNCHER}" >&2; exit 2; }

exec bash "${LAUNCHER}"
