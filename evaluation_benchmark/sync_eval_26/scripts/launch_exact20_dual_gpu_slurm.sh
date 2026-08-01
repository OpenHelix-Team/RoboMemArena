#!/usr/bin/env bash
set -euo pipefail

# Submit the strict two-GPU evaluator with the CPU affinity mode validated for
# LIBERO/MuJoCo. The inner runner is intentionally separate so an existing
# persistent allocation can still run it directly.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a new formal output root}"

PARTITION="${PARTITION:-acd_u}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM_MB="${MEM_MB:-163840}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
JOB_NAME="${JOB_NAME:-lhs_sync26_exact20_$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${TMUX:-}" && "${ALLOW_NON_TMUX:-0}" != "1" ]]; then
  echo "launch this long evaluation from tmux, or set ALLOW_NON_TMUX=1 intentionally" >&2
  exit 2
fi

exec srun \
  -p "${PARTITION}" \
  --gres=gpu:2 \
  -c "${CPUS_PER_TASK}" \
  --mem="${MEM_MB}M" \
  --time="${TIME_LIMIT}" \
  --cpu-bind=none \
  --job-name="${JOB_NAME}" \
  env CPU_AFFINITY_MODE=none \
  bash "${REPO_ROOT}/evaluation_benchmark/sync_eval_26/scripts/run_exact20_dual_gpu.sh" "$@"
