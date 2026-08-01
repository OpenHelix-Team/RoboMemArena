#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the formal output root}"
: "${CUDA_VISIBLE_DEVICES:?run inside a two-GPU Slurm allocation}"

if [[ "${CPU_AFFINITY_MODE:-}" != "none" ]]; then
  echo "strict exact20 must be launched with --cpu-bind=none through launch_exact20_dual_gpu_slurm.sh" >&2
  echo "for a separately verified persistent allocation, set CPU_AFFINITY_MODE=none explicitly" >&2
  exit 2
fi

IFS=',' read -r -a GPU_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_DEVICES[@]}" -ne 2 ]] || [[ -z "${GPU_DEVICES[0]}" ]] || [[ -z "${GPU_DEVICES[1]}" ]]; then
  echo "strict exact20 requires exactly two visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ "${GPU_DEVICES[0]}" == "${GPU_DEVICES[1]}" ]]; then
  echo "strict exact20 requires two distinct visible GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to verify the locked official runtime" >&2
  exit 2
fi

umask 0002
mkdir -p "${OUTPUT_ROOT}/_allocation"
METADATA_PATH="${OUTPUT_ROOT}/_allocation/dual_gpu_${SLURM_JOB_ID:-local}_${SLURM_STEP_ID:-0}.json"
"${PYTHON_BIN}" - "${METADATA_PATH}" "${REPO_ROOT}" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": sys.argv[2],
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "slurm": {
        key: os.environ[key]
        for key in (
            "SLURM_JOB_ID",
            "SLURM_STEP_ID",
            "SLURM_JOB_NAME",
            "SLURM_JOB_NODELIST",
            "SLURM_JOB_ACCOUNT",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_JOB_PARTITION",
        )
        if key in os.environ
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" evaluation_benchmark/sync_eval_26/scripts/schedule_exact20.py \
  --tasks "${TASKS:-all}" \
  --episodes "${EPISODES:-20}" \
  --output-root "${OUTPUT_ROOT}" \
  --gpu-bundles "${CUDA_VISIBLE_DEVICES}" \
  "$@"
