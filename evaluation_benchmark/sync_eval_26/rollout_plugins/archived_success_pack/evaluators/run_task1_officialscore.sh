#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=${ASSET_OPENPI_ROOT}
INFER_ROOT=${ASSET_OPENPI_INFERENCE_ROOT}
OPENPI_PYTHON=${OPENPI_ROOT}/.venv/bin/python3
INFER_PYTHON=${INFER_ROOT}/.venv/bin/python
EVAL_PY=${EVAL_PY:-${INFER_ROOT}/task1_eval/eval_task1_qwen3_sync_endpose_hold.py}

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID=${RUN_ID:-task1_sync5_endposehold_fullvlmv2noflip_successindex_seed104_${STAMP}}
PORT=${PORT:-8065}
LOG_BASE=${LOG_BASE:-${INFER_ROOT}/logs_task1_async}
RUN_ROOT=${LOG_BASE}/task1_sync/${RUN_ID}

LORA_PATH=${LORA_PATH:-none}
BASE_MODEL_DIR=${BASE_MODEL_DIR:-${INFER_ROOT}/output/task1_qwen3_vl_fullvlm_v2_simpngpath_wristunlimited_breakfastprompt_lfp_fullft_freezevit_bs4_len8192_20260412_011556}
VLA_CONFIG=${VLA_CONFIG:-pi05_libero_robomemarena_fullvlm_v2_noflip_dataset}
VLA_POLICY=${VLA_POLICY:-${ASSET_OPENPI_ROOT}/checkpoints/pi05_libero_robomemarena_fullvlm_v2_noflip_dataset/fullvlm_v2_robomemarena_noflip_v2_bs128_4gpu_20260507_183338/35999}
BDDL=${BDDL:-${ASSET_OFFICIAL_RUNTIME_ROOT}/bddl/1_cookies_tomato_basket.bddl}
ENDPOSE_HOLD_TARGETS_JSON=${ENDPOSE_HOLD_TARGETS_JSON:-${INFER_ROOT}/task1_eval/task1_subtask_end_poses_successindex_seed100_199.json}
ENDPOSE_HOLD_POS_TOL=${ENDPOSE_HOLD_POS_TOL:-0.04}
ENDPOSE_HOLD_MIN_ACTIVE_STEPS=${ENDPOSE_HOLD_MIN_ACTIVE_STEPS:-20}
ENDPOSE_HOLD_CONSECUTIVE=${ENDPOSE_HOLD_CONSECUTIVE:-2}
POST_HOLD_RELEASE_VLA_STEPS=${POST_HOLD_RELEASE_VLA_STEPS:-0}
PREVENT_SUBTASK_REGRESSION=${PREVENT_SUBTASK_REGRESSION:-0}
REGRESSION_GUARD_AFTER_HOLD_RELEASE=${REGRESSION_GUARD_AFTER_HOLD_RELEASE:-0}
TASK1_ACCEPT_RAW_VLM_OUTPUT=${TASK1_ACCEPT_RAW_VLM_OUTPUT:-0}
TASK1_DISABLE_OUTPUT_NORMALIZE=${TASK1_DISABLE_OUTPUT_NORMALIZE:-0}

ASYNC_VLM=${ASYNC_VLM:-0}
VLM_INPUT_PROFILE=${VLM_INPUT_PROFILE:-fullvlm_256}
VLM_PROMPT_PROFILE=${VLM_PROMPT_PROFILE:-task1_kf5}
VLM_MATCH_VLA_PREPROCESS=${VLM_MATCH_VLA_PREPROCESS:-1}
VLM_MATCH_TRAINING_JPEG_ROUNDTRIP=${VLM_MATCH_TRAINING_JPEG_ROUNDTRIP:-0}
VLM_TRAINING_JPEG_QUALITY=${VLM_TRAINING_JPEG_QUALITY:-30}
VLA_RESIZE_SIZE=${VLA_RESIZE_SIZE:-256}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-1}
MAX_STEPS=${MAX_STEPS:-2000}
REPLAN_STEPS=${REPLAN_STEPS:-5}
K_MAX=${K_MAX:-0}
SEED=${SEED:-104}

if [[ "${ASYNC_VLM}" != "0" ]]; then
  echo "[ERROR] This launcher is sync-only. Set ASYNC_VLM=0 or leave it unset." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"

export OPENPI_ROOT
export TARGET_LIBERO_PATH=${TARGET_LIBERO_PATH:-${ASSET_OFFICIAL_RUNTIME_ROOT}/LIBERO/libero}
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TASK1_ACCEPT_RAW_VLM_OUTPUT TASK1_DISABLE_OUTPUT_NORMALIZE

VLA_LOG=${RUN_ROOT}/vla_server.log
EVAL_LOG=${RUN_ROOT}/eval.launch.log
DRIVER_LOG=${RUN_ROOT}/driver.log

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ -n "${GPU_COUNT}" ]] && [[ "${GPU_COUNT}" -gt 0 ]]; then
    for ((i=0; i<GPU_COUNT; i++)); do
      GPU_IDS+=("${i}")
    done
  fi
fi

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] No visible GPUs before launching VLA/VLM." | tee -a "${DRIVER_LOG}"
  exit 1
fi

DEFAULT_VLA_GPU="${GPU_IDS[0]}"
if [[ "${#GPU_IDS[@]}" -ge 2 ]]; then
  DEFAULT_VLM_GPU="${GPU_IDS[1]}"
else
  DEFAULT_VLM_GPU="${GPU_IDS[0]}"
fi
VLA_CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES:-${DEFAULT_VLA_GPU}}"
VLM_CUDA_VISIBLE_DEVICES="${VLM_CUDA_VISIBLE_DEVICES:-${DEFAULT_VLM_GPU}}"

cd "${OPENPI_ROOT}"

{
  echo "[INFO] RUN_ID=${RUN_ID}"
  echo "[INFO] RUN_ROOT=${RUN_ROOT}"
  echo "[INFO] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
  echo "[INFO] LORA_PATH=${LORA_PATH}"
  echo "[INFO] VLA_CONFIG=${VLA_CONFIG}"
  echo "[INFO] VLA_POLICY=${VLA_POLICY}"
  echo "[INFO] ASYNC_VLM=${ASYNC_VLM} REPLAN_STEPS=${REPLAN_STEPS} SEED=${SEED}"
  echo "[INFO] ENDPOSE_HOLD_TARGETS_JSON=${ENDPOSE_HOLD_TARGETS_JSON}"
  echo "[INFO] ENDPOSE_HOLD_POS_TOL=${ENDPOSE_HOLD_POS_TOL}"
  echo "[INFO] POST_HOLD_RELEASE_VLA_STEPS=${POST_HOLD_RELEASE_VLA_STEPS}"
  echo "[INFO] PREVENT_SUBTASK_REGRESSION=${PREVENT_SUBTASK_REGRESSION}"
  echo "[INFO] REGRESSION_GUARD_AFTER_HOLD_RELEASE=${REGRESSION_GUARD_AFTER_HOLD_RELEASE}"
  echo "[INFO] TASK1_ACCEPT_RAW_VLM_OUTPUT=${TASK1_ACCEPT_RAW_VLM_OUTPUT}"
  echo "[INFO] TASK1_DISABLE_OUTPUT_NORMALIZE=${TASK1_DISABLE_OUTPUT_NORMALIZE}"
  echo "[INFO] CUDA_VISIBLE_DEVICES(raw)=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[INFO] GPU binding: VLA_CUDA_VISIBLE_DEVICES=${VLA_CUDA_VISIBLE_DEVICES} VLM_CUDA_VISIBLE_DEVICES=${VLM_CUDA_VISIBLE_DEVICES}"
} | tee -a "${DRIVER_LOG}"

CUDA_VISIBLE_DEVICES="${VLA_CUDA_VISIBLE_DEVICES}" "${OPENPI_PYTHON}" -u scripts/serve_policy.py --port "${PORT}" \
  policy:checkpoint \
  --policy.config="${VLA_CONFIG}" \
  --policy.dir="${VLA_POLICY}" \
  > "${VLA_LOG}" 2>&1 &
VLA_PID=$!

cleanup() {
  kill "${VLA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if ! kill -0 "${VLA_PID}" 2>/dev/null; then
    echo "[ERROR] VLA server exited early. See ${VLA_LOG}" | tee -a "${DRIVER_LOG}"
    tail -n 80 "${VLA_LOG}" || true
    exit 1
  fi
  if python3 - <<PY
import socket
s = socket.socket()
try:
    s.settimeout(1.0)
    s.connect(("127.0.0.1", ${PORT}))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY
  then
    echo "[INFO] VLA server ready." | tee -a "${DRIVER_LOG}"
    break
  fi
  sleep 2
done

python3 - <<PY
import socket
s = socket.socket()
try:
    s.settimeout(1.0)
    s.connect(("127.0.0.1", ${PORT}))
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
finally:
    s.close()
PY

echo "[INFO] Starting sync endpose-hold eval." | tee -a "${DRIVER_LOG}"
cd "${INFER_ROOT}"
EXTRA_ARGS=()
if [[ "${VLM_MATCH_VLA_PREPROCESS}" == "0" ]]; then
  EXTRA_ARGS+=(--no-vlm-match-vla-preprocess)
elif [[ "${VLM_MATCH_VLA_PREPROCESS}" != "1" ]]; then
  echo "[ERROR] VLM_MATCH_VLA_PREPROCESS must be 0 or 1, got ${VLM_MATCH_VLA_PREPROCESS}" | tee -a "${DRIVER_LOG}"
  exit 1
fi
if [[ "${VLM_MATCH_TRAINING_JPEG_ROUNDTRIP}" == "1" ]]; then
  EXTRA_ARGS+=(--vlm-match-training-jpeg-roundtrip)
  EXTRA_ARGS+=(--vlm-training-jpeg-quality "${VLM_TRAINING_JPEG_QUALITY}")
elif [[ "${VLM_MATCH_TRAINING_JPEG_ROUNDTRIP}" != "0" ]]; then
  echo "[ERROR] VLM_MATCH_TRAINING_JPEG_ROUNDTRIP must be 0 or 1, got ${VLM_MATCH_TRAINING_JPEG_ROUNDTRIP}" | tee -a "${DRIVER_LOG}"
  exit 1
fi
if [[ "${PREVENT_SUBTASK_REGRESSION}" == "1" ]]; then
  EXTRA_ARGS+=(--prevent-subtask-regression)
elif [[ "${PREVENT_SUBTASK_REGRESSION}" != "0" ]]; then
  echo "[ERROR] PREVENT_SUBTASK_REGRESSION must be 0 or 1, got ${PREVENT_SUBTASK_REGRESSION}" | tee -a "${DRIVER_LOG}"
  exit 1
fi
if [[ "${REGRESSION_GUARD_AFTER_HOLD_RELEASE}" == "1" ]]; then
  EXTRA_ARGS+=(--prevent-subtask-regression-after-hold-release)
elif [[ "${REGRESSION_GUARD_AFTER_HOLD_RELEASE}" != "0" ]]; then
  echo "[ERROR] REGRESSION_GUARD_AFTER_HOLD_RELEASE must be 0 or 1, got ${REGRESSION_GUARD_AFTER_HOLD_RELEASE}" | tee -a "${DRIVER_LOG}"
  exit 1
fi

CUDA_VISIBLE_DEVICES="${VLM_CUDA_VISIBLE_DEVICES}" "${INFER_PYTHON}" -u "${EVAL_PY}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --bddl-file "${BDDL}" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --lora-path "${LORA_PATH}" \
  --vlm-model-type qwen3_vl \
  --vlm-device cuda:0 \
  --vlm-input-profile "${VLM_INPUT_PROFILE}" \
  --vlm-prompt-profile "${VLM_PROMPT_PROFILE}" \
  --resize-size "${VLA_RESIZE_SIZE}" \
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --max-steps "${MAX_STEPS}" \
  --replan-steps "${REPLAN_STEPS}" \
  --k-max "${K_MAX}" \
  --seed "${SEED}" \
  --log-base "${LOG_BASE}" \
  --run-id "${RUN_ID}" \
  --endpose-hold-targets-json "${ENDPOSE_HOLD_TARGETS_JSON}" \
  --endpose-hold-pos-tol "${ENDPOSE_HOLD_POS_TOL}" \
  --endpose-hold-min-active-steps "${ENDPOSE_HOLD_MIN_ACTIVE_STEPS}" \
  --endpose-hold-consecutive "${ENDPOSE_HOLD_CONSECUTIVE}" \
  --post-hold-release-vla-steps "${POST_HOLD_RELEASE_VLA_STEPS}" \
  "${EXTRA_ARGS[@]}" \
  > "${EVAL_LOG}" 2>&1

status=$?
echo "[INFO] Eval exit status=${status}" | tee -a "${DRIVER_LOG}"
echo "[INFO] Logs: ${EVAL_LOG} ${VLA_LOG}" | tee -a "${DRIVER_LOG}"
exit "${status}"
