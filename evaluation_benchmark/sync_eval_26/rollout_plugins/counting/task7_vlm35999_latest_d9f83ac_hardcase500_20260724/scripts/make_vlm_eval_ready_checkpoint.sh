#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 SOURCE_CHECKPOINT PROCESSOR_SOURCE OUTPUT_DIR" >&2
  exit 2
fi

SOURCE_CHECKPOINT="$1"
PROCESSOR_SOURCE="$2"
OUTPUT_DIR="$3"

for path in "${SOURCE_CHECKPOINT}/model.safetensors" "${PROCESSOR_SOURCE}/processor_config.json"; do
  [[ -f "${path}" ]] || { echo "[ERROR] required file missing: ${path}" >&2; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"
for path in "${SOURCE_CHECKPOINT}"/*; do
  [[ -e "${path}" ]] || continue
  ln -sfn "${path}" "${OUTPUT_DIR}/$(basename "${path}")"
done
ln -sfn "${PROCESSOR_SOURCE}/processor_config.json" "${OUTPUT_DIR}/processor_config.json"

for path in "${OUTPUT_DIR}/model.safetensors" "${OUTPUT_DIR}/config.json" "${OUTPUT_DIR}/processor_config.json"; do
  [[ -e "${path}" ]] || { echo "[ERROR] eval-ready output is incomplete: ${path}" >&2; exit 1; }
done

echo "${OUTPUT_DIR}"
