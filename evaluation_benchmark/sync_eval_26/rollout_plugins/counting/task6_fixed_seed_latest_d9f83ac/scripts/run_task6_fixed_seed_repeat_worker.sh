#!/usr/bin/env bash
set -euo pipefail
umask 0002

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${RUNNER:-${PACK_DIR}/scripts/run_autonomous_task.sh}"

RUN_GROUP="${RUN_GROUP:?Set RUN_GROUP.}"
WORKER_ID="${WORKER_ID:?Set WORKER_ID.}"
REPEAT_START="${REPEAT_START:?Set REPEAT_START.}"
REPEAT_COUNT="${REPEAT_COUNT:?Set REPEAT_COUNT.}"
FIXED_SEED="${FIXED_SEED:-100}"
VLM_CKPT="${VLM_CKPT:?Set VLM_CKPT.}"
RUNS_BASE="${RUNS_BASE:-${PACK_DIR}/runs_autonomous/${RUN_GROUP}/workers/worker${WORKER_ID}}"
LOG_DIR="${LOG_DIR:-${PACK_DIR}/logs_autonomous/${RUN_GROUP}/worker${WORKER_ID}}"
BASE_PORT="${BASE_PORT:-29800}"
RUNTIME_HOME="${RUNTIME_HOME:-${HOME}}"

[[ "${REPEAT_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "[ERROR] invalid REPEAT_COUNT=${REPEAT_COUNT}" >&2; exit 2; }
[[ "${REPEAT_START}" =~ ^[0-9]+$ ]] || { echo "[ERROR] invalid REPEAT_START=${REPEAT_START}" >&2; exit 2; }
mkdir -p "${RUNS_BASE}" "${LOG_DIR}"

RESULTS="${RUNS_BASE}/repeat_results.tsv"
printf 'worker_id\trepeat_index\tseed\texit_code\tstage_score_pct\tstage_success_rate\tgoal_success_rate\trun_dir\n' > "${RESULTS}"

for ((offset = 0; offset < REPEAT_COUNT; offset += 1)); do
  repeat_index=$((REPEAT_START + offset))
  port=$((BASE_PORT + WORKER_ID * 100 + offset))
  run_id="${RUN_GROUP}_w${WORKER_ID}_r$(printf '%02d' "${repeat_index}")"
  out_root="${RUNS_BASE}/${run_id}"
  launch_log="${LOG_DIR}/${run_id}.log"

  set +e
  TASK_ID=6 \
  VLM_CKPT="${VLM_CKPT}" \
  EVALUATOR_FILE_OVERRIDE="${PACK_DIR}/evaluators/eval_counting_autonomous_guarded_d9f83ac.py" \
  NUM_TRIALS=1 \
  SEED="${FIXED_SEED}" \
  REPLAN_STEPS=1 \
  POST_STAGE_STEPS=30 \
  VLM_INTERVAL=25 \
  HOLD_AFTER_REQUIRED_STAGES=1 \
  PORT="${port}" \
  RUN_ID="${run_id}" \
  OUT_ROOT="${out_root}" \
  RUNTIME_HOME="${RUNTIME_HOME}" \
  bash "${RUNNER}" > "${launch_log}" 2>&1
  rc=$?
  set -e

  # Parse evaluator TSV with csv rather than Bash IFS: its empty `error` field
  # would otherwise shift all metrics left because tab is whitespace to read.
  python - "${RESULTS}" "${WORKER_ID}" "${repeat_index}" "${FIXED_SEED}" "${rc}" "${out_root}" <<'PY'
import csv
import pathlib
import sys

results_path, worker_id, repeat_index, seed, exit_code, out_root = sys.argv[1:]
summary_path = pathlib.Path(out_root) / "summary.tsv"
metrics = {"stage_score_pct": "", "stage_success_rate": "", "goal_success_rate": ""}
if summary_path.is_file():
    with summary_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows:
        metrics = {key: rows[-1].get(key, "") for key in metrics}

with open(results_path, "a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=[
        "worker_id", "repeat_index", "seed", "exit_code", "stage_score_pct",
        "stage_success_rate", "goal_success_rate", "run_dir",
    ], delimiter="\t")
    writer.writerow({
        "worker_id": worker_id,
        "repeat_index": repeat_index,
        "seed": seed,
        "exit_code": exit_code,
        **metrics,
        "run_dir": out_root,
    })
PY
done

python - "${RESULTS}" "${RUNS_BASE}/worker_summary.json" <<'PY'
import csv
import json
import sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
successes = sum(row["stage_success_rate"] == "1.0000" for row in rows)
json.dump(
    {"episodes": len(rows), "stage_successes": successes, "stage_success_rate": successes / max(1, len(rows)), "rows": rows},
    open(sys.argv[2], "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
PY
