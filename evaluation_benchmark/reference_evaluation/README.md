# 26-Task Reference Evaluation

This folder provides reference VLM/VLA evaluation code for RoboMemArena.

We provide a quick start at `task1_nomap_reference/`. If you want to train/evaluate all tasks, please use `tasks2_26_vlm5_reference/` for VLM5 Tasks 2-26.

The required Python/runtime environment is the official OpenPI environment. Please install and activate OpenPI first, then set the local paths and checkpoints described below.

The reference evaluation folder is `evaluation_benchmark/reference_evaluation/`.
Async reference code is kept separately in `evaluation_benchmark/async_vlm26_reference/`.
For async evaluation logic reference, see [Async 26-Task Reference Evaluation](../async_vlm26_reference/README.md).
The model-agnostic benchmark interface remains in `evaluation_benchmark/scripts/`.

## Weight Interface (HF-ready)

Use interface variables to connect weights (HF or local) instead of hardcoding URLs:

```bash
# unified weight source
export WEIGHT_SOURCE=hf            # hf or local

# HF repo entry (example)
export HF_REPO_ID=<your_hf_repo_id>
export VLM_HF_SUBDIR=vlm_task1
export VLA_HF_SUBDIR=vla_alltask/params

# local resolved paths used by scripts
export VLM_CKPT=/abs/path/to/vlm_task1
export VLA_CKPT=/abs/path/to/vla_alltask/params
```

For local-only usage:

```bash
export WEIGHT_SOURCE=local
export VLM_CKPT=/abs/path/to/vlm_task1
export VLA_CKPT=/abs/path/to/vla_alltask/params
```

## Weight-to-Code Mapping

| Code path | Weights |
|---|---|
| `task1_nomap_reference/eval_task1_nomap_reference.py` | `VLM_HF_SUBDIR=vlm_task1` + `VLA_HF_SUBDIR=vla_alltask/params` |
| `tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py` | `VLM_HF_SUBDIR=vlm_task1` + `VLA_HF_SUBDIR=vla_alltask/params` |

## Files

```text
reference_evaluation/
  README.md
  task1_nomap_reference/
    eval_task1_nomap_reference.py
  tasks2_26_vlm5_reference/
    eval_tasks2_26_vlm_vla.py
    run_tasks2_26_vlm_vla_csr_tsr.sh
    fullvlm_v2_26_memory_tasks.json
```

## Task 1 Evaluation Code

`task1_nomap_reference/eval_task1_nomap_reference.py` contains the Task 1 evaluation code.

OpenPI source interface:

- Default: use the bundled minimal runtime at `third_party/openpi_minimal`.
- Optional: use your own OpenPI source tree by setting `OPENPI_ROOT`.
- Required entries under `OPENPI_ROOT`:
  - `scripts/serve_policy.py`
  - `packages/openpi/src`
  - `packages/openpi-client/src`

Required local environment variables:

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional; defaults to repo-bundled third_party/openpi_minimal
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm_task1
```

Example:

```bash
cd evaluation_benchmark/reference_evaluation/task1_nomap_reference
python eval_task1_nomap_reference.py \
  --bddl-file ../../bddl/1_cookies_tomato_basket.bddl \
  --base-model-dir "$VLM_CKPT" \
  --host 127.0.0.1 \
  --port 8026 \
  --resize-size 256 \
  --num-trials-per-task 1 \
  --max-steps 2000
```

The VLA policy server should be launched separately with the all-task VLA
checkpoint params.

## Tasks 2-26 Evaluation Code

The VLM5 Tasks 2-26 evaluation code currently lives in `tasks2_26_vlm5_reference/`. Task 1 is intentionally separated and should use the Task 1 evaluation code above.

OpenPI source interface:

- Default: use the bundled minimal runtime at `third_party/openpi_minimal`.
- Optional: point `OPENPI_ROOT` to your own OpenPI checkout. No code changes are needed.

Required local inputs:

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional; defaults to repo-bundled third_party/openpi_minimal
export OPENPI_INFERENCE_ROOT=/abs/path/to/openpi_inference
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm_task1
export VLA_CKPT=/abs/path/to/vla_alltask/params
# optional override; default in runner is pi05_robomemarena
export VLA_CONFIG=<your_vla_config_name>
```

Run the default 25-task set:

```bash
cd evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference
bash run_tasks2_26_vlm_vla_csr_tsr.sh
```

Run a subset:

```bash
cd evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference
export TASKS_JSON='[2,3,4]'
bash run_tasks2_26_vlm_vla_csr_tsr.sh
```

## Outputs

The runner writes outputs under `OUT_ROOT`:

```text
${OUT_ROOT}/summary.tsv
${OUT_ROOT}/summary.json
${OUT_ROOT}/aggregate.json
${OUT_ROOT}/prompt_trace.tsv
${OUT_ROOT}/videos/task*/...
${OUT_ROOT}/task*/ep*/sync_vlm_trace.jsonl
```

Metric names:

- `CSR`: average stage/process completion percentage
- `TSR`: final BDDL goal success rate

Example HF weight repository: `https://huggingface.co/huashuolei/PrediMem`
