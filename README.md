# RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark

RoboMemArena is a comprehensive and challenging robotic memory benchmark with 26 manipulation tasks, demonstration data, and evaluation BDDL.

## Links

[![arXiv](https://img.shields.io/badge/arXiv-2605.10921-b31b1b?style=for-the-badge)](https://arxiv.org/html/2605.10921v1)
[![Project Page](https://img.shields.io/badge/Project-Page-76b900?style=for-the-badge)](https://robomemarena.github.io/)
[![Leaderboard Results](https://img.shields.io/badge/Leaderboard-Results-007ec6?style=for-the-badge)](https://robomemarena.github.io/leaderboard.html)
[![Dataset Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-f3b900?style=for-the-badge)](https://huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena)

## Dataset Structure

The dataset is hosted on Hugging Face:

[![Dataset Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-f3b900?style=for-the-badge)](https://huggingface.co/datasets/RoboMemArenaBenchmark/RoboMemArena)

The released dataset is organized as:

- 4 high-level category folders
- each category folder contains task subfolders (for example, `1_cookies_tomato_basket_dataset`)
- each task subfolder keeps the same internal structure

```
<dataset_root>/
├── <category_1>/
│   └── 1_cookies_tomato_basket_dataset/
│       └── subtask_data/         # Keyframe-annotated HDF5 episodes (used for training)
│           ├── pick_cookies_0_seed100_task1.hdf5
│           ├── pick_cookies_0_seed101_task1.hdf5
│           └── ...
├── <category_2>/
├── <category_3>/
└── <category_4>/
```

Filename convention for subtask HDF5 files:
`<primitive>_<subtask_order>_seed<seed>_task<task_id>.hdf5`, where
`subtask_order` is the 0-based order index used in task decomposition.

The same task seed links multiple subtask files into one complete long-horizon
trajectory. For example, files with `seed100` under the same task folder should
be sorted by `subtask_order` and concatenated to reconstruct the full task
episode. If your training target is the complete task rather than individual
subtasks, group files by `task_id` and `seed`, then concatenate the ordered
subtask episodes before training.

### Key Directories

| Directory | Description |
|-----------|-------------|
| **subtask_data/** | Sub-episodes with **keyframe annotations**; each HDF5 contains `data/demo_*` with `actions`, `obs/agentview_rgb`, `obs/eye_in_hand_rgb`, `obs/ee_states`, `obs/gripper_states`, `obs/joint_states` |

### HDF5 Format (subtask_data)

- `data/demo_{id}/actions` — (T, 7) end-effector actions
- `data/demo_{id}/obs/agentview_rgb` — (T, 256, 256, 3) top-down view
- `data/demo_{id}/obs/eye_in_hand_rgb` — (T, 256, 256, 3) wrist camera
- `data/demo_{id}/obs/ee_states`, `gripper_states`, `joint_states` — robot state

## RLDS Conversion

Use `RoboMemArena_dataset_builder.py` to convert HDF5 to RLDS (TFDS) format:

RLDS conversion is a data-only step; a CPU Python environment is sufficient.

- Python 3.9
- `tensorflow==2.13.0`
- `tensorflow-datasets==4.9.2`
- `h5py==3.9.0`
- `numpy==1.24.3`

Example environment setup:

```bash
conda create -n robomemarena-rlds python=3.9 -y
conda activate robomemarena-rlds
pip install tensorflow==2.13.0 tensorflow-datasets==4.9.2 h5py==3.9.0 numpy==1.24.3
```

Set the source dataset root and run the builder from this repository root:

```bash
export ROBOMEMARENA_DATA_ROOT=/path/to/<dataset_root>
python -c "
import RoboMemArena_dataset_builder as b
import tensorflow_datasets as tfds
ds_builder = b.RoboMemArenaDataset(data_dir='/path/to/output')
ds_builder.download_and_prepare()
"
```

## Evaluation BDDL

Evaluation tasks are defined in the `bddl/` folder:

- `1_*.bddl` … `26_*.bddl` — 26 BDDL task definitions using contiguous benchmark task IDs

These BDDL files can be used with the provided LIBERO-compatible evaluation environment.
The full 26-task benchmark descriptions are available here:

- [Benchmark Task Details](https://robomemarena.github.io/#task-details)
- [Evaluation Benchmark Overview](evaluation_benchmark/README.md)
- [26-Task Reference Evaluation](evaluation_benchmark/reference_evaluation/README.md)
- [Task Evaluation Code Guide](evaluation_benchmark/docs/task_evaluation_code_guide.md)

## OpenPI Runtime Interface

Evaluation scripts support two OpenPI runtime paths:

- Bundled minimal runtime (default): `third_party/openpi_minimal`
- External OpenPI source (optional): set `OPENPI_ROOT=/abs/path/to/openpi`

If using external OpenPI source, ensure the following exist:

- `scripts/serve_policy.py`
- `packages/openpi/src`
- `packages/openpi-client/src`

## LIBERO Environment Setup

Evaluation requires a LIBERO-compatible environment. This repo includes a local fork under `evaluation_benchmark/libero_fork/`:

```bash
# Clone RoboMemArena (includes submodules)
git clone --recurse-submodules https://github.com/OpenHelix-Team/RoboMemArena.git
cd RoboMemArena

# If you already cloned without --recurse-submodules:
# git submodule update --init --recursive

# Make the local LIBERO fork importable
export PYTHONPATH="${PWD}/evaluation_benchmark/libero_fork:${PYTHONPATH}"
```

Copy the `bddl/` folder from this repo to your evaluation config path, or set `LIBERO_BDDL_PATH` to point to `bddl/` for RoboMemArena task definitions.

## PrediMem S2 Training Add-on

This repo also includes a minimal add-on folder:

- [predictive_coding_head/](predictive_coding_head/)

It provides reusable integration logic for PrediMem S2 training with a **Predictive Coding Head** in a Qwen3-VL style training pipeline.
See:

- [predictive_coding_head/README.md](predictive_coding_head/README.md)

For the S1 low-level policy, both the environment setup and the training logic
can directly follow the official OpenPI repository:
https://github.com/Physical-Intelligence/openpi

For VLM training data construction, users may follow the official Qwen3-VL
multimodal `messages` / `content` format described here:
https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#using-transformers-to-chat

## Contact

For questions, please contact me via WeChat: `leshuaigeye` or email: `leihuashuohit@gmail.com`.

## Citation

If you find RoboMemArena useful in your research, please cite:

```bibtex
@article{robomemarena2025,
  title   = {RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark},
  author  = {Huashuo Lei and Wenxuan Song and Huarui Zhang and Jieyuan Pei and Jiayi Chen and Haodong Yan and Han Zhao and Pengxiang Ding and Zhipeng Zhang and Lida Huang and Donglin Wang and Yan Wang and Haoang Li},
  journal = {arXiv preprint arXiv:2605.10921},
  year    = {2026}
}
```
