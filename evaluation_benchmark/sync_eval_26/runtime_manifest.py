from __future__ import annotations

import hashlib
import json
import os
import pwd
import socket
import subprocess
from pathlib import Path
from typing import Any

from .models import TaskProfile


RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_GPUS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_LOCALID",
    "SLURM_PROCID",
    "SLURM_STEP_GPUS",
    "SLURM_STEP_ID",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def git_state(repo_root: Path) -> dict[str, Any]:
    return {
        "head": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "branch", "--show-current"),
        "status_porcelain": _git_output(repo_root, "status", "--porcelain"),
    }


def execution_identity() -> dict[str, Any]:
    return {
        "unix_user": pwd.getpwuid(os.getuid()).pw_name,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "slurm": {
            key: os.environ[key] for key in RUNTIME_ENV_KEYS if key in os.environ
        },
    }


def resolve_runtime_topology(
    profile: TaskProfile,
    raw_devices: str | None = None,
) -> dict[str, Any]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "") if raw_devices is None else raw_devices
    devices = tuple(item.strip() for item in raw.split(",") if item.strip())
    topology = profile.runtime_topology
    if len(devices) != topology.allocation_gpus:
        raise RuntimeError(
            "topology_gpu_count_mismatch: "
            f"task={profile.task_id} required={topology.allocation_gpus} "
            f"visible={list(devices)}"
        )
    return {
        "required_gpus": topology.allocation_gpus,
        "visible_devices": list(devices),
        "vla_device": devices[topology.vla_visible_index],
        "vlm_device": devices[topology.vlm_visible_index],
    }


def explicit_runtime_env(
    profile: TaskProfile,
    *,
    episode_index: int,
    seed: int,
    output_dir: Path,
    scorer_lock_path: Path,
) -> dict[str, str]:
    values = dict(profile.runtime_env)
    values.update(
        {
            "TASK_ID": str(profile.task_id),
            "EPISODE_INDEX": str(episode_index),
            "SEED": str(seed),
            "OUTPUT_DIR": str(output_dir),
            "VLA_CHECKPOINT": str(profile.vla_checkpoint),
            "VLM_CHECKPOINT": str(profile.vlm_checkpoint),
            "NORM_STATS_PATH": str(profile.norm_path),
            "NORM_STATS_SHA256": profile.norm_sha256,
            "BDDL_PATH": str(profile.bddl_path),
            "OFFICIAL_SCORER_LOCK": str(scorer_lock_path),
            "REPLAN_STEPS": str(profile.replan_steps),
            "MAX_STEPS": str(profile.max_steps),
        }
    )
    return dict(sorted(values.items()))


def profile_manifest(
    profile: TaskProfile,
    *,
    profile_path: Path,
) -> dict[str, Any]:
    return {
        "task_id": profile.task_id,
        "task_name": profile.task_name,
        "status": profile.status,
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": file_sha256(profile_path),
        "plugin_kind": profile.plugin_kind,
        "plugin_entrypoint": str(profile.plugin_entrypoint),
        "bddl_path": str(profile.bddl_path),
        "vla_checkpoint": str(profile.vla_checkpoint),
        "vlm_checkpoint": str(profile.vlm_checkpoint),
        "norm_path": str(profile.norm_path),
        "norm_sha256": profile.norm_sha256,
        "seed_mode": profile.seed_mode,
        "base_seed": profile.seed,
        "replan_steps": profile.replan_steps,
        "max_steps": profile.max_steps,
        "runtime_topology_contract": {
            "allocation_gpus": profile.runtime_topology.allocation_gpus,
            "vla_visible_index": profile.runtime_topology.vla_visible_index,
            "vlm_visible_index": profile.runtime_topology.vlm_visible_index,
        },
        "prompt_config": profile.prompt_config,
        "source_paths": [str(path) for path in profile.source_paths],
        "source_hashes": dict(sorted(profile.source_hashes.items())),
        "hf_assets": profile.hf_assets,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
