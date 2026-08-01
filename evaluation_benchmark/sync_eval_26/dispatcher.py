from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .aggregate import EpisodeRecord, load_official_episode
from .assets import load_external_assets
from .frozen_subprocess import FrozenSubprocessPlugin
from .models import TaskProfile
from .native_plugin import NativePlugin
from .plugin_api import RolloutRequest, RolloutResult
from .profile_loader import load_profile, verify_profile_assets
from .runtime_manifest import (
    execution_identity,
    explicit_runtime_env,
    git_state,
    profile_manifest,
    resolve_runtime_topology,
    write_json_atomic,
)
from .scoring_bridge import OfficialScoringBridge
from .upstream_lock import LOCK_RELATIVE_PATH, verify_upstream_lock


@dataclass(frozen=True)
class DispatchResult:
    rollout: RolloutResult
    official: EpisodeRecord
    manifest_path: Path


def seed_for_episode(profile: TaskProfile, episode_index: int) -> int:
    if profile.seed_mode == "fixed":
        return profile.seed
    return profile.seed + episode_index


def _plugin_for(profile: TaskProfile) -> FrozenSubprocessPlugin:
    if profile.plugin_kind == "native":
        return NativePlugin()
    if profile.plugin_kind == "frozen-subprocess":
        return FrozenSubprocessPlugin()
    raise ValueError(f"unsupported plugin kind: {profile.plugin_kind}")


def dispatch_episode(
    *,
    repo_root: Path,
    profile_path: Path,
    episode_index: int,
    output_root: Path,
) -> DispatchResult:
    repo_root = repo_root.resolve()
    assets = load_external_assets()
    lock = verify_upstream_lock(repo_root, runtime_root=assets.official_runtime_root)
    profile = load_profile(profile_path, assets=assets)
    verify_profile_assets(profile)
    runtime_topology = resolve_runtime_topology(profile)
    seed = seed_for_episode(profile, episode_index)
    episode_dir = output_root / f"task{profile.task_id:02d}" / f"ep{episode_index:03d}"
    request = RolloutRequest(
        repo_root=repo_root,
        task_id=profile.task_id,
        episode_index=episode_index,
        seed=seed,
        output_dir=episode_dir,
        profile=profile,
        scorer_lock_path=repo_root / LOCK_RELATIVE_PATH,
        runtime_env={
            "VLA_CUDA_VISIBLE_DEVICES": str(runtime_topology["vla_device"]),
            "VLM_CUDA_VISIBLE_DEVICES": str(runtime_topology["vlm_device"]),
            "VLA_LOCAL_DEVICE": str(runtime_topology["vla_device"]),
            "VLM_LOCAL_DEVICE": str(runtime_topology["vlm_device"]),
        },
    )
    rollout = _plugin_for(profile).run(request)
    if rollout.return_code != 0:
        raise RuntimeError(
            f"rollout failed task={profile.task_id} ep={episode_index}: "
            f"{rollout.termination_reason}; log={rollout.episode_log}"
        )
    bridge = OfficialScoringBridge(repo_root, assets=assets)
    official = load_official_episode(
        rollout.official_result_path, bridge, expected_commit=lock.commit
    )
    if official.task_id != profile.task_id or official.episode_index != episode_index:
        raise ValueError(
            f"official result identity mismatch: profile_task={profile.task_id} "
            f"episode={episode_index} result={official}"
        )

    manifest_path = episode_dir / "run_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            **profile_manifest(profile, profile_path=profile_path),
            "episode_index": episode_index,
            "seed": seed,
            "scorer_commit": lock.commit,
            "stage_done": official.stage_done,
            "csr_pct": official.stage_score_pct,
            "tsr": int(official.stage_success),
            "execution": execution_identity(),
            "runtime_topology": runtime_topology,
            "git": git_state(repo_root),
            "explicit_runtime_env": explicit_runtime_env(
                profile,
                episode_index=episode_index,
                seed=seed,
                output_dir=episode_dir.resolve(),
                scorer_lock_path=request.scorer_lock_path.resolve(),
            ),
            "rollout": {
                "return_code": rollout.return_code,
                "episode_log": str(rollout.episode_log),
                "official_result_path": str(rollout.official_result_path),
                "video_paths": [str(path) for path in rollout.video_paths],
                "prompt_trace": (
                    str(rollout.prompt_trace) if rollout.prompt_trace else None
                ),
                "termination_reason": rollout.termination_reason,
            },
        },
    )
    return DispatchResult(
        rollout=rollout, official=official, manifest_path=manifest_path
    )


def dispatch_range(
    *,
    repo_root: Path,
    profile_path: Path,
    episode_indices: Iterable[int],
    output_root: Path,
) -> list[DispatchResult]:
    return [
        dispatch_episode(
            repo_root=repo_root,
            profile_path=profile_path,
            episode_index=episode_index,
            output_root=output_root,
        )
        for episode_index in episode_indices
    ]
