from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .aggregate import EpisodeRecord, load_official_episode
from .assets import load_external_assets
from .profile_loader import file_sha256, load_profile, verify_profile_assets
from .scoring_bridge import OfficialScoringBridge
from .upstream_lock import verify_upstream_lock


@dataclass(frozen=True)
class EpisodeValidation:
    valid: bool
    reason: str
    record: EpisodeRecord | None = None
    manifest: dict | None = None


def _matches_runtime_topology(profile, payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    try:
        required_gpus = int(payload["required_gpus"])
        visible_devices = [str(item) for item in payload["visible_devices"]]
        vla_device = str(payload["vla_device"])
        vlm_device = str(payload["vlm_device"])
    except (KeyError, TypeError, ValueError):
        return False
    topology = profile.runtime_topology
    if required_gpus != topology.allocation_gpus:
        return False
    if len(visible_devices) != topology.allocation_gpus:
        return False
    if not all(visible_devices) or len(set(visible_devices)) != len(visible_devices):
        return False
    return (
        vla_device == visible_devices[topology.vla_visible_index]
        and vlm_device == visible_devices[topology.vlm_visible_index]
    )


def validate_episode(
    *,
    repo_root: Path,
    profile_path: Path,
    episode_dir: Path,
    require_video: bool = True,
) -> EpisodeValidation:
    try:
        assets = load_external_assets()
        lock = verify_upstream_lock(repo_root, runtime_root=assets.official_runtime_root)
        profile = load_profile(profile_path, assets=assets)
        try:
            verify_profile_assets(profile)
        except (FileNotFoundError, ValueError) as exc:
            return EpisodeValidation(
                False,
                f"profile_assets_invalid:{type(exc).__name__}",
            )
        episode_dir = episode_dir.resolve()
        episode_index = int(episode_dir.name.removeprefix("ep"))
        expected_seed = (
            profile.seed
            if profile.seed_mode == "fixed"
            else profile.seed + episode_index
        )
        manifest_path = episode_dir / "run_manifest.json"
        if not manifest_path.is_file():
            return EpisodeValidation(False, "missing_run_manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "task_id": profile.task_id,
            "episode_index": episode_index,
            "seed": expected_seed,
            "profile_sha256": file_sha256(profile_path),
            "vla_checkpoint": str(profile.vla_checkpoint),
            "vlm_checkpoint": str(profile.vlm_checkpoint),
            "norm_path": str(profile.norm_path),
            "norm_sha256": profile.norm_sha256,
            "scorer_commit": lock.commit,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                return EpisodeValidation(
                    False,
                    f"manifest_mismatch:{key}",
                    manifest=manifest,
                )
        if manifest.get("source_hashes") != dict(sorted(profile.source_hashes.items())):
            return EpisodeValidation(
                False,
                "manifest_mismatch:source_hashes",
                manifest=manifest,
            )
        if "runtime_topology" not in manifest:
            return EpisodeValidation(
                False, "missing_runtime_topology", manifest=manifest
            )
        if not _matches_runtime_topology(profile, manifest["runtime_topology"]):
            return EpisodeValidation(
                False, "manifest_mismatch:runtime_topology", manifest=manifest
            )
        official_path = Path(manifest["rollout"]["official_result_path"])
        if not official_path.is_file():
            return EpisodeValidation(
                False, "missing_official_result", manifest=manifest
            )
        official_payload = json.loads(official_path.read_text(encoding="utf-8"))
        if int(official_payload.get("source_seed", -1)) != expected_seed:
            return EpisodeValidation(
                False,
                "official_source_seed_mismatch",
                manifest=manifest,
            )
        if int(official_payload.get("source_episode", -1)) != 0:
            return EpisodeValidation(
                False,
                "official_source_episode_mismatch",
                manifest=manifest,
            )
        score_log = Path(str(official_payload.get("score_log", "")))
        if not score_log.is_file() or not score_log.resolve().is_relative_to(
            episode_dir
        ):
            return EpisodeValidation(
                False,
                "invalid_score_log",
                manifest=manifest,
            )
        record = load_official_episode(
            official_path,
            OfficialScoringBridge(repo_root, assets=assets),
            expected_commit=lock.commit,
        )
        if (
            record.task_id != profile.task_id
            or record.episode_index != int(manifest["episode_index"])
            or record.seed != int(manifest["seed"])
        ):
            return EpisodeValidation(
                False, "episode_identity_mismatch", manifest=manifest
            )
        if not official_path.resolve().is_relative_to(episode_dir):
            return EpisodeValidation(
                False,
                "official_result_outside_episode",
                manifest=manifest,
            )
        log_path = Path(manifest["rollout"]["episode_log"])
        if not log_path.is_file() or not log_path.resolve().is_relative_to(episode_dir):
            return EpisodeValidation(False, "missing_episode_log", manifest=manifest)
        videos = [Path(item) for item in manifest["rollout"].get("video_paths", [])]
        if require_video and (
            not videos
            or not all(
                item.is_file() and item.resolve().is_relative_to(episode_dir)
                for item in videos
            )
        ):
            return EpisodeValidation(False, "missing_video", manifest=manifest)
        prompt_trace = manifest["rollout"].get("prompt_trace")
        if prompt_trace:
            prompt_path = Path(prompt_trace)
            if not prompt_path.is_file() or not prompt_path.resolve().is_relative_to(
                episode_dir
            ):
                return EpisodeValidation(
                    False,
                    "invalid_prompt_trace",
                    manifest=manifest,
                )
        return EpisodeValidation(True, "valid", record=record, manifest=manifest)
    except Exception as exc:
        return EpisodeValidation(False, f"{type(exc).__name__}:{exc}")
