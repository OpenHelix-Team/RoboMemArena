from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from .assets import ExternalAssets, load_external_assets, resolve_tokens
from .models import ProfileError, ProfileVerification, RuntimeTopology, TaskProfile


REQUIRED_FIELDS = {
    "task_id",
    "task_name",
    "status",
    "plugin_kind",
    "plugin_entrypoint",
    "bddl_path",
    "vla_checkpoint",
    "vlm_checkpoint",
    "norm_path",
    "norm_sha256",
    "seed",
    "seed_mode",
    "replan_steps",
    "max_steps",
    "runtime_topology",
}
VALID_STATUSES = {"frozen-success", "best-local", "experimental"}
VALID_PLUGIN_KINDS = {"native", "frozen-subprocess"}
VALID_SEED_MODES = {"increment", "fixed"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute_path(
    value: Any,
    field: str,
    *,
    sync_root: Path,
    assets: ExternalAssets,
    task_id: int,
) -> Path:
    raw = resolve_tokens(
        str(value), sync_root=sync_root, assets=assets, task_id=task_id
    )
    path = Path(raw)
    if not path.is_absolute():
        raise ProfileError(f"{field} path must be absolute after resolution: {path}")
    return path


def _sync_root_from_profile(profile_path: Path) -> Path:
    profile_path = profile_path.resolve()
    parents = profile_path.parents
    if len(parents) < 4:
        raise ProfileError(f"profile path is outside sync_eval_26/profiles: {profile_path}")
    sync_dir = parents[1]
    if sync_dir.name != "sync_eval_26" or parents[0].name != "profiles":
        raise ProfileError(
            "profiles must live under evaluation_benchmark/sync_eval_26/profiles: "
            f"{profile_path}"
        )
    return parents[3]


def expected_vla_assets(
    *, assets: ExternalAssets | None = None
) -> tuple[Path, Path, str]:
    active = assets or load_external_assets()
    return active.vla_checkpoint, active.norm_path, active.norm_sha256


def parse_profile(
    payload: Mapping[str, Any],
    *,
    sync_root: Path,
    assets: ExternalAssets | None = None,
) -> TaskProfile:
    missing = sorted(REQUIRED_FIELDS - payload.keys())
    if missing:
        raise ProfileError(f"missing required profile fields: {', '.join(missing)}")
    active_assets = assets or load_external_assets()
    sync_root = sync_root.resolve()

    task_id = int(payload["task_id"])
    if not 1 <= task_id <= 26:
        raise ProfileError(f"task_id must be in 1..26: {task_id}")
    status = str(payload["status"])
    if status not in VALID_STATUSES:
        raise ProfileError(f"invalid status: {status}")
    plugin_kind = str(payload["plugin_kind"])
    if plugin_kind not in VALID_PLUGIN_KINDS:
        raise ProfileError(f"invalid plugin_kind: {plugin_kind}")
    seed_mode = str(payload["seed_mode"])
    if seed_mode not in VALID_SEED_MODES:
        raise ProfileError(f"invalid seed_mode: {seed_mode}")

    expected_checkpoint, expected_norm_path, expected_norm_sha256 = expected_vla_assets(
        assets=active_assets
    )
    vla_checkpoint = _absolute_path(
        payload["vla_checkpoint"],
        "vla_checkpoint",
        sync_root=sync_root,
        assets=active_assets,
        task_id=task_id,
    )
    if vla_checkpoint != expected_checkpoint:
        raise ProfileError(
            f"profile VLA checkpoint does not match external asset config: "
            f"expected={expected_checkpoint} actual={vla_checkpoint}"
        )
    norm_path = _absolute_path(
        payload["norm_path"],
        "norm_path",
        sync_root=sync_root,
        assets=active_assets,
        task_id=task_id,
    )
    if norm_path != expected_norm_path:
        raise ProfileError(
            f"profile norm path does not match external asset config: "
            f"expected={expected_norm_path} actual={norm_path}"
        )
    norm_sha256 = str(
        resolve_tokens(
            str(payload["norm_sha256"]),
            sync_root=sync_root,
            assets=active_assets,
            task_id=task_id,
        )
    )
    if norm_sha256 != expected_norm_sha256:
        raise ProfileError(
            f"profile norm hash does not match external asset config: "
            f"expected={expected_norm_sha256} actual={norm_sha256}"
        )
    vlm_checkpoint = _absolute_path(
        payload["vlm_checkpoint"],
        "vlm_checkpoint",
        sync_root=sync_root,
        assets=active_assets,
        task_id=task_id,
    )
    expected_vlm_checkpoint = active_assets.vlm_checkpoint_for(task_id)
    if vlm_checkpoint != expected_vlm_checkpoint:
        raise ProfileError(
            "profile VLM checkpoint does not match external asset config: "
            f"expected={expected_vlm_checkpoint} actual={vlm_checkpoint}"
        )

    source_paths = tuple(
        _absolute_path(
            item,
            "source_paths",
            sync_root=sync_root,
            assets=active_assets,
            task_id=task_id,
        )
        for item in payload.get("source_paths", [])
    )
    source_hashes = {
        str(
            _absolute_path(
                key,
                "source_hashes",
                sync_root=sync_root,
                assets=active_assets,
                task_id=task_id,
            )
        ): str(value)
        for key, value in payload.get("source_hashes", {}).items()
    }
    runtime_env = {
        str(key): resolve_tokens(
            str(value), sync_root=sync_root, assets=active_assets, task_id=task_id
        )
        for key, value in payload.get("runtime_env", {}).items()
    }
    raw_topology = payload["runtime_topology"]
    if not isinstance(raw_topology, Mapping):
        raise ProfileError("runtime_topology must be a mapping")
    required_topology_fields = {
        "allocation_gpus",
        "vla_visible_index",
        "vlm_visible_index",
    }
    missing_topology = sorted(required_topology_fields - raw_topology.keys())
    if missing_topology:
        raise ProfileError(
            "runtime_topology missing fields: " + ", ".join(missing_topology)
        )
    runtime_topology = RuntimeTopology(
        allocation_gpus=int(raw_topology["allocation_gpus"]),
        vla_visible_index=int(raw_topology["vla_visible_index"]),
        vlm_visible_index=int(raw_topology["vlm_visible_index"]),
    )
    return TaskProfile(
        task_id=task_id,
        task_name=str(payload["task_name"]),
        status=status,  # type: ignore[arg-type]
        plugin_kind=plugin_kind,  # type: ignore[arg-type]
        plugin_entrypoint=_absolute_path(
            payload["plugin_entrypoint"],
            "plugin_entrypoint",
            sync_root=sync_root,
            assets=active_assets,
            task_id=task_id,
        ),
        bddl_path=_absolute_path(
            payload["bddl_path"],
            "bddl_path",
            sync_root=sync_root,
            assets=active_assets,
            task_id=task_id,
        ),
        vla_checkpoint=vla_checkpoint,
        vlm_checkpoint=vlm_checkpoint,
        norm_path=norm_path,
        norm_sha256=norm_sha256,
        seed=int(payload["seed"]),
        seed_mode=seed_mode,  # type: ignore[arg-type]
        replan_steps=int(payload["replan_steps"]),
        max_steps=int(payload["max_steps"]),
        runtime_topology=runtime_topology,
        runtime_env=runtime_env,
        prompt_config=dict(payload.get("prompt_config", {})),
        source_paths=source_paths,
        source_hashes=source_hashes,
        hf_assets=dict(payload.get("hf_assets", {})),
    )


def load_profile(path: Path, *, assets: ExternalAssets | None = None) -> TaskProfile:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProfileError(f"profile root must be a mapping: {path}")
    return parse_profile(
        payload,
        sync_root=_sync_root_from_profile(path),
        assets=assets,
    )


def verify_profile_assets(profile: TaskProfile) -> ProfileVerification:
    paths = (
        profile.plugin_entrypoint,
        profile.bddl_path,
        profile.vla_checkpoint,
        profile.vlm_checkpoint,
        profile.norm_path,
        *profile.source_paths,
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    actual_norm = file_sha256(profile.norm_path)
    if actual_norm != profile.norm_sha256:
        raise ProfileError(
            f"norm SHA256 mismatch: expected={profile.norm_sha256} actual={actual_norm}"
        )
    checked_hashes = 1
    for raw_path, expected in profile.source_hashes.items():
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise ProfileError(
                f"source SHA256 mismatch: {path}: expected={expected} actual={actual}"
            )
        checked_hashes += 1
    return ProfileVerification(
        task_id=profile.task_id,
        checked_paths=len(paths),
        checked_hashes=checked_hashes,
    )
