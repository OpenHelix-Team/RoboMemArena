from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ASSET_CONFIG_ENV = "SYNC_EVAL_ASSET_CONFIG"
ROOT_TOKEN = "${SYNC26_ROOT}"
ASSET_TOKEN_PREFIX = "${ASSET_"


class AssetConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExternalAssets:
    vla_checkpoint: Path
    norm_path: Path
    norm_sha256: str
    vlm_checkpoints: Mapping[int, Path]
    official_runtime_root: Path
    openpi_root: Path
    openpi_inference_root: Path
    fullvlm_data_root: Path
    h5dump_bin: Path | None

    def vlm_checkpoint_for(self, task_id: int) -> Path:
        try:
            return self.vlm_checkpoints[task_id]
        except KeyError as exc:
            raise AssetConfigError(
                f"missing VLM checkpoint for task {task_id}"
            ) from exc


def _absolute(value: Any, field: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        raise AssetConfigError(f"{field} must be an absolute path: {path}")
    return path


def load_external_assets(config_path: Path | None = None) -> ExternalAssets:
    if config_path is None:
        raw = os.environ.get(ASSET_CONFIG_ENV, "").strip()
        if not raw:
            raise AssetConfigError(
                f"set {ASSET_CONFIG_ENV} to an external asset YAML file"
            )
        config_path = Path(raw)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AssetConfigError(f"asset config must be a mapping: {config_path}")
    required = {
        "vla_checkpoint",
        "norm_path",
        "norm_sha256",
        "vlm_checkpoints",
        "official_runtime_root",
        "openpi_root",
        "openpi_inference_root",
        "fullvlm_data_root",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise AssetConfigError(
            f"asset config missing fields: {', '.join(missing)}"
        )
    norm_sha256 = str(payload["norm_sha256"])
    if len(norm_sha256) != 64 or any(char not in "0123456789abcdef" for char in norm_sha256.lower()):
        raise AssetConfigError("norm_sha256 must be a 64-character hexadecimal digest")
    raw_vlm_checkpoints = payload["vlm_checkpoints"]
    if not isinstance(raw_vlm_checkpoints, Mapping):
        raise AssetConfigError("vlm_checkpoints must be a mapping of task ID to path")
    vlm_checkpoints: dict[int, Path] = {}
    for raw_task_id, raw_path in raw_vlm_checkpoints.items():
        try:
            task_id = int(raw_task_id)
        except (TypeError, ValueError) as exc:
            raise AssetConfigError(
                f"invalid VLM task ID: {raw_task_id!r}"
            ) from exc
        if task_id < 1 or task_id > 26:
            raise AssetConfigError(f"VLM task ID must be in 1..26: {task_id}")
        if task_id in vlm_checkpoints:
            raise AssetConfigError(f"duplicate VLM task ID: {task_id}")
        vlm_checkpoints[task_id] = _absolute(
            raw_path, f"vlm_checkpoints[{task_id}]"
        )
    missing_vlm_tasks = sorted(set(range(1, 27)) - set(vlm_checkpoints))
    if missing_vlm_tasks:
        raise AssetConfigError(
            "vlm_checkpoints missing task IDs: "
            + ", ".join(str(task_id) for task_id in missing_vlm_tasks)
        )
    h5dump_raw = payload.get("h5dump_bin")
    return ExternalAssets(
        vla_checkpoint=_absolute(payload["vla_checkpoint"], "vla_checkpoint"),
        norm_path=_absolute(payload["norm_path"], "norm_path"),
        norm_sha256=norm_sha256.lower(),
        vlm_checkpoints=vlm_checkpoints,
        official_runtime_root=_absolute(
            payload["official_runtime_root"], "official_runtime_root"
        ),
        openpi_root=_absolute(payload["openpi_root"], "openpi_root"),
        openpi_inference_root=_absolute(
            payload["openpi_inference_root"], "openpi_inference_root"
        ),
        fullvlm_data_root=_absolute(payload["fullvlm_data_root"], "fullvlm_data_root"),
        h5dump_bin=_absolute(h5dump_raw, "h5dump_bin") if h5dump_raw else None,
    )


def token_values(
    *, sync_root: Path, assets: ExternalAssets, task_id: int
) -> dict[str, str]:
    values = {
        ROOT_TOKEN: str(sync_root.resolve()),
        "${ASSET_VLA_CHECKPOINT}": str(assets.vla_checkpoint),
        "${ASSET_NORM_PATH}": str(assets.norm_path),
        "${ASSET_NORM_SHA256}": assets.norm_sha256,
        "${ASSET_VLM_CHECKPOINT}": str(assets.vlm_checkpoint_for(task_id)),
        "${ASSET_OFFICIAL_RUNTIME_ROOT}": str(assets.official_runtime_root),
        "${ASSET_OPENPI_ROOT}": str(assets.openpi_root),
        "${ASSET_OPENPI_INFERENCE_ROOT}": str(assets.openpi_inference_root),
        "${ASSET_FULLVLM_DATA_ROOT}": str(assets.fullvlm_data_root),
    }
    if assets.h5dump_bin is not None:
        values["${ASSET_H5DUMP_BIN}"] = str(assets.h5dump_bin)
    return values


def resolve_tokens(
    value: str, *, sync_root: Path, assets: ExternalAssets, task_id: int
) -> str:
    resolved = value
    for token, replacement in token_values(
        sync_root=sync_root, assets=assets, task_id=task_id
    ).items():
        resolved = resolved.replace(token, replacement)
    if ROOT_TOKEN in resolved or ASSET_TOKEN_PREFIX in resolved:
        raise AssetConfigError(f"unresolved public asset token: {value}")
    return resolved
