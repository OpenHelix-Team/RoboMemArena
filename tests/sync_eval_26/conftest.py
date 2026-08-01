from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from evaluation_benchmark.sync_eval_26.assets import ExternalAssets


def make_external_assets(root: Path) -> ExternalAssets:
    vla = root / "assets" / "vla"
    norm = vla / "assets" / "policy_assets" / "norm_stats.json"
    official = root / "assets" / "official"
    openpi = root / "assets" / "openpi"
    inference = root / "assets" / "openpi_inference"
    data = root / "assets" / "fullvlm_v2"
    h5dump = root / "assets" / "bin" / "h5dump"
    for path in (vla, official, openpi, inference, data):
        path.mkdir(parents=True, exist_ok=True)
    norm.parent.mkdir(parents=True, exist_ok=True)
    norm.write_text("test norm\n", encoding="utf-8")
    h5dump.parent.mkdir(parents=True, exist_ok=True)
    h5dump.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    vlm_checkpoints: dict[int, Path] = {}
    for task_id in range(1, 27):
        checkpoint = root / "assets" / "vlm" / f"task{task_id:02d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        vlm_checkpoints[task_id] = checkpoint
    return ExternalAssets(
        vla_checkpoint=vla,
        norm_path=norm,
        norm_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),
        vlm_checkpoints=vlm_checkpoints,
        official_runtime_root=official,
        openpi_root=openpi,
        openpi_inference_root=inference,
        fullvlm_data_root=data,
        h5dump_bin=h5dump,
    )


def write_asset_config(path: Path, assets: ExternalAssets) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "vla_checkpoint": str(assets.vla_checkpoint),
                "norm_path": str(assets.norm_path),
                "norm_sha256": assets.norm_sha256,
                "vlm_checkpoints": {
                    task_id: str(checkpoint)
                    for task_id, checkpoint in assets.vlm_checkpoints.items()
                },
                "official_runtime_root": str(assets.official_runtime_root),
                "openpi_root": str(assets.openpi_root),
                "openpi_inference_root": str(assets.openpi_inference_root),
                "fullvlm_data_root": str(assets.fullvlm_data_root),
                "h5dump_bin": str(assets.h5dump_bin),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def materialize_profile_bddls(profile_dir: Path, assets: ExternalAssets) -> None:
    for path in profile_dir.glob("task[0-9][0-9].yaml"):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_bddl = str(payload["bddl_path"])
        bddl = Path(
            raw_bddl.replace(
                "${ASSET_OFFICIAL_RUNTIME_ROOT}",
                str(assets.official_runtime_root),
            )
        )
        bddl.parent.mkdir(parents=True, exist_ok=True)
        bddl.write_text("test bddl\n", encoding="utf-8")
