from pathlib import Path

import pytest
import yaml

from evaluation_benchmark.sync_eval_26.assets import ExternalAssets
from evaluation_benchmark.sync_eval_26.models import ProfileError
from evaluation_benchmark.sync_eval_26.profile_loader import load_profile, parse_profile

from .conftest import make_external_assets


def valid_profile_payload(assets: ExternalAssets) -> dict[str, object]:
    return {
        "task_id": 1,
        "task_name": "task1",
        "status": "frozen-success",
        "plugin_kind": "frozen-subprocess",
        "plugin_entrypoint": "/tmp/plugin.py",
        "bddl_path": "/tmp/task1.bddl",
        "vla_checkpoint": str(assets.vla_checkpoint),
        "vlm_checkpoint": str(assets.vlm_checkpoint_for(1)),
        "norm_path": str(assets.norm_path),
        "norm_sha256": assets.norm_sha256,
        "seed": 104,
        "seed_mode": "increment",
        "replan_steps": 5,
        "max_steps": 1600,
        "runtime_topology": {
            "allocation_gpus": 2,
            "vla_visible_index": 0,
            "vlm_visible_index": 1,
        },
    }


def test_profile_rejects_relative_checkpoint(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    payload = valid_profile_payload(assets)
    payload["vlm_checkpoint"] = "relative/checkpoint"

    with pytest.raises(ProfileError, match="absolute"):
        parse_profile(payload, sync_root=tmp_path, assets=assets)


def test_profile_rejects_wrong_vla(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    payload = valid_profile_payload(assets)
    payload["vla_checkpoint"] = "/tmp/not-expected"

    with pytest.raises(ProfileError, match="does not match external asset config"):
        parse_profile(payload, sync_root=tmp_path, assets=assets)


def test_profile_resolves_task_specific_public_asset_tokens(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    payload = valid_profile_payload(assets)
    payload.update(
        {
            "vla_checkpoint": "${ASSET_VLA_CHECKPOINT}",
            "vlm_checkpoint": "${ASSET_VLM_CHECKPOINT}",
            "norm_path": "${ASSET_NORM_PATH}",
            "norm_sha256": "${ASSET_NORM_SHA256}",
        }
    )

    profile = parse_profile(payload, sync_root=tmp_path, assets=assets)

    assert profile.vla_checkpoint == assets.vla_checkpoint
    assert profile.vlm_checkpoint == assets.vlm_checkpoint_for(1)
    assert profile.norm_path == assets.norm_path


def test_profile_rejects_mismatched_norm(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    payload = valid_profile_payload(assets)
    payload["norm_path"] = "/tmp/wrong_norm.json"

    with pytest.raises(ProfileError, match="norm path does not match external asset config"):
        parse_profile(payload, sync_root=tmp_path, assets=assets)


def test_profile_rejects_wrong_norm_hash(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    payload = valid_profile_payload(assets)
    payload["norm_sha256"] = "0" * 64

    with pytest.raises(ProfileError, match="norm hash does not match external asset config"):
        parse_profile(payload, sync_root=tmp_path, assets=assets)


def test_profile_requires_runtime_topology(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    payload = valid_profile_payload(assets)
    payload.pop("runtime_topology")
    with pytest.raises(ProfileError, match="runtime_topology"):
        parse_profile(payload, sync_root=tmp_path, assets=assets)


def test_valid_profile_loads_from_the_public_package_layout(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    path = tmp_path / "evaluation_benchmark" / "sync_eval_26" / "profiles" / "task01.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(valid_profile_payload(assets)), encoding="utf-8")

    profile = load_profile(path, assets=assets)

    assert profile.task_id == 1
    assert profile.vla_checkpoint == assets.vla_checkpoint
    assert profile.norm_path == assets.norm_path
    assert profile.runtime_topology.allocation_gpus == 2
