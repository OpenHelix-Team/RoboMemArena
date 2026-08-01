import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from evaluation_benchmark.sync_eval_26 import validation
from evaluation_benchmark.sync_eval_26.profile_loader import file_sha256, load_profile

from .conftest import (
    make_external_assets,
    materialize_profile_bddls,
    write_asset_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "evaluation_benchmark" / "sync_eval_26" / "profiles"
PROFILE_PATH = PROFILE_DIR / "task01.yaml"
EXPECTED_COMMIT = "d9f83ac5182e25ad7f0a301a77a0b667f2392df1"


class _Bridge:
    @staticmethod
    def stage_names(_task_id: int) -> list[str]:
        return ["01_Place_Cookies_Basket", "02_Place_Tomato_Basket"]

    @staticmethod
    def stage_score_pct(_task_id: int, stage_done: dict[str, bool]) -> float:
        return 100.0 * sum(stage_done.values()) / 2

    @staticmethod
    def stage_success(_task_id: int, stage_done: dict[str, bool]) -> bool:
        return all(stage_done.values())


def _configure_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    assets = make_external_assets(tmp_path)
    materialize_profile_bddls(PROFILE_DIR, assets)
    config = tmp_path / "assets.yaml"
    write_asset_config(config, assets)
    monkeypatch.setenv("SYNC_EVAL_ASSET_CONFIG", str(config))
    monkeypatch.setattr(
        validation,
        "verify_upstream_lock",
        lambda *_args, **_kwargs: SimpleNamespace(commit=EXPECTED_COMMIT),
    )
    monkeypatch.setattr(validation, "OfficialScoringBridge", lambda *_args, **_kwargs: _Bridge())
    return assets


def _episode_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_path: Path = PROFILE_PATH,
) -> tuple[Path, Path]:
    assets = _configure_runtime(monkeypatch, tmp_path)
    profile = load_profile(profile_path, assets=assets)
    episode_dir = tmp_path / "task01" / "ep000"
    episode_dir.mkdir(parents=True)
    score_log = episode_dir / "score.log"
    driver_log = episode_dir / "driver.log"
    video = episode_dir / "episode.mp4"
    score_log.write_text("official score evidence\n", encoding="utf-8")
    driver_log.write_text("driver evidence\n", encoding="utf-8")
    video.write_bytes(b"video")
    official_path = episode_dir / "official_episode.json"
    official_path.write_text(
        json.dumps(
            {
                "task_id": 1,
                "episode_index": 0,
                "seed": 104,
                "source_seed": 104,
                "source_episode": 0,
                "scorer_commit": EXPECTED_COMMIT,
                "stage_done": {name: True for name in _Bridge.stage_names(1)},
                "score_log": str(score_log),
            }
        ),
        encoding="utf-8",
    )
    manifest_path = episode_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_id": 1,
                "episode_index": 0,
                "seed": 104,
                "profile_sha256": file_sha256(profile_path),
                "vla_checkpoint": str(profile.vla_checkpoint),
                "vlm_checkpoint": str(profile.vlm_checkpoint),
                "norm_path": str(profile.norm_path),
                "norm_sha256": profile.norm_sha256,
                "scorer_commit": EXPECTED_COMMIT,
                "source_hashes": profile.source_hashes,
                "runtime_topology": {
                    "required_gpus": profile.runtime_topology.allocation_gpus,
                    "visible_devices": ["0", "1"],
                    "vla_device": "0",
                    "vlm_device": "1",
                },
                "rollout": {
                    "official_result_path": str(official_path),
                    "episode_log": str(driver_log),
                    "video_paths": [str(video)],
                    "prompt_trace": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return episode_dir, official_path


def _validate(profile_path: Path, episode_dir: Path):
    return validation.validate_episode(
        repo_root=REPO_ROOT,
        profile_path=profile_path,
        episode_dir=episode_dir,
    )


def test_validation_rejects_manifest_source_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir, _ = _episode_fixture(tmp_path, monkeypatch)
    manifest_path = episode_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_path = next(iter(manifest["source_hashes"]))
    manifest["source_hashes"][first_path] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _validate(PROFILE_PATH, episode_dir)

    assert not result.valid
    assert result.reason == "manifest_mismatch:source_hashes"


def test_validation_rejects_stale_profile_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
    source = REPO_ROOT / "evaluation_benchmark" / "sync_eval_26" / "legacy_adapter.py"
    payload["plugin_entrypoint"] = str(source)
    payload["source_paths"] = [str(source)]
    payload["source_hashes"] = {str(source): "0" * 64}
    profile_path = (
        tmp_path / "profiles" / "evaluation_benchmark" / "sync_eval_26" / "profiles" / "task01.yaml"
    )
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    episode_dir, _ = _episode_fixture(tmp_path, monkeypatch, profile_path)

    result = _validate(profile_path, episode_dir)

    assert not result.valid
    assert result.reason == "profile_assets_invalid:ProfileError"


def test_validation_rejects_missing_runtime_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir, _ = _episode_fixture(tmp_path, monkeypatch)
    manifest_path = episode_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("runtime_topology")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _validate(PROFILE_PATH, episode_dir)

    assert not result.valid
    assert result.reason == "missing_runtime_topology"


def test_validation_rejects_wrong_runtime_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir, _ = _episode_fixture(tmp_path, monkeypatch)
    manifest_path = episode_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_topology"] = {
        "required_gpus": 2,
        "visible_devices": ["0"],
        "vla_device": "0",
        "vlm_device": "0",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _validate(PROFILE_PATH, episode_dir)

    assert not result.valid
    assert result.reason == "manifest_mismatch:runtime_topology"


def test_validation_rejects_source_seed_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir, official_path = _episode_fixture(tmp_path, monkeypatch)
    payload = json.loads(official_path.read_text(encoding="utf-8"))
    payload["source_seed"] = 999
    official_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate(PROFILE_PATH, episode_dir)

    assert not result.valid
    assert result.reason == "official_source_seed_mismatch"


def test_validation_rejects_source_episode_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir, official_path = _episode_fixture(tmp_path, monkeypatch)
    payload = json.loads(official_path.read_text(encoding="utf-8"))
    payload["source_episode"] = 1
    official_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate(PROFILE_PATH, episode_dir)

    assert not result.valid
    assert result.reason == "official_source_episode_mismatch"


def test_validation_rejects_score_log_outside_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_dir, official_path = _episode_fixture(tmp_path, monkeypatch)
    outside_log = tmp_path / "outside.log"
    outside_log.write_text("wrong run\n", encoding="utf-8")
    payload = json.loads(official_path.read_text(encoding="utf-8"))
    payload["score_log"] = str(outside_log)
    official_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate(PROFILE_PATH, episode_dir)

    assert not result.valid
    assert result.reason == "invalid_score_log"
