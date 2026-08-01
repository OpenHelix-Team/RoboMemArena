from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evaluation_benchmark.sync_eval_26.aggregate import EpisodeRecord
from evaluation_benchmark.sync_eval_26.episode_ledger import (
    MappedManifest,
    build_mapped_ledger_rows,
    build_ledger_rows,
    merge_inventory_rows,
)
from evaluation_benchmark.sync_eval_26.validation import EpisodeValidation


def _manifest(root: Path, task_id: int, episode_index: int) -> Path:
    episode = root / f"task{task_id:02d}" / f"ep{episode_index:03d}"
    episode.mkdir(parents=True)
    official = episode / "official_episode.json"
    official.write_text(json.dumps({"goal_success": True}), encoding="utf-8")
    log = episode / "driver.log"
    log.write_text("log\n", encoding="utf-8")
    video = episode / "episode.mp4"
    video.write_bytes(b"video")
    manifest = episode / "run_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "episode_index": episode_index,
                "seed": 100 + episode_index,
                "profile_sha256": "profile",
                "scorer_commit": "scorer",
                "vla_checkpoint": "/vla",
                "vlm_checkpoint": "/vlm",
                "norm_path": "/norm",
                "rollout": {
                    "official_result_path": str(official),
                    "episode_log": str(log),
                    "video_paths": [str(video)],
                },
            }
        ),
        encoding="utf-8",
    )
    return episode


def _validator(*, episode_dir: Path, **_: object) -> EpisodeValidation:
    episode_index = int(episode_dir.name.removeprefix("ep"))
    if episode_index == 1:
        return EpisodeValidation(False, "manifest_mismatch:profile_sha256")
    return EpisodeValidation(
        True,
        "valid",
        record=EpisodeRecord(
            task_id=1,
            episode_index=episode_index,
            seed=100 + episode_index,
            stage_done={"stage": True},
            stage_score_pct=100.0,
            stage_success=True,
            scorer_commit="scorer",
            official_result_path=episode_dir / "official_episode.json",
        ),
    )


def test_ledger_keeps_missing_and_invalid_rows(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    _manifest(root, 1, 0)
    _manifest(root, 1, 1)
    monkeypatch.setattr(
        "evaluation_benchmark.sync_eval_26.episode_ledger.load_profile",
        lambda _: SimpleNamespace(seed=100, seed_mode="incrementing"),
    )

    rows, candidates = build_ledger_rows(
        repo_root=tmp_path,
        profile_dir=tmp_path / "profiles",
        output_roots=[root],
        task_ids=[1],
        episodes=3,
        validator=_validator,
    )

    assert [row["selection_status"] for row in rows] == [
        "valid_selected",
        "invalid_candidate",
        "missing",
    ]
    assert rows[0]["stage_success"] == "true"
    assert rows[1]["validation_reason"] == "manifest_mismatch:profile_sha256"
    assert rows[2]["validation_reason"] == "missing_run_manifest"
    assert len(candidates) == 2


def test_ledger_marks_duplicate_valid_candidates_and_uses_root_priority(
    monkeypatch, tmp_path: Path
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_episode = _manifest(first, 1, 0)
    _manifest(second, 1, 0)
    monkeypatch.setattr(
        "evaluation_benchmark.sync_eval_26.episode_ledger.load_profile",
        lambda _: SimpleNamespace(seed=100, seed_mode="fixed"),
    )

    rows, candidates = build_ledger_rows(
        repo_root=tmp_path,
        profile_dir=tmp_path / "profiles",
        output_roots=[first, second],
        task_ids=[1],
        episodes=1,
        validator=_validator,
    )

    assert rows[0]["selection_status"] == "multiple_valid_selected_by_root_priority"
    assert rows[0]["candidate_count"] == 2
    assert rows[0]["valid_candidate_count"] == 2
    assert rows[0]["episode_dir"] == str(first_episode)
    assert [row["selected"] for row in candidates] == ["true", "false"]


def test_mapped_fixed_seed_trial_uses_global_trial_index(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "shard_a"
    episode = _manifest(source, 1, 0)
    monkeypatch.setattr(
        "evaluation_benchmark.sync_eval_26.episode_ledger.load_profile",
        lambda _: SimpleNamespace(seed=100, seed_mode="fixed"),
    )

    rows, candidates = build_mapped_ledger_rows(
        repo_root=tmp_path,
        profile_dir=tmp_path / "profiles",
        mapped_manifests=[
            MappedManifest(
                task_id=1,
                episode_index=7,
                source_root=source,
                manifest_path=episode / "run_manifest.json",
            )
        ],
        task_ids=[1],
        episodes=8,
        validator=_validator,
    )

    assert rows[7]["selection_status"] == "valid_selected"
    assert rows[7]["episode_index"] == 7
    assert rows[7]["expected_seed"] == 100
    assert rows[7]["episode_dir"] == str(episode)
    assert candidates[0]["episode_index"] == 7


def test_ledger_keeps_ineligible_historical_candidate_out_of_formal_row(
    monkeypatch, tmp_path: Path
) -> None:
    current = tmp_path / "current"
    historical = tmp_path / "historical"
    _manifest(historical, 1, 0)
    monkeypatch.setattr(
        "evaluation_benchmark.sync_eval_26.episode_ledger.load_profile",
        lambda _: SimpleNamespace(seed=100, seed_mode="fixed"),
    )

    rows, candidates = build_ledger_rows(
        repo_root=tmp_path,
        profile_dir=tmp_path / "profiles",
        output_roots=[current, historical],
        task_ids=[1],
        episodes=1,
        eligible_task_ids_by_root=[frozenset({1}), frozenset()],
        validator=_validator,
    )

    assert rows[0]["selection_status"] == "only_ineligible_candidates"
    assert rows[0]["valid_candidate_count"] == 0
    assert candidates[0]["eligible_for_selection"] == "false"
    assert candidates[0]["selected"] == "true"


def test_inventory_merge_does_not_promote_ineligible_historical_row() -> None:
    current_missing = {
        "task_id": 4,
        "episode_index": 11,
        "expected_seed": 115,
        "selection_status": "missing",
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "validation_reason": "missing_run_manifest",
    }
    historical = {
        "task_id": 4,
        "episode_index": 11,
        "expected_seed": 115,
        "selection_status": "only_ineligible_candidates",
        "candidate_count": 1,
        "valid_candidate_count": 0,
        "validation_reason": "valid",
        "manifest_path": "/historical/task04/ep011/run_manifest.json",
    }

    rows, candidates = merge_inventory_rows(
        inventories=[
            {"ledger_rows": [current_missing], "candidate_rows": []},
            {
                "ledger_rows": [historical],
                "candidate_rows": [
                    {
                        "task_id": 4,
                        "episode_index": 11,
                        "manifest_path": historical["manifest_path"],
                        "eligible_for_selection": "false",
                        "selected": "true",
                    }
                ],
            },
        ],
        task_ids=[4],
        episodes=12,
    )

    assert rows[11]["selection_status"] == "only_ineligible_candidates"
    assert rows[11]["valid_candidate_count"] == 0
    assert candidates[0]["selected"] == "true"
