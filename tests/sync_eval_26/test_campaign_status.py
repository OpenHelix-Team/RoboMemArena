from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.campaign_status import (
    build_campaign_status,
    write_campaign_status,
)


LEDGER_COLUMNS = (
    "task_id",
    "episode_index",
    "expected_seed",
    "selection_status",
    "validation_reason",
    "stage_success",
    "stage_score_pct",
    "goal_success",
    "scorer_commit",
    "profile_sha256",
    "manifest_path",
    "official_result_path",
    "video_paths_json",
)


def _ledger(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _row(
    task_id: int,
    episode_index: int,
    *,
    status: str = "valid_selected",
    stage_success: bool = True,
    stage_score_pct: float = 100.0,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "episode_index": episode_index,
        "expected_seed": 100 + episode_index,
        "selection_status": status,
        "validation_reason": "valid" if status == "valid_selected" else "missing",
        "stage_success": str(stage_success).lower() if status == "valid_selected" else "",
        "stage_score_pct": stage_score_pct if status == "valid_selected" else "",
        "goal_success": "false" if status == "valid_selected" else "",
        "scorer_commit": "d9f83ac",
        "profile_sha256": "profile",
        "manifest_path": "/manifest",
        "official_result_path": "/official",
        "video_paths_json": '["/video.mp4"]',
    }


def test_campaign_status_reports_progress_tsr_and_csr(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.tsv"
    _ledger(
        ledger,
        [
            _row(1, 0, stage_success=True, stage_score_pct=100.0),
            _row(1, 1, stage_success=False, stage_score_pct=50.0),
            _row(2, 0, status="missing"),
            _row(2, 1, status="invalid_candidate"),
        ],
    )

    status, task_rows = build_campaign_status(
        ledger_path=ledger,
        campaign_id="test",
        campaign_identity="identity",
        expected_tasks=2,
        episodes_per_task=2,
    )

    assert status["progress"] == {
        "expected_episodes": 4,
        "valid_episodes": 2,
        "accounted_episodes": 2,
        "initialization_failure_episodes": 0,
        "remaining_episodes": 2,
        "completion_pct": 50.0,
        "stage_success_episodes": 1,
        "tsr_pct": 50.0,
        "csr_pct": 75.0,
    }
    assert task_rows[0]["valid_episodes"] == 2
    assert task_rows[0]["tsr_pct"] == 50.0
    assert task_rows[0]["csr_pct"] == 75.0
    assert task_rows[1]["missing_episodes"] == 1
    assert task_rows[1]["invalid_episodes"] == 1


def test_campaign_status_counts_documented_initialization_failure_as_zero(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.tsv"
    _ledger(
        ledger,
        [
            _row(1, 0, stage_success=True, stage_score_pct=100.0),
            {
                **_row(1, 1, status="missing"),
                "selection_status": "counted_initialization_failure",
                "validation_reason": "counted_zero: environment initialization failed",
                "stage_success": "false",
                "stage_score_pct": 0.0,
                "goal_success": "false",
            },
        ],
    )

    status, task_rows = build_campaign_status(
        ledger_path=ledger,
        campaign_id="test",
        campaign_identity="identity",
        expected_tasks=1,
        episodes_per_task=2,
    )

    assert status["progress"] == {
        "expected_episodes": 2,
        "valid_episodes": 1,
        "accounted_episodes": 2,
        "initialization_failure_episodes": 1,
        "remaining_episodes": 0,
        "completion_pct": 100.0,
        "stage_success_episodes": 1,
        "tsr_pct": 50.0,
        "csr_pct": 50.0,
    }
    assert task_rows[0]["valid_episodes"] == 1
    assert task_rows[0]["accounted_episodes"] == 2
    assert task_rows[0]["counted_initialization_failures"] == 1
    assert task_rows[0]["status"] == "complete"


def test_campaign_status_refuses_same_identity_regression(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.tsv"
    json_path = tmp_path / "CAMPAIGN_STATUS.json"
    tsv_path = tmp_path / "CAMPAIGN_STATUS.tsv"
    _ledger(ledger, [_row(1, 0)])
    status, task_rows = build_campaign_status(
        ledger_path=ledger,
        campaign_id="test",
        campaign_identity="identity",
        expected_tasks=1,
        episodes_per_task=1,
    )
    write_campaign_status(
        status=status,
        task_rows=task_rows,
        json_path=json_path,
        tsv_path=tsv_path,
    )

    regressed = json.loads(json.dumps(status))
    regressed["progress"]["valid_episodes"] = 0
    with pytest.raises(RuntimeError, match="refusing campaign progress regression"):
        write_campaign_status(
            status=regressed,
            task_rows=task_rows,
            json_path=json_path,
            tsv_path=tsv_path,
        )


def test_campaign_status_archives_explicit_profile_revision(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.tsv"
    json_path = tmp_path / "CAMPAIGN_STATUS.json"
    tsv_path = tmp_path / "CAMPAIGN_STATUS.tsv"
    _ledger(ledger, [_row(1, 0)])
    original, task_rows = build_campaign_status(
        ledger_path=ledger,
        campaign_id="test",
        campaign_identity="old-identity",
        expected_tasks=1,
        episodes_per_task=1,
    )
    write_campaign_status(
        status=original,
        task_rows=task_rows,
        json_path=json_path,
        tsv_path=tsv_path,
    )

    revised, revised_rows = build_campaign_status(
        ledger_path=ledger,
        campaign_id="test",
        campaign_identity="new-identity",
        expected_tasks=1,
        episodes_per_task=1,
    )
    write_campaign_status(
        status=revised,
        task_rows=revised_rows,
        json_path=json_path,
        tsv_path=tsv_path,
        allow_profile_revision=True,
        profile_revision_reason="Task26 canonical metric is fixed seed repeats.",
    )

    current = json.loads(json_path.read_text(encoding="utf-8"))
    assert current["campaign_identity"] == "new-identity"
    assert current["profile_revision"]["previous_campaign_identity"] == "old-identity"
    assert list(tmp_path.glob("CAMPAIGN_STATUS.*_profile_revision.json"))
    assert list(tmp_path.glob("CAMPAIGN_STATUS.*_profile_revision.tsv"))


def test_campaign_status_tsv_uses_unix_line_endings(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.tsv"
    json_path = tmp_path / "CAMPAIGN_STATUS.json"
    tsv_path = tmp_path / "CAMPAIGN_STATUS.tsv"
    _ledger(ledger, [_row(1, 0)])
    status, task_rows = build_campaign_status(
        ledger_path=ledger,
        campaign_id="test",
        campaign_identity="identity",
        expected_tasks=1,
        episodes_per_task=1,
    )

    write_campaign_status(
        status=status,
        task_rows=task_rows,
        json_path=json_path,
        tsv_path=tsv_path,
    )

    assert b"\r" not in tsv_path.read_bytes()
