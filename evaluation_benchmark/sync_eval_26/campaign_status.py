from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .episode_ledger import VALID_SELECTION_STATUSES


# A counted initialization failure is deliberately different from a valid
# rollout: the environment never reached policy inference, but the user asked
# that the documented zero-score outcome occupy its fixed campaign slot rather
# than be retried indefinitely.  Keep this separate so completion and model
# evidence cannot be confused.
COUNTED_INITIALIZATION_FAILURE_STATUSES = frozenset(
    {"counted_initialization_failure"}
)
ACCOUNTED_SELECTION_STATUSES = (
    VALID_SELECTION_STATUSES | COUNTED_INITIALIZATION_FAILURE_STATUSES
)


TASK_STATUS_COLUMNS = (
    "task_id",
    "expected_episodes",
    "valid_episodes",
    "accounted_episodes",
    "remaining_episodes",
    "missing_episodes",
    "invalid_episodes",
    "counted_initialization_failures",
    "stage_success_episodes",
    "tsr_pct",
    "csr_pct",
    "status",
)


def _round_pct(numerator: float, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _read_ledger(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def compute_campaign_identity(
    *,
    profile_dir: Path,
    scorer_lock_path: Path,
) -> str:
    digest = hashlib.sha256()
    inputs = [scorer_lock_path, *sorted(profile_dir.glob("task*.yaml"))]
    if len(inputs) != 27:
        raise RuntimeError(
            f"campaign identity requires scorer lock plus 26 profiles; got {len(inputs)}"
        )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_campaign_status(
    *,
    ledger_path: Path,
    campaign_id: str,
    campaign_identity: str,
    expected_tasks: int = 26,
    episodes_per_task: int = 20,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _read_ledger(ledger_path)
    keyed: dict[int, list[dict[str, str]]] = {
        task_id: [] for task_id in range(1, expected_tasks + 1)
    }
    for row in rows:
        try:
            task_id = int(row["task_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if task_id in keyed:
            keyed[task_id].append(row)

    task_rows: list[dict[str, object]] = []
    total_valid = 0
    total_accounted = 0
    total_initialization_failures = 0
    total_stage_success = 0
    total_stage_score = 0.0
    for task_id in range(1, expected_tasks + 1):
        task_ledger = keyed[task_id]
        valid = [
            row
            for row in task_ledger
            if row.get("selection_status") in VALID_SELECTION_STATUSES
        ]
        counted_initialization_failures = [
            row
            for row in task_ledger
            if row.get("selection_status") in COUNTED_INITIALIZATION_FAILURE_STATUSES
        ]
        accounted = [
            row
            for row in task_ledger
            if row.get("selection_status") in ACCOUNTED_SELECTION_STATUSES
        ]
        missing = sum(row.get("selection_status") == "missing" for row in task_ledger)
        invalid = sum(
            row.get("selection_status")
            not in ACCOUNTED_SELECTION_STATUSES | {"missing"}
            for row in task_ledger
        )
        stage_success = sum(
            row.get("stage_success", "").lower() == "true" for row in accounted
        )
        scores = [
            float(row["stage_score_pct"])
            for row in accounted
            if row.get("stage_score_pct", "") != ""
        ]
        valid_count = len(valid)
        accounted_count = len(accounted)
        task_score_sum = sum(scores)
        remaining = max(0, episodes_per_task - accounted_count)
        task_rows.append(
            {
                "task_id": task_id,
                "expected_episodes": episodes_per_task,
                "valid_episodes": valid_count,
                "accounted_episodes": accounted_count,
                "remaining_episodes": remaining,
                "missing_episodes": missing,
                "invalid_episodes": invalid,
                "counted_initialization_failures": len(
                    counted_initialization_failures
                ),
                "stage_success_episodes": stage_success,
                "tsr_pct": _round_pct(stage_success, accounted_count),
                "csr_pct": round(task_score_sum / len(scores), 4) if scores else 0.0,
                "status": "complete"
                if accounted_count == episodes_per_task
                else "running",
            }
        )
        total_valid += valid_count
        total_accounted += accounted_count
        total_initialization_failures += len(counted_initialization_failures)
        total_stage_success += stage_success
        total_stage_score += task_score_sum

    expected = expected_tasks * episodes_per_task
    progress = {
        "expected_episodes": expected,
        "valid_episodes": total_valid,
        "accounted_episodes": total_accounted,
        "initialization_failure_episodes": total_initialization_failures,
        "remaining_episodes": expected - total_accounted,
        "completion_pct": _round_pct(total_accounted, expected),
        "stage_success_episodes": total_stage_success,
        "tsr_pct": _round_pct(total_stage_success, total_accounted),
        "csr_pct": round(total_stage_score / total_accounted, 4)
        if total_accounted
        else 0.0,
    }
    status: dict[str, object] = {
        "schema_version": 2,
        "campaign_id": campaign_id,
        "campaign_identity": campaign_identity,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_ledger": str(ledger_path.resolve()),
        "progress": progress,
        "tasks": {str(row["task_id"]): row for row in task_rows},
    }
    return status, task_rows


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_tsv(
    path: Path,
    rows: Iterable[dict[str, object]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_campaign_status(
    *,
    status: dict[str, object],
    task_rows: list[dict[str, object]],
    json_path: Path,
    tsv_path: Path,
    allow_profile_revision: bool = False,
    profile_revision_reason: str = "",
) -> None:
    if json_path.is_file():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        previous_identity = previous.get("campaign_identity")
        current_identity = status.get("campaign_identity")
        if previous_identity != current_identity:
            if not allow_profile_revision:
                raise RuntimeError(
                    "refusing campaign identity change: "
                    f"previous={previous_identity} current={current_identity}"
                )
            if not profile_revision_reason:
                raise RuntimeError("profile revision requires a non-empty reason")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            archive_json = json_path.with_name(
                f"{json_path.stem}.{timestamp}_profile_revision.json"
            )
            archive_tsv = tsv_path.with_name(
                f"{tsv_path.stem}.{timestamp}_profile_revision.tsv"
            )
            _atomic_json(archive_json, previous)
            if tsv_path.is_file():
                archive_tsv.write_bytes(tsv_path.read_bytes())
            status["profile_revision"] = {
                "previous_campaign_identity": previous_identity,
                "reason": profile_revision_reason,
                "archived_status_json": str(archive_json.resolve()),
                "archived_status_tsv": str(archive_tsv.resolve()),
            }
        previous_valid = int(previous["progress"]["valid_episodes"])
        current_valid = int(status["progress"]["valid_episodes"])
        if current_valid < previous_valid:
            raise RuntimeError(
                "refusing campaign progress regression: "
                f"previous={previous_valid} current={current_valid}"
            )
        previous_accounted = int(
            previous["progress"].get("accounted_episodes", previous_valid)
        )
        current_accounted = int(
            status["progress"].get("accounted_episodes", current_valid)
        )
        if current_accounted < previous_accounted:
            raise RuntimeError(
                "refusing campaign accounted-progress regression: "
                f"previous={previous_accounted} current={current_accounted}"
            )
    _atomic_json(json_path, status)
    _atomic_tsv(tsv_path, task_rows, TASK_STATUS_COLUMNS)
