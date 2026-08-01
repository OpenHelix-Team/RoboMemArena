from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .scoring_bridge import OfficialScoringBridge


@dataclass(frozen=True)
class EpisodeRecord:
    task_id: int
    episode_index: int
    seed: int
    stage_done: dict[str, bool]
    stage_score_pct: float
    stage_success: bool
    scorer_commit: str
    official_result_path: Path


@dataclass(frozen=True)
class TaskAggregate:
    task_id: int
    episodes: int
    successes: int
    csr_pct: float
    tsr: float


def load_official_episode(
    path: Path,
    bridge: OfficialScoringBridge,
    expected_commit: str,
) -> EpisodeRecord:
    payload = json.loads(path.read_text(encoding="utf-8"))
    commit = str(payload["scorer_commit"])
    if commit != expected_commit:
        raise ValueError(
            f"scorer commit mismatch in {path}: expected={expected_commit} actual={commit}"
        )
    task_id = int(payload["task_id"])
    stage_done = {str(k): bool(v) for k, v in payload["stage_done"].items()}
    expected_names = bridge.stage_names(task_id)
    if list(stage_done) != expected_names:
        raise ValueError(
            f"official stage names mismatch for task {task_id}: "
            f"expected={expected_names} actual={list(stage_done)}"
        )
    return EpisodeRecord(
        task_id=task_id,
        episode_index=int(payload["episode_index"]),
        seed=int(payload["seed"]),
        stage_done=stage_done,
        stage_score_pct=bridge.stage_score_pct(task_id, stage_done),
        stage_success=bridge.stage_success(task_id, stage_done),
        scorer_commit=commit,
        official_result_path=path,
    )


def aggregate_task(
    records: Iterable[EpisodeRecord], *, expected_episodes: int
) -> TaskAggregate:
    rows = list(records)
    if not rows:
        raise ValueError("no episode records")
    keys = [(row.task_id, row.episode_index) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/episode record")
    task_ids = {row.task_id for row in rows}
    if len(task_ids) != 1:
        raise ValueError(f"records contain multiple tasks: {sorted(task_ids)}")
    if len(rows) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episodes, got {len(rows)}")

    return TaskAggregate(
        task_id=rows[0].task_id,
        episodes=len(rows),
        successes=sum(int(row.stage_success) for row in rows),
        csr_pct=sum(row.stage_score_pct for row in rows) / len(rows),
        tsr=sum(int(row.stage_success) for row in rows) / len(rows),
    )
