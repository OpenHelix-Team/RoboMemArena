"""Compatibility bridge between archived sync rollouts and the current base evaluator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def run_archived_sync_rollout(
    legacy_rollout: Callable[..., tuple[float, dict[str, bool], bool, list[Any], list[Any]]],
    stage_success_from_stage_done: Callable[[int, dict[str, bool]], bool],
    *,
    task_id: int,
    env: Any,
    client: Any,
    planner: Any,
    args: Any,
    stage_specs: list[Any],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override: Any,
    vlm_camera_pose: dict | None,
    logger: Any,
    **_current_base_only: Any,
) -> tuple[float, dict[str, bool], bool, dict[str, Any], list[Any], list[Any]]:
    """Adapt the current base evaluator's call contract to the archived rollout.

    The archived rollout already performs the locked official stage scoring.  The
    current base evaluator only additionally requires ``task_id`` and a
    diagnostics mapping, so this bridge does not reinterpret rollout results.
    """
    planner_task_id = int(planner.task_info.task_id)
    if task_id != planner_task_id:
        raise ValueError(
            f"base task_id={task_id} does not match planner task_id={planner_task_id}"
        )

    stage_pct, stage_done, goal_success, replay, replay_wrist = legacy_rollout(
        env=env,
        client=client,
        planner=planner,
        args=args,
        stage_specs=stage_specs,
        goal_monitor_dict=goal_monitor_dict,
        goal_check_override=goal_check_override,
        vlm_camera_pose=vlm_camera_pose,
        logger=logger,
    )
    diagnostics = {
        "stage_success": bool(stage_success_from_stage_done(task_id, stage_done)),
        # The current official base evaluator audits these fields before it
        # writes prompt traces and videos.  Archived rollouts do not implement
        # an extra-pour monitor, so record that explicitly rather than leaving
        # the current-base contract partial.
        "extra_pour_detected": False,
        "failure_reason": "legacy_rollout_completed",
        "rollout_contract": "archived_sync_endpose_hold",
    }
    return stage_pct, stage_done, goal_success, diagnostics, replay, replay_wrist
