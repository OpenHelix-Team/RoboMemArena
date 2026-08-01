import sys
from pathlib import Path

import pytest


EVALUATOR_DIR = (
    Path(__file__).resolve().parents[2]
    / "evaluation_benchmark"
    / "sync_eval_26"
    / "rollout_plugins"
    / "archived_success_pack"
    / "evaluators"
)
sys.path.insert(0, str(EVALUATOR_DIR))

from legacy_rollout_compat import run_archived_sync_rollout  # noqa: E402


class _Planner:
    class task_info:
        task_id = 5


def test_archived_rollout_accepts_current_base_contract() -> None:
    received: dict[str, object] = {}

    def legacy_rollout(**kwargs):
        received.update(kwargs)
        return 66.7, {"01_Open": True, "02_Place": False}, False, ["main"], ["wrist"]

    result = run_archived_sync_rollout(
        legacy_rollout,
        lambda task_id, done: task_id == 5 and all(done.values()),
        task_id=5,
        env="env",
        client="client",
        planner=_Planner(),
        args="args",
        stage_specs=["stage"],
        goal_monitor_dict={},
        goal_check_override=None,
        vlm_camera_pose=None,
        logger="logger",
        fail_on_extra_pour=False,
        extra_pour_monitor_steps=0,
        post_goal_steps=200,
    )

    score, stage_done, goal, diagnostics, main, wrist = result
    assert (score, stage_done, goal, main, wrist) == (
        66.7,
        {"01_Open": True, "02_Place": False},
        False,
        ["main"],
        ["wrist"],
    )
    assert diagnostics["stage_success"] is False
    assert diagnostics["extra_pour_detected"] is False
    assert diagnostics["failure_reason"] == "legacy_rollout_completed"
    assert diagnostics["rollout_contract"] == "archived_sync_endpose_hold"
    assert "task_id" not in received
    assert "post_goal_steps" not in received


def test_archived_rollout_rejects_task_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        run_archived_sync_rollout(
            lambda **_kwargs: (0.0, {}, False, [], []),
            lambda _task_id, _done: False,
            task_id=4,
            env=None,
            client=None,
            planner=_Planner(),
            args=None,
            stage_specs=[],
            goal_monitor_dict={},
            goal_check_override=None,
            vlm_camera_pose=None,
            logger=None,
        )
