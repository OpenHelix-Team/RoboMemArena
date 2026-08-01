#!/usr/bin/env python3
"""Task16 VLM-prompted pour evaluator with an explicit return-to-upright assist.

This is intentionally a separate evaluator from the autonomous guarded path.
The VLM still supplies every prompt.  Once Task16 has completed Lift, its own
accepted prompt is a milk-pour prompt, the milk is already over the red mug,
and the official body tilt has crossed the pour threshold, this wrapper only
overrides rotational action channels to return the held milk upright.  It
never injects a prompt, moves an object, or changes the remote scorer.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

import eval_counting_autonomous_guarded_d9f83ac as guarded


base = guarded.base
stage_eval = base.stage_eval
_ORIGINAL_PATCH_ENV_RESOLUTION = base.patch_env_resolution

_TASK_ID = 16
_OBJECT_NAME = "milk_1"
_TARGET_KIND = "site"
_TARGET_NAME = "red_coffee_mug_1_default_site"
_MOVE_THRESHOLD = 0.15
_RETURN_THRESHOLD = 0.08
_DEFAULT_TARGET_RADIUS = 0.20


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _active_task16_pour_prompt() -> bool:
    planner = guarded._ACTIVE_PLANNER
    if planner is None or planner.task_info.task_id != _TASK_ID:
        return False
    prompt = planner.prompt_guard.accepted_prompt.strip().lower()
    return prompt in {
        "pour milk into red coffee mug 1st",
        "pour milk into red coffee mug 2nd",
    }


def _assist_enabled() -> bool:
    planner = guarded._ACTIVE_PLANNER
    return bool(
        _truthy(os.environ.get("POUR_RETURN_ASSIST"))
        and planner is not None
        and planner.task_info.task_id == _TASK_ID
    )


def _log(message: str, *args: Any) -> None:
    planner = guarded._ACTIVE_PLANNER
    if planner is not None and planner.logger is not None:
        planner.logger.info(message, *args)


def _new_assist_state() -> dict[str, Any]:
    return {
        "last_stage_count": -1,
        "baseline_tilt": None,
        "last_vla_rotation": None,
        "return_active": False,
        "return_steps": 0,
        "return_cycles": 0,
    }


def _task16_snapshot(env: Any) -> tuple[float, float, bool]:
    tilt = stage_eval._body_tilt_angle(env, _OBJECT_NAME)
    source = stage_eval._current_body_pos(env, _OBJECT_NAME)
    target = stage_eval._current_target_center(env, _TARGET_KIND, _TARGET_NAME)
    if tilt is None or source is None or target is None:
        raise RuntimeError(
            "Task16 pour-return assist cannot read milk tilt or red-mug target center."
        )
    radius = float(os.environ.get("POUR_RETURN_ASSIST_TARGET_RADIUS", _DEFAULT_TARGET_RADIUS))
    xy_distance = float(np.linalg.norm(np.asarray(source)[:2] - np.asarray(target)[:2]))
    return float(tilt), xy_distance, bool(xy_distance <= radius)


def _build_return_action(raw_action: Any, last_vla_rotation: Any) -> np.ndarray:
    """Keep every VLA channel except rotation and reverse the latest rotation direction."""
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    rotation = np.asarray(last_vla_rotation, dtype=np.float32).reshape(-1)
    if raw.size < 7 or rotation.size != 3:
        raise ValueError(
            f"Task16 pour-return assist expects a 7-D action and 3-D rotation, got {raw.shape}/{rotation.shape}."
        )
    rotation_norm = float(np.linalg.norm(rotation))
    if rotation_norm < 1e-6:
        raise ValueError("Task16 pour-return assist has no nonzero VLA rotation to reverse.")
    magnitude = float(os.environ.get("POUR_RETURN_ASSIST_ROTATION_MAGNITUDE", "0.8"))
    action = raw.copy()
    action[3:6] = -rotation / rotation_norm * magnitude
    return action


def _install_pour_return_env_patch() -> None:
    """Patch only this process's environment class after the official resolution patch."""
    _ORIGINAL_PATCH_ENV_RESOLUTION()
    if not _truthy(os.environ.get("POUR_RETURN_ASSIST")):
        return

    env_cls = base.ec._get_env_class()
    if getattr(env_cls, "_task16_pour_return_assist_installed", False):
        return

    original_reset = env_cls.reset
    original_step = env_cls.step

    def patched_reset(self, *args: Any, **kwargs: Any):
        result = original_reset(self, *args, **kwargs)
        self._task16_pour_return_assist_state = _new_assist_state()
        return result

    def patched_step(self, action: Any):
        if not _assist_enabled():
            return original_step(self, action)

        planner = guarded._ACTIVE_PLANNER
        assert planner is not None
        stage_count = int(planner.prompt_guard.stage_count)
        state = getattr(self, "_task16_pour_return_assist_state", None)
        if state is None:
            state = _new_assist_state()
            self._task16_pour_return_assist_state = state

        if state["last_stage_count"] != stage_count:
            state["last_stage_count"] = stage_count
            state["baseline_tilt"] = None
            state["last_vla_rotation"] = None
            state["return_active"] = False
            state["return_steps"] = 0
            _log(
                "[POUR_RETURN_ASSIST_STAGE_RESET] task=16 stage_count=%s prompt=%s",
                stage_count,
                planner.prompt_guard.accepted_prompt,
            )

        # Pour stages start after Lift and stop after the required two pours.
        if stage_count < 1 or stage_count >= 3 or not _active_task16_pour_prompt():
            return original_step(self, action)

        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if raw_action.size < 7:
            raise ValueError(f"Task16 pour-return assist received malformed action shape {raw_action.shape}.")

        tilt_before, distance_before, in_target_before = _task16_snapshot(self)
        if state["baseline_tilt"] is None:
            state["baseline_tilt"] = float(tilt_before)
            _log(
                "[POUR_RETURN_ASSIST_BASELINE] task=16 stage_count=%s tilt=%.5f target_xy=%.5f prompt=%s",
                stage_count,
                tilt_before,
                distance_before,
                planner.prompt_guard.accepted_prompt,
            )

        raw_rotation = raw_action[3:6].copy()
        if not state["return_active"] and float(np.linalg.norm(raw_rotation)) >= 1e-4:
            state["last_vla_rotation"] = raw_rotation

        applied_action = raw_action
        if state["return_active"]:
            applied_action = _build_return_action(
                raw_action,
                state["last_vla_rotation"],
            )
            state["return_steps"] += 1
            if state["return_steps"] in {1, 5, 10, 20}:
                _log(
                    "[POUR_RETURN_ASSIST_STEP] task=16 cycle=%s step=%s tilt=%.5f target_xy=%.5f raw_rot=%s applied_rot=%s",
                    state["return_cycles"] + 1,
                    state["return_steps"],
                    tilt_before,
                    distance_before,
                    np.array2string(raw_rotation, precision=4),
                    np.array2string(applied_action[3:6], precision=4),
                )

        result = original_step(self, applied_action.tolist())
        tilt_after, distance_after, in_target_after = _task16_snapshot(self)
        baseline = float(state["baseline_tilt"])
        deviation_after = abs(tilt_after - baseline)

        if not state["return_active"]:
            if (
                in_target_after
                and deviation_after >= _MOVE_THRESHOLD
                and state["last_vla_rotation"] is not None
            ):
                state["return_active"] = True
                state["return_steps"] = 0
                _log(
                    "[POUR_RETURN_ASSIST_ARM] task=16 cycle=%s baseline=%.5f tilt=%.5f deviation=%.5f target_xy=%.5f rotation=%s",
                    state["return_cycles"] + 1,
                    baseline,
                    tilt_after,
                    deviation_after,
                    distance_after,
                    np.array2string(np.asarray(state["last_vla_rotation"]), precision=4),
                )
            return result

        max_steps = int(os.environ.get("POUR_RETURN_ASSIST_MAX_STEPS", "24"))
        if deviation_after <= _RETURN_THRESHOLD:
            state["return_active"] = False
            state["return_cycles"] += 1
            state["return_steps"] = 0
            state["baseline_tilt"] = float(tilt_after)
            state["last_vla_rotation"] = None
            _log(
                "[POUR_RETURN_ASSIST_DONE] task=16 cycle=%s tilt=%.5f deviation=%.5f target_xy=%.5f",
                state["return_cycles"],
                tilt_after,
                deviation_after,
                distance_after,
            )
        elif state["return_steps"] >= max_steps:
            state["return_active"] = False
            state["return_steps"] = 0
            state["baseline_tilt"] = float(tilt_after)
            state["last_vla_rotation"] = None
            _log(
                "[POUR_RETURN_ASSIST_TIMEOUT] task=16 tilt=%.5f deviation=%.5f target_xy=%.5f max_steps=%s",
                tilt_after,
                deviation_after,
                distance_after,
                max_steps,
            )
        return result

    env_cls.reset = patched_reset
    env_cls.step = patched_step
    env_cls._task16_pour_return_assist_installed = True
    base.ec._get_env_class = lambda: env_cls


base.patch_env_resolution = _install_pour_return_env_patch

if __name__ == "__main__":
    if not _truthy(os.environ.get("POUR_RETURN_ASSIST")):
        raise RuntimeError("Set POUR_RETURN_ASSIST=1 to run this controller-assisted evaluator.")
    os.environ.setdefault("ASYNC_VLM", "0")
    base.main()
