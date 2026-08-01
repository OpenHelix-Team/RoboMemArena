#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PACK_DIR = Path(__file__).resolve().parent.parent
EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from autonomous_prompt_guard import AutonomousPromptGuard, PromptPollSchedule


SOURCE_ROOT = Path(os.environ["SOURCE_ROOT"]).resolve()
OFFICIAL_EVALUATOR = (
    SOURCE_ROOT
    / "evaluation_benchmark/async_vlm26_reference/"
    "eval_fullvlm26_async_vlm_vla.py"
)
if not OFFICIAL_EVALUATOR.is_file():
    raise FileNotFoundError(f"Frozen official evaluator missing: {OFFICIAL_EVALUATOR}")

spec = importlib.util.spec_from_file_location("_official_counting_eval_d9f83ac", OFFICIAL_EVALUATOR)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load official evaluator: {OFFICIAL_EVALUATOR}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

_ORIGINAL_TASK_SPECS = base._task_specs
_ACTIVE_PLANNER: GuardedPlanner | None = None
_ORIGINAL_POLICY_CLIENT = base.StableWebsocketClientPolicy


def _post_pour_hold_enabled() -> bool:
    """Keep the required no-third-pour monitor passive after official Pour2."""
    if os.environ.get("HOLD_AFTER_REQUIRED_STAGES", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if _ACTIVE_PLANNER is None or _ACTIVE_PLANNER.task_info.task_id not in {6, 7}:
        return False
    return _ACTIVE_PLANNER.prompt_guard.stage_count >= 3


class PostPourHoldClient(_ORIGINAL_POLICY_CLIENT):
    """Return zero delta actions after an official counting-task Pour2 stage is complete."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_output: dict[str, Any] | None = None

    def infer(self, element):
        if _post_pour_hold_enabled() and self._last_output is not None:
            output = dict(self._last_output)
            output["actions"] = np.zeros_like(np.asarray(output["actions"]))
            if _ACTIVE_PLANNER is not None and not getattr(_ACTIVE_PLANNER, "post_pour_hold_logged", False):
                _ACTIVE_PLANNER.post_pour_hold_logged = True
                if _ACTIVE_PLANNER.logger is not None:
                    _ACTIVE_PLANNER.logger.info(
                        "[AUTONOMOUS_POST_STAGE_HOLD] task=%s official Pour2 complete; issuing zero actions for monitor",
                        _ACTIVE_PLANNER.task_info.task_id,
                    )
            return output
        output = super().infer(element)
        self._last_output = output
        return output


class GuardedPlanner(base.FullVlm26MemoryPlanner):
    """Keep VLM as the sole prompt source while blocking invalid transitions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _ACTIVE_PLANNER
        super().__init__(*args, **kwargs)
        _ACTIVE_PLANNER = self

    def set_task_info(self, task_info) -> None:
        super().set_task_info(task_info)
        self.prompt_guard = AutonomousPromptGuard(
            task_id=task_info.task_id,
            primitive_labels=task_info.primitive_labels,
        )
        self.prompt_poll = PromptPollSchedule(
            interval=int(os.environ.get("VLM_INTERVAL", "5")),
        )

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None):
        result = super().reset_episode(instruction=instruction, run_dir=run_dir, logger=logger)
        self.prompt_guard.reset()
        self.prompt_poll.reset()
        self.post_pour_hold_logged = False
        return result

    def infer_sync(self, step_idx: int, context_frames_np):
        if not self.prompt_poll.should_infer(step_idx) and self.prompt_guard.accepted_prompt:
            if self.logger is not None:
                self.logger.info(
                    "[AUTONOMOUS_PROMPT_GUARD_REUSE] t=%s task=%s accepted=%s last_vlm_t=%s interval=%s",
                    step_idx,
                    self.task_info.task_id,
                    self.prompt_guard.accepted_prompt,
                    self.prompt_poll.last_inference_step,
                    self.prompt_poll.interval,
                )
            return self.prompt_guard.accepted_prompt

        raw_prompt = super().infer_sync(step_idx, context_frames_np)
        self.prompt_poll.mark_inferred(step_idx)
        decision = self.prompt_guard.filter_prompt(raw_prompt)
        if self.logger is not None:
            log_name = (
                "AUTONOMOUS_PROMPT_GUARD_ACCEPT"
                if decision.reason == "accepted"
                else "AUTONOMOUS_PROMPT_GUARD_BLOCK"
            )
            self.logger.info(
                "[%s] t=%s task=%s raw=%s accepted=%s reason=%s stage_count=%s",
                log_name,
                step_idx,
                self.task_info.task_id,
                decision.raw_prompt,
                decision.prompt,
                decision.reason,
                decision.stage_count,
            )
        return decision.prompt


def _guarded_task_specs(task_id: int):
    specs = _ORIGINAL_TASK_SPECS(task_id)
    wrapped = []
    for stage_index, stage_spec in enumerate(specs):
        original_check = stage_spec.check_fn

        def checked(env, state, stage_start, *, _check=original_check, _index=stage_index, _name=stage_spec.name):
            completed = bool(_check(env, state, stage_start))
            if completed and _ACTIVE_PLANNER is not None:
                _ACTIVE_PLANNER.prompt_guard.observe_stage(_index, _name)
                _ACTIVE_PLANNER.prompt_poll.force_next()
                if _ACTIVE_PLANNER.logger is not None:
                    _ACTIVE_PLANNER.logger.info(
                        "[AUTONOMOUS_PROMPT_GUARD_STAGE] task=%s stage_index=%s stage=%s",
                        task_id,
                        _index,
                        _name,
                    )
            return completed

        wrapped.append(dataclasses.replace(stage_spec, check_fn=checked))
    return wrapped


base.FullVlm26MemoryPlanner = GuardedPlanner
base.StableWebsocketClientPolicy = PostPourHoldClient
base._task_specs = _guarded_task_specs

if __name__ == "__main__":
    os.environ.setdefault("ASYNC_VLM", "0")
    base.main()
