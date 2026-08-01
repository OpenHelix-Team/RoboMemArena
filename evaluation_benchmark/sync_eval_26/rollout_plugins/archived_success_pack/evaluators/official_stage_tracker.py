"""Ordered tracker for the frozen RoboMemArena stage definitions."""

from __future__ import annotations

from typing import Any


class OrderedOfficialStageTracker:
    def __init__(self, stage_module: Any, *, task_id: int, env: Any) -> None:
        self._stage_module = stage_module
        self._task_id = task_id
        self._specs = list(stage_module._task_specs(task_id))
        self._state = stage_module._build_initial_state(env)
        self._stage_index = 0
        self._stage_start = self._state["step_idx"]
        self.done = {spec.name: False for spec in self._specs}

    def update(self, *, env: Any, obs: Any) -> list[str]:
        self._stage_module._update_state(obs, self._state)
        if self._stage_index >= len(self._specs):
            return []
        spec = self._specs[self._stage_index]
        if not spec.check_fn(env, self._state, self._stage_start):
            return []
        self.done[spec.name] = True
        self._stage_index += 1
        self._stage_start = self._state["step_idx"]
        return [spec.name]
