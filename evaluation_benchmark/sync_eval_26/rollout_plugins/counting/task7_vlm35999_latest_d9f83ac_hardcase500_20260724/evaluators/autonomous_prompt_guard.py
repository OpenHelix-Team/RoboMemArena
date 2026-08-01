from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptDecision:
    prompt: str
    raw_prompt: str
    reason: str
    stage_count: int


class PromptPollSchedule:
    """Throttle expensive VLM calls while retaining only VLM-produced prompts."""

    def __init__(self, *, interval: int) -> None:
        self.interval = max(1, int(interval))
        self.reset()

    def reset(self) -> None:
        self.last_inference_step: int | None = None
        self.force_inference = False

    def should_infer(self, step_idx: int) -> bool:
        if self.last_inference_step is None or self.force_inference:
            return True
        return int(step_idx) - self.last_inference_step >= self.interval

    def mark_inferred(self, step_idx: int) -> None:
        self.last_inference_step = int(step_idx)
        self.force_inference = False

    def force_next(self) -> None:
        self.force_inference = True


_MIN_COMPLETED_STAGES_BY_TASK: dict[int, dict[str, int]] = {
    6: {
        "pick tomato sauce": 0,
        "pour tomato sauce over cookies 1st": 1,
        "pour tomato sauce over cookies 2nd": 2,
        "place tomato sauce bowl drainer": 3,
    },
    7: {
        "pick tomato sauce": 0,
        "pour tomato sauce into frypan 1st": 1,
        "pour tomato sauce into frypan 2nd": 2,
        "place tomato sauce bowl drainer": 3,
    },
    10: {
        "pick wine bottle": 0,
        "pour wine into mug 1st": 1,
        "pour wine into mug 2nd": 2,
        "place wine bottle on table": 3,
    },
    15: {
        "pick butter": 0,
        "place butter in frypan": 0,
        "pick milk": 0,
        "pour milk in frypan 1st": 1,
        "pour milk in frypan 2nd": 2,
        "place milk on table": 3,
    },
    16: {
        "pick milk": 0,
        "pour milk into red coffee mug 1st": 1,
        "pour milk into red coffee mug 2nd": 2,
        "place milk bowl drainer": 3,
    },
}


def _normalize(prompt: str) -> str:
    return " ".join(str(prompt).strip().lower().replace("_", " ").split())


class AutonomousPromptGuard:
    """Reject invalid VLM transitions without generating a replacement prompt."""

    def __init__(self, *, task_id: int, primitive_labels: list[str]) -> None:
        self.task_id = int(task_id)
        self.primitive_labels = [_normalize(label) for label in primitive_labels]
        self.minimum_stages = _MIN_COMPLETED_STAGES_BY_TASK.get(self.task_id, {})
        self.reset()

    def reset(self) -> None:
        self.accepted_prompt = ""
        self.stage_count = 0
        self.stage_names: list[str] = []

    def observe_stage(self, stage_index: int, stage_name: str) -> None:
        self.stage_count = max(self.stage_count, int(stage_index) + 1)
        if stage_name not in self.stage_names:
            self.stage_names.append(stage_name)

    def filter_prompt(self, raw_prompt: str) -> PromptDecision:
        raw = _normalize(raw_prompt)
        if not raw or not self.minimum_stages:
            if raw:
                self.accepted_prompt = raw
            return PromptDecision(self.accepted_prompt, raw, "accepted", self.stage_count)

        if raw not in self.minimum_stages:
            prompt = self.accepted_prompt or raw
            return PromptDecision(prompt, raw, "unknown_prompt", self.stage_count)

        required = self.minimum_stages[raw]
        if self.stage_count < required:
            prompt = self.accepted_prompt or raw
            return PromptDecision(prompt, raw, "required_stage_incomplete", self.stage_count)

        if self.accepted_prompt in self.primitive_labels:
            current_index = self.primitive_labels.index(self.accepted_prompt)
            raw_index = self.primitive_labels.index(raw)
            if raw_index < current_index:
                return PromptDecision(self.accepted_prompt, raw, "regression", self.stage_count)

        self.accepted_prompt = raw
        return PromptDecision(raw, raw, "accepted", self.stage_count)
