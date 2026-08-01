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


@dataclass
class Task8OraclePickAfterPlace:
    """Diagnostic-only Task8 prompt injection after a VLM-selected place prompt."""

    delay_steps: int
    place_prompt_step: int | None = None

    def reset(self) -> None:
        self.place_prompt_step = None

    def should_force(
        self,
        *,
        task_id: int,
        step_idx: int,
        accepted_prompt: str,
        stage_count: int,
    ) -> bool:
        if self.delay_steps < 0 or int(task_id) != 8 or int(stage_count) >= 1:
            return False
        if self.place_prompt_step is None:
            if _normalize(accepted_prompt) == "place chocolate in frypan":
                self.place_prompt_step = int(step_idx)
            return False
        return int(step_idx) >= self.place_prompt_step + self.delay_steps


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

# This is deliberately opt-in.  It preserves a VLM-selected forward prompt
# while rejecting only a later VLM attempt to go back to an earlier primitive.
# It never synthesizes or advances a prompt.
_SEMANTIC_ORDER_BY_TASK: dict[int, dict[str, int]] = {
    8: {
        "pick chocolate": 0,
        "place chocolate in frypan": 1,
        "pick tomato sauce": 2,
        "pour tomato sauce into frypan 1st": 3,
        "pour tomato sauce into frypan 2nd": 4,
        "place tomato sauce bowl drainer": 5,
    },
}


def _normalize(prompt: str) -> str:
    return " ".join(str(prompt).strip().lower().replace("_", " ").split())


class AutonomousPromptGuard:
    """Keep VLM prompts autonomous while rejecting only physical-stage regressions."""

    def __init__(
        self,
        *,
        task_id: int,
        primitive_labels: list[str],
        stage_latch: bool = False,
        no_prompt_regression: bool = False,
    ) -> None:
        self.task_id = int(task_id)
        self.primitive_labels = [_normalize(label) for label in primitive_labels]
        self.minimum_stages = _MIN_COMPLETED_STAGES_BY_TASK.get(self.task_id, {})
        self.semantic_order = _SEMANTIC_ORDER_BY_TASK.get(self.task_id, {})
        self.stage_latch_enabled = bool(stage_latch)
        self.no_prompt_regression = bool(no_prompt_regression)
        self.reset()

    def reset(self) -> None:
        self.accepted_prompt = ""
        self.stage_count = 0
        self.stage_names: list[str] = []
        self.stage_latch_pending = False
        self.highest_vlm_prompt_order: int | None = None

    def observe_stage(self, stage_index: int, stage_name: str) -> None:
        previous_count = self.stage_count
        self.stage_count = max(self.stage_count, int(stage_index) + 1)
        if stage_name not in self.stage_names:
            self.stage_names.append(stage_name)
        if (
            self.stage_latch_enabled
            and self.stage_count > previous_count
            and self.stage_count < 3
        ):
            self.stage_latch_pending = True

    def force_prompt(self, prompt: str, *, raw_prompt: str = "") -> PromptDecision:
        """Record an explicitly marked diagnostic prompt injection."""
        normalized = _normalize(prompt)
        self.accepted_prompt = normalized
        prompt_order = self.semantic_order.get(normalized)
        if self.no_prompt_regression and prompt_order is not None:
            self.highest_vlm_prompt_order = max(
                prompt_order,
                self.highest_vlm_prompt_order if self.highest_vlm_prompt_order is not None else prompt_order,
            )
        return PromptDecision(normalized, _normalize(raw_prompt), "oracle_forced", self.stage_count)

    def _finish_decision(self, decision: PromptDecision) -> PromptDecision:
        """Release only on a VLM-produced known prompt beyond the completed stage."""
        if not self.stage_latch_pending:
            return decision
        required = self.minimum_stages.get(decision.raw_prompt)
        if decision.reason == "accepted" and required is not None and required >= self.stage_count:
            self.stage_latch_pending = False
        return decision

    def filter_prompt(self, raw_prompt: str) -> PromptDecision:
        raw = _normalize(raw_prompt)
        if not raw:
            return self._finish_decision(
                PromptDecision(self.accepted_prompt, raw, "accepted", self.stage_count)
            )

        known_physical_prompt = raw in self.minimum_stages
        known_semantic_prompt = raw in self.semantic_order
        if not known_physical_prompt and not known_semantic_prompt:
            prompt = self.accepted_prompt or raw
            return self._finish_decision(
                PromptDecision(prompt, raw, "unknown_prompt", self.stage_count)
            )

        prompt_order = self.semantic_order.get(raw)
        if (
            self.no_prompt_regression
            and prompt_order is not None
            and self.highest_vlm_prompt_order is not None
            and prompt_order < self.highest_vlm_prompt_order
        ):
            prompt = self.accepted_prompt or raw
            return self._finish_decision(
                PromptDecision(prompt, raw, "semantic_regression", self.stage_count)
            )

        # A prompt is only a regression after the corresponding physical stage
        # has actually completed.  Do not use the prompt order to block a VLM
        # forward guess, and allow recovery to pick before any stage completes.
        required = self.minimum_stages.get(raw)
        if required is not None and self.stage_count > required:
            prompt = self.accepted_prompt or raw
            return self._finish_decision(
                PromptDecision(prompt, raw, "regression", self.stage_count)
            )

        self.accepted_prompt = raw
        if self.no_prompt_regression and prompt_order is not None:
            self.highest_vlm_prompt_order = max(
                prompt_order,
                self.highest_vlm_prompt_order if self.highest_vlm_prompt_order is not None else prompt_order,
            )
        return self._finish_decision(PromptDecision(raw, raw, "accepted", self.stage_count))
