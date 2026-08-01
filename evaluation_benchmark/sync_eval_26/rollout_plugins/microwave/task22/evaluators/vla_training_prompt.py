"""Format VLA policy prompts exactly as they appeared in the 35999 training data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingPromptTemplate:
    task_id: int
    allowed_subtasks: tuple[str, ...]
    template: str
    source_meta_sha256: str
    config_path: Path

    @property
    def template_sha256(self) -> str:
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()

    def format(self, current_subtask: str) -> str:
        normalized = " ".join(str(current_subtask).strip().lower().split())
        if normalized not in self.allowed_subtasks:
            raise ValueError(
                f"{self.config_path}: unsupported subtask {current_subtask!r}; "
                f"expected one of {self.allowed_subtasks}"
            )
        return self.template.format(current_subtask=normalized)


def load_training_prompt_template(path: Path, *, expected_task_id: int) -> TrainingPromptTemplate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    task_id = int(payload["task_id"])
    if task_id != int(expected_task_id):
        raise ValueError(f"{path}: task_id={task_id}, expected {expected_task_id}")
    template = str(payload["template"])
    if template.count("{current_subtask}") != 1:
        raise ValueError(f"{path}: template must contain exactly one {{current_subtask}} placeholder")
    allowed = tuple(" ".join(str(value).strip().lower().split()) for value in payload["allowed_subtasks"])
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError(f"{path}: allowed_subtasks must be nonempty and unique")
    source = payload.get("source", {})
    return TrainingPromptTemplate(
        task_id=task_id,
        allowed_subtasks=allowed,
        template=template,
        source_meta_sha256=str(source.get("meta_tasks_sha256", "")),
        config_path=path,
    )
