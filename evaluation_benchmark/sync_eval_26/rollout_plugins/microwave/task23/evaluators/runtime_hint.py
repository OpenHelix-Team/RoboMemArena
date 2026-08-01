from __future__ import annotations


def _normalize_subtask(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def should_inject_hold_state_hint(
    enabled: bool,
    allowed_subtasks_raw: str,
    hold_active: bool,
    held_subtask: str,
    *,
    state_phase: str = "active",
    configured_phase: str = "active",
) -> bool:
    if not enabled or not hold_active:
        return False
    if _normalize_subtask(state_phase) != _normalize_subtask(configured_phase):
        return False
    allowed = {
        _normalize_subtask(item)
        for item in str(allowed_subtasks_raw).split(",")
        if _normalize_subtask(item)
    }
    return not allowed or _normalize_subtask(held_subtask) in allowed
