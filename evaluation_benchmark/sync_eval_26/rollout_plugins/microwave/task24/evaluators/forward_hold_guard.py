from __future__ import annotations


def should_block_pick_forward(
    *,
    enabled: bool,
    hold_active: bool,
    current_subtask: str,
    next_subtask: str,
    current_index: int | None,
    next_index: int | None,
    selected_subtasks: set[str],
    hold_started_before: bool = False,
) -> bool:
    if not enabled or hold_active or hold_started_before or not current_subtask.startswith("pick "):
        return False
    if selected_subtasks and current_subtask not in selected_subtasks:
        return False
    if current_index is None or next_index is None:
        return False
    return next_subtask != current_subtask and next_index > current_index
