from __future__ import annotations


def should_block_forward_until_hold(
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
    """Keep a VLM-selected forward transition pending until the EEF has held.

    This is deliberately a transition timing guard only.  It never generates
    or substitutes a prompt: the caller retains the current VLM prompt until
    the VLM itself supplies a later prompt after the EEF hold has started.
    """
    if not enabled or hold_active or hold_started_before:
        return False
    if selected_subtasks and current_subtask not in selected_subtasks:
        return False
    if current_index is None or next_index is None:
        return False
    return next_subtask != current_subtask and next_index > current_index


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
    if not current_subtask.startswith("pick "):
        return False
    return should_block_forward_until_hold(
        enabled=enabled,
        hold_active=hold_active,
        current_subtask=current_subtask,
        next_subtask=next_subtask,
        current_index=current_index,
        next_index=next_index,
        selected_subtasks=selected_subtasks,
        hold_started_before=hold_started_before,
    )
