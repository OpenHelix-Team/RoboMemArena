from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


DirectionGateResult = tuple[bool, float | None, float | None, float | None, str]


def evaluate_eef_direction_gate(
    eef_pos_history: Sequence[np.ndarray],
    target_pos: np.ndarray,
    current_dist: float,
    signature: dict[str, Any] | None,
    *,
    default_window: int,
    min_displacement: float,
    cos_min: float,
    trend_eps: float,
) -> DirectionGateResult:
    """Check an optional EEF motion direction signature near a hold target."""
    if signature is None:
        return True, None, None, None, "not_configured"

    window = max(1, int(signature.get("window", default_window) or default_window))
    if len(eef_pos_history) <= window:
        return False, None, None, None, "short_history"

    start_pos = np.asarray(eef_pos_history[-1 - window], dtype=np.float64)
    end_pos = np.asarray(eef_pos_history[-1], dtype=np.float64)
    motion = end_pos - start_pos
    displacement = float(np.linalg.norm(motion))
    if displacement < min_displacement:
        return False, None, displacement, None, "low_displacement"

    motion_dir = motion / displacement
    target_dir = np.asarray(signature["direction_mean"], dtype=np.float64)
    cos_sim = float(np.dot(motion_dir, target_dir))
    prev_dist = float(np.linalg.norm(start_pos - target_pos))
    trend_ok = current_dist <= prev_dist + trend_eps
    if cos_sim < cos_min:
        return False, cos_sim, displacement, prev_dist, "cos_low"
    if not trend_ok:
        return False, cos_sim, displacement, prev_dist, "moving_away"
    return True, cos_sim, displacement, prev_dist, "ok"
