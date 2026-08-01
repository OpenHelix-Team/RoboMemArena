from __future__ import annotations

import numpy as np


def place_release_guard_distance(
    prompt: str,
    current_eef: np.ndarray,
    target_eef: np.ndarray,
    *,
    enabled: bool,
) -> float | None:
    """Return EEF distance for an enabled place primitive, otherwise None."""
    normalized = " ".join(str(prompt).strip().lower().split())
    if not enabled or not normalized.startswith("place "):
        return None
    current = np.asarray(current_eef, dtype=np.float64).reshape(-1)
    target = np.asarray(target_eef, dtype=np.float64).reshape(-1)
    if current.size != 3 or target.size != 3:
        raise ValueError("current_eef and target_eef must each contain exactly three values")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise ValueError("current_eef and target_eef must be finite")
    return float(np.linalg.norm(current - target))


def should_keep_place_gripper_closed(
    prompt: str,
    current_eef: np.ndarray,
    target_eef: np.ndarray,
    tolerance: float,
    *,
    enabled: bool,
) -> tuple[bool, float | None]:
    """Keep grasping until the EEF reaches the demonstrated deep release pose."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    distance = place_release_guard_distance(
        prompt,
        current_eef,
        target_eef,
        enabled=enabled,
    )
    return (distance is not None and distance > tolerance), distance
