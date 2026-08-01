from __future__ import annotations

from typing import Any


MICROWAVE_JOINT_CANDIDATES = (
    "microwave_1_microjoint",
    "microwave_1_door_joint",
    "microwave_1_hinge",
    "microwave_1_door_hinge",
    "microwave_1_root_joint",
)


def _sim_from_env(env: Any) -> Any | None:
    sim = getattr(env, "sim", None)
    if sim is not None:
        return sim
    wrapped = getattr(env, "env", None)
    return getattr(wrapped, "sim", None) if wrapped is not None else None


def microwave_joint_angle(env: Any) -> float | None:
    """Read the microwave door joint using the pinned remote scorer's lookup order."""
    sim = _sim_from_env(env)
    if sim is None:
        return None
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None

    joint_names = [str(name) for name in getattr(model, "joint_names", [])]
    candidates = list(MICROWAVE_JOINT_CANDIDATES)
    candidates.extend(
        name
        for name in joint_names
        if "microwave" in name.lower() and "door" in name.lower()
    )
    for name in candidates:
        if name not in joint_names:
            continue
        joint_id = int(model.joint_name2id(name))
        qpos_address = int(model.jnt_qposadr[joint_id])
        return float(data.qpos[qpos_address])
    return None
