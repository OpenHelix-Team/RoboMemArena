from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

import eval_common as ec


# Shared Task1-26 stage/goal checker used by both the adapter benchmark
# and the VLM/VLA reference path. Keep this in sync with
# evaluation_benchmark/openpi_minimal_runtime/retry_tasks2_26_stage_from_anygrasp.py.

@dataclass
class StageSpec:
    name: str
    check_fn: Callable[[Any, dict[str, Any], int], bool]

DRAWER_TASK_OPTIONAL_FINAL_STAGE = {
    4: "09_Close_Top_Drawer_Final",
    5: "09_Close_Middle_Drawer_Final",
    11: "06_Close_Middle_Drawer",
    12: "04_Close_Middle_Drawer",
    13: "04_Close_Middle_Drawer",
    14: "06_Close_Middle_Drawer",
    17: "04_Close_Middle_Drawer",
}

TASK22_OPTIONAL_FINAL_STAGE = "07_Close_Microwave"


def _is_drawer_task(task_id: int | None) -> bool:
    return task_id in DRAWER_TASK_OPTIONAL_FINAL_STAGE


def _optional_final_stage_name(task_id: int | None) -> str | None:
    if task_id == 22:
        return TASK22_OPTIONAL_FINAL_STAGE
    return DRAWER_TASK_OPTIONAL_FINAL_STAGE.get(task_id)


def _counted_stage_names(task_id: int | None, stage_done: dict[str, bool]) -> list[str]:
    optional_name = _optional_final_stage_name(task_id)
    return [name for name in stage_done if name != optional_name]


def _stage_success_from_stage_done(task_id: int | None, stage_done: dict[str, bool]) -> bool:
    counted_names = _counted_stage_names(task_id, stage_done)
    return bool(counted_names) and all(stage_done.get(name, False) for name in counted_names)


def _stage_score_pct(task_id: int | None, stage_done: dict[str, bool]) -> float:
    counted_names = _counted_stage_names(task_id, stage_done)
    num_done = sum(1 for name in counted_names if stage_done.get(name, False))
    return 100.0 * num_done / max(1, len(counted_names))


def _patch_env_resolution() -> None:
    base_env = ec._get_env_class()
    orig_init = base_env.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["camera_heights"] = 480
        kwargs["camera_widths"] = 640
        return orig_init(self, *args, **kwargs)

    base_env.__init__ = patched_init
    ec._get_env_class = lambda: base_env

def _name_variants(name: str) -> list[str]:
    out = [name]
    if not name.endswith("_main"):
        out.append(f"{name}_main")
    if name.endswith("_main"):
        out.append(name[:-5])
    return out

def _current_body_pos(env: Any, name: str) -> np.ndarray | None:
    return ec._body_pos(env, name)

def _current_site_pos(env: Any, name: str) -> np.ndarray | None:
    for cand in _name_variants(name):
        try:
            sid = env.sim.model.site_name2id(cand)
            return np.asarray(env.sim.data.site_xpos[sid], dtype=np.float32).copy()
        except Exception:
            continue
    return None

def _current_target_center(env: Any, target_kind: str, target_name: str) -> np.ndarray | None:
    if target_kind == "body":
        return _current_body_pos(env, target_name)
    if target_kind == "site":
        return _current_site_pos(env, target_name)
    raise ValueError(f"Unsupported target_kind: {target_kind}")

def _initial_body_pos(state: dict[str, Any], name: str) -> np.ndarray | None:
    for cand in _name_variants(name):
        if cand in state["initial_body_pos"]:
            return state["initial_body_pos"][cand]
    return None

def _initial_site_pos(state: dict[str, Any], name: str) -> np.ndarray | None:
    for cand in _name_variants(name):
        if cand in state["initial_site_pos"]:
            return state["initial_site_pos"][cand]
    return None

def _body_geom_center(env: Any, body_name: str) -> np.ndarray | None:
    bid = ec._resolve_body_id(env, body_name)
    if bid is None:
        return None
    geom_start = int(env.sim.model.body_geomadr[bid])
    geom_num = int(env.sim.model.body_geomnum[bid])
    if geom_num <= 0:
        return _current_body_pos(env, body_name)
    acc = np.zeros(3, dtype=np.float32)
    for i in range(geom_num):
        acc += np.asarray(env.sim.data.geom_xpos[geom_start + i], dtype=np.float32)
    return acc / float(geom_num)

def _drawer_handle_pos(env: Any, drawer: str) -> np.ndarray | None:
    return _current_body_pos(env, f"wooden_cabinet_1_{drawer}_handle")

def _microwave_anchor_pose(env: Any) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    site_names = [str(x) for x in env.sim.model.site_names]
    for site_name in ("microwave_1_heating_region", "microwave_1_top_side"):
        if site_name in site_names:
            sid = env.sim.model.site_name2id(site_name)
            pos = np.asarray(env.sim.data.site_xpos[sid], dtype=np.float32).copy()
            mat = np.asarray(env.sim.data.site_xmat[sid], dtype=np.float32).reshape(3, 3).copy()
            return pos, mat
    bid = ec._resolve_body_id(env, "microwave_1")
    if bid is None:
        return None, None
    pos = np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32).copy()
    mat = np.asarray(env.sim.data.body_xmat[bid], dtype=np.float32).reshape(3, 3).copy()
    return pos, mat

def _calc_microwave_handle_pos(env: Any) -> np.ndarray | None:
    site_pos, site_mat = _microwave_anchor_pose(env)
    if site_pos is None or site_mat is None:
        return None
    right_dir = site_mat @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    front_dir = site_mat @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    handle_pos = site_pos.copy()
    handle_pos += right_dir * 0.15
    handle_pos += front_dir * 0.05
    handle_pos[2] += 0.03
    return handle_pos.astype(np.float32)

def _microwave_joint_angle(env: Any) -> float | None:
    candidates = [
        "microwave_1_microjoint",
        "microwave_1_door_joint",
        "microwave_1_hinge",
        "microwave_1_door_hinge",
        "microwave_1_root_joint",
    ]
    joint_names = [str(x) for x in env.sim.model.joint_names]
    for name in candidates:
        if name in joint_names:
            jid = env.sim.model.joint_name2id(name)
            adr = int(env.sim.model.jnt_qposadr[jid])
            return float(env.sim.data.qpos[adr])
    for name in joint_names:
        low = name.lower()
        if "microwave" in low and "door" in low:
            jid = env.sim.model.joint_name2id(name)
            adr = int(env.sim.model.jnt_qposadr[jid])
            return float(env.sim.data.qpos[adr])
    return None

def _tilt_from_quat(quat: np.ndarray) -> float:
    z_axis = R.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()[:, 2]
    return float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))

def _body_tilt_angle(env: Any, name: str) -> float | None:
    for cand in _name_variants(name):
        try:
            bid = env.sim.model.body_name2id(cand)
            mat = np.asarray(env.sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
            z_axis = mat[:, 2]
            return float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))
        except Exception:
            continue
    return None

def _body_axis_tilts(env: Any, name: str) -> tuple[float, float, float] | None:
    for cand in _name_variants(name):
        try:
            bid = env.sim.model.body_name2id(cand)
            mat = np.asarray(env.sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
            return tuple(float(np.arccos(np.clip(mat[:, i][2], -1.0, 1.0))) for i in range(3))
        except Exception:
            continue
    return None

def _append_task6_tilt_debug(
    *,
    step_idx: int,
    stage_start: int,
    stage_len: int,
    tilt: float,
    baseline: float | None,
    deviation: float | None,
    max_deviation: float | None,
    moved: bool,
    returned: bool,
    axis_tilts: tuple[float, float, float] | None,
) -> None:
    path = os.environ.get("TASK6_TILT_TRACE_CSV")
    if not path:
        return
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not trace_path.exists()
    ax = axis_tilts if axis_tilts is not None else (np.nan, np.nan, np.nan)
    with trace_path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "step_idx,stage_start,stage_len,tilt_z,baseline_z,deviation_z,"
                "max_deviation_z,moved,returned,axis_x_tilt,axis_y_tilt,axis_z_tilt\n"
            )
        f.write(
            f"{step_idx},{stage_start},{stage_len},{tilt:.8f},"
            f"{baseline if baseline is not None else np.nan:.8f},"
            f"{deviation if deviation is not None else np.nan:.8f},"
            f"{max_deviation if max_deviation is not None else np.nan:.8f},"
            f"{int(moved)},{int(returned)},"
            f"{ax[0]:.8f},{ax[1]:.8f},{ax[2]:.8f}\n"
        )

def _build_initial_state(env: Any) -> dict[str, Any]:
    body_names = [str(x) for x in env.sim.model.body_names]
    site_names = [str(x) for x in env.sim.model.site_names]
    joint_names = [str(x) for x in env.sim.model.joint_names]
    initial_body_pos = {
        name: np.asarray(env.sim.data.body_xpos[i], dtype=np.float32).copy()
        for i, name in enumerate(body_names)
    }
    initial_site_pos = {
        name: np.asarray(env.sim.data.site_xpos[i], dtype=np.float32).copy()
        for i, name in enumerate(site_names)
    }
    initial_joint_qpos = {}
    for i, name in enumerate(joint_names):
        try:
            adr = int(env.sim.model.jnt_qposadr[i])
            initial_joint_qpos[name] = float(env.sim.data.qpos[adr])
        except Exception:
            continue
    return {
        "step_idx": 0,
        "tilt_angles": [],
        "initial_body_pos": initial_body_pos,
        "initial_site_pos": initial_site_pos,
        "initial_joint_qpos": initial_joint_qpos,
        "initial_microwave_handle_pos": _calc_microwave_handle_pos(env),
        "last_obs": None,
    }

def _update_state(obs: Any, state: dict[str, Any]) -> None:
    quat = None
    if isinstance(obs, dict):
        quat = obs.get("robot0_eef_quat")
    if quat is not None:
        state["tilt_angles"].append(_tilt_from_quat(quat))
        state["step_idx"] = len(state["tilt_angles"])
    state["last_obs"] = obs

def _segment_tilts(state: dict[str, Any], stage_start: int) -> np.ndarray:
    vals = state["tilt_angles"][stage_start:]
    if not vals:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(vals, dtype=np.float32)

def _lift_abs(obj_name: str, z_thresh: float) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        pos = _current_body_pos(env, obj_name)
        return pos is not None and float(pos[2]) > z_thresh

    return check

def _lift_rel(
    obj_name: str,
    delta: float,
    plate1_max_rise: float | None = None,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        pos = _current_body_pos(env, obj_name)
        init_pos = _initial_body_pos(state, obj_name)
        if pos is None or init_pos is None:
            return False
        if float(pos[2] - init_pos[2]) <= delta:
            return False
        if plate1_max_rise is not None:
            plate_pos = _current_body_pos(env, "plate_1")
            plate_init = _initial_body_pos(state, "plate_1")
            if plate_pos is None or plate_init is None:
                return False
            if float(plate_pos[2] - plate_init[2]) > plate1_max_rise:
                return False
        return True

    return check

def _in_container_body(
    obj_name: str,
    target_name: str,
    xy_thresh: float,
    z_low: float,
    z_high: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        obj_pos = _current_body_pos(env, obj_name)
        tgt_pos = _current_body_pos(env, target_name)
        if obj_pos is None or tgt_pos is None:
            return False
        xy_dist = float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2]))
        z_delta = float(obj_pos[2] - tgt_pos[2])
        return xy_dist < xy_thresh and z_low < z_delta < z_high

    return check

def _in_container_site(
    obj_name: str,
    site_name: str,
    x_thresh: float,
    y_thresh: float,
    z_low: float,
    z_high: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        obj_pos = _current_body_pos(env, obj_name)
        site_pos = _current_site_pos(env, site_name)
        if obj_pos is None or site_pos is None:
            return False
        x_diff = abs(float(obj_pos[0] - site_pos[0]))
        y_diff = abs(float(obj_pos[1] - site_pos[1]))
        z_diff = float(obj_pos[2] - site_pos[2])
        return x_diff < x_thresh and y_diff < y_thresh and z_low < z_diff < z_high

    return check

def _in_drawer_radius(
    obj_name: str,
    region_name: str,
    horizontal_thresh: float,
    z_thresh: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        obj_pos = _current_body_pos(env, obj_name)
        region_pos = _current_site_pos(env, region_name)
        if obj_pos is None or region_pos is None:
            return False
        horizontal_dist = float(np.linalg.norm(obj_pos[:2] - region_pos[:2]))
        height_diff = abs(float(obj_pos[2] - region_pos[2]))
        return horizontal_dist < horizontal_thresh and height_diff < z_thresh

    return check

def _in_drawer_y_window(
    obj_name: str,
    region_name: str,
    x_thresh: float,
    y_low_offset: float,
    y_high_offset: float,
    z_thresh: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        obj_pos = _current_body_pos(env, obj_name)
        region_pos = _current_site_pos(env, region_name)
        if obj_pos is None or region_pos is None:
            return False
        in_x = abs(float(obj_pos[0] - region_pos[0])) < x_thresh
        in_y = float(region_pos[1] + y_low_offset) < float(obj_pos[1]) < float(region_pos[1] + y_high_offset)
        in_z = abs(float(obj_pos[2] - region_pos[2])) < z_thresh
        return in_x and in_y and in_z

    return check

def _drawer_open_handle(drawer: str, threshold: float) -> Callable[[Any, dict[str, Any], int], bool]:
    handle_name = f"wooden_cabinet_1_{drawer}_handle"

    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _drawer_handle_pos(env, drawer)
        init = _initial_body_pos(state, handle_name)
        if cur is None or init is None:
            return False
        return float(np.linalg.norm(cur - init)) >= threshold

    return check

def _drawer_closed_handle(drawer: str, threshold: float) -> Callable[[Any, dict[str, Any], int], bool]:
    handle_name = f"wooden_cabinet_1_{drawer}_handle"

    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _drawer_handle_pos(env, drawer)
        init = _initial_body_pos(state, handle_name)
        if cur is None or init is None:
            return False
        return float(np.linalg.norm(cur - init)) <= threshold

    return check

def _drawer_open_pull(
    region_name: str,
    closed_y: float,
    threshold: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        region_pos = _current_site_pos(env, region_name)
        if region_pos is None:
            return False
        pull_distance = closed_y - float(region_pos[1])
        return pull_distance > threshold

    return check

def _drawer_closed_pull(
    region_name: str,
    closed_y: float,
    threshold: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        region_pos = _current_site_pos(env, region_name)
        if region_pos is None:
            return False
        pull_distance = closed_y - float(region_pos[1])
        return pull_distance < threshold

    return check

def _drawer_open_abs(
    region_name: str,
    initial_y: float | None,
    threshold: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        region_pos = _current_site_pos(env, region_name)
        init_pos = _initial_site_pos(state, region_name)
        if region_pos is None:
            return False
        if init_pos is not None:
            ref_y = float(init_pos[1])
        elif initial_y is not None:
            ref_y = float(initial_y)
        else:
            return False
        return abs(float(region_pos[1] - ref_y)) > threshold

    return check

def _drawer_closed_abs(
    region_name: str,
    initial_y: float | None,
    threshold: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        region_pos = _current_site_pos(env, region_name)
        init_pos = _initial_site_pos(state, region_name)
        if region_pos is None:
            return False
        if init_pos is not None:
            ref_y = float(init_pos[1])
        elif initial_y is not None:
            ref_y = float(initial_y)
        else:
            return False
        return abs(float(region_pos[1] - ref_y)) < threshold

    return check

def _microwave_open(
    joint_thresh: float,
    fallback_dist_thresh: float = 0.08,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        angle = _microwave_joint_angle(env)
        if angle is not None:
            return abs(angle) > joint_thresh
        cur = _calc_microwave_handle_pos(env)
        init = state.get("initial_microwave_handle_pos")
        if cur is None or init is None:
            return False
        return float(np.linalg.norm(cur - init)) > fallback_dist_thresh

    return check

def _microwave_closed(dist_thresh: float = 0.05) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        angle = _microwave_joint_angle(env)
        if angle is not None:
            return abs(angle) < 0.15
        cur = _calc_microwave_handle_pos(env)
        init = state.get("initial_microwave_handle_pos")
        if cur is not None and init is not None:
            return float(np.linalg.norm(cur - init)) < dist_thresh
        return False

    return check

def _in_microwave(obj_name: str, xy_thresh: float = 0.20) -> Callable[[Any, dict[str, Any], int], bool]:
    return _in_container_site(obj_name, "microwave_1_heating_region", xy_thresh, xy_thresh, -1.0, 1.0)

def _cabinet2(obj_name: str, xy_thresh: float, z_low: float, z_high: float) -> Callable[[Any, dict[str, Any], int], bool]:
    return _in_container_body(obj_name, "wooden_cabinet_2", xy_thresh, z_low, z_high)

def _on_plate(obj_name: str, plate_name: str = "plate_2") -> Callable[[Any, dict[str, Any], int], bool]:
    return _in_container_body(obj_name, plate_name, 0.06, 0.01, 0.10)

def _table_return(obj_name: str, radius: float) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _current_body_pos(env, obj_name)
        init = _initial_body_pos(state, obj_name)
        if cur is None or init is None:
            return False
        distance = float(np.linalg.norm(cur - init))
        return distance < radius and 0.0 < float(cur[2]) < 0.80

    return check

def _near_fixed_position(
    obj_name: str,
    target: np.ndarray,
    xy_thresh: float,
    z_thresh: float,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _current_body_pos(env, obj_name)
        if cur is None:
            return False
        xy_dist = float(np.linalg.norm(cur[:2] - target[:2]))
        z_diff = abs(float(cur[2] - target[2]))
        return xy_dist < xy_thresh and z_diff < z_thresh

    return check

def _pour_stage(
    range_thresh: float,
    min_steps: int,
    hold_angle: float | None = None,
    hold_frames: int | None = None,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        tilts = _segment_tilts(state, stage_start)
        if len(tilts) < min_steps:
            return False
        tilt_range = float(tilts.max() - tilts.min())
        if tilt_range <= range_thresh:
            return False
        if hold_angle is not None and hold_frames is not None:
            return int(np.sum(tilts > hold_angle)) > hold_frames
        return True

    return check

def _body_pour_stage(
    obj_name: str,
    target_kind: str,
    target_name: str,
    target_radius: float = 0.20,
    move_thresh: float = 0.15,
    return_thresh: float = 0.10,
    min_steps: int = 10,
    warmup: int = 5,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        tilt = _body_tilt_angle(env, obj_name)
        src_pos = _current_body_pos(env, obj_name)
        tgt_pos = _current_target_center(env, target_kind, target_name)
        if tilt is None or src_pos is None or tgt_pos is None:
            return False
        axis_tilts = _body_axis_tilts(env, obj_name)
        step_idx = int(state.get("step_idx", len(state.get("tilt_angles", []))))
        records = state.setdefault(f"pour_tilt_records_{obj_name}", [])
        in_target = float(np.linalg.norm(src_pos[:2] - tgt_pos[:2])) <= float(target_radius)
        if not records or int(records[-1][0]) != step_idx:
            records.append((step_idx, float(tilt), bool(in_target)))
        stage_records = [(step, val, hit) for step, val, hit in records if int(step) >= int(stage_start)]
        vals = np.asarray([val for _, val, _ in stage_records], dtype=np.float32)
        in_target_flags = np.asarray([bool(hit) for _, _, hit in stage_records], dtype=bool)
        finite_mask = np.isfinite(vals)
        vals = vals[finite_mask]
        in_target_flags = in_target_flags[finite_mask]
        if len(vals) < min_steps:
            _append_task6_tilt_debug(
                step_idx=step_idx,
                stage_start=stage_start,
                stage_len=len(vals),
                tilt=float(tilt),
                baseline=None,
                deviation=None,
                max_deviation=None,
                moved=False,
                returned=False,
                axis_tilts=axis_tilts,
            )
            return False
        baseline = float(np.median(vals[: max(1, min(warmup, len(vals)))]))
        deviations = np.abs(vals - baseline)
        moved = np.where(deviations >= move_thresh)[0]
        max_deviation = float(np.max(deviations)) if len(deviations) else 0.0
        returned = False
        if len(moved) == 0:
            _append_task6_tilt_debug(
                step_idx=step_idx,
                stage_start=stage_start,
                stage_len=len(vals),
                tilt=float(tilt),
                baseline=baseline,
                deviation=float(abs(float(tilt) - baseline)),
                max_deviation=max_deviation,
                moved=False,
                returned=False,
                axis_tilts=axis_tilts,
            )
            return False
        move_idx = int(moved[0])
        return_candidates = np.where(deviations[move_idx + 1 :] <= return_thresh)[0]
        returned = len(return_candidates) > 0
        in_target_during_pour = False
        if returned:
            return_idx = move_idx + 1 + int(return_candidates[0])
            in_target_during_pour = bool(np.any(in_target_flags[move_idx : return_idx + 1]))
        _append_task6_tilt_debug(
            step_idx=step_idx,
            stage_start=stage_start,
            stage_len=len(vals),
            tilt=float(tilt),
            baseline=baseline,
            deviation=float(abs(float(tilt) - baseline)),
            max_deviation=max_deviation,
            moved=True,
            returned=returned and in_target_during_pour,
            axis_tilts=axis_tilts,
        )
        return returned and in_target_during_pour

    return check

def _tomato_body_pour_stage(
    target_kind: str,
    target_name: str,
    target_radius: float = 0.20,
    move_thresh: float = 0.15,
    return_thresh: float = 0.10,
    min_steps: int = 10,
    warmup: int = 5,
) -> Callable[[Any, dict[str, Any], int], bool]:
    return _body_pour_stage("tomato_sauce_1", target_kind, target_name, target_radius, move_thresh, return_thresh, min_steps, warmup)

def _counting_pour_stages(
    obj_name: str,
    label: str,
    target_kind: str,
    target_name: str,
    target_radius: float = 0.20,
    move_thresh: float = 0.15,
    return_thresh: float = 0.10,
) -> list[StageSpec]:
    return [
        StageSpec(f"01_Lift_{label}", _lift_rel(obj_name, 0.03)),
        StageSpec("02_Pour_One", _body_pour_stage(obj_name, target_kind, target_name, target_radius, move_thresh, return_thresh)),
        StageSpec("03_Pour_Two", _body_pour_stage(obj_name, target_kind, target_name, target_radius, move_thresh, return_thresh)),
    ]

COUNTING_POUR_TASKS = {6, 7, 8, 9, 10, 15, 16, 22}


def _counting_pour_target(task_id: int) -> tuple[str, str, str] | None:
    if task_id in {6, 22}:
        return "tomato_sauce_1", "body", "cookies_1"
    if task_id == 7:
        return "tomato_sauce_1", "site", "frypan_1_default_site"
    if task_id == 8:
        return "tomato_sauce_1", "body", "chocolate_pudding_1"
    if task_id == 9:
        return "tomato_sauce_1", "body", "butter_1"
    if task_id == 10:
        return "wine_bottle_1", "site", "white_yellow_mug_1_default_site"
    if task_id == 15:
        return "milk_1", "body", "butter_1"
    if task_id == 16:
        return "milk_1", "site", "red_coffee_mug_1_default_site"
    return None


def _is_counting_pour_task(task_id: int | None) -> bool:
    return task_id in COUNTING_POUR_TASKS


def _extra_pour_check(task_id: int) -> Callable[[Any, dict[str, Any], int], bool] | None:
    target = _counting_pour_target(task_id)
    if target is None:
        return None
    obj_name, target_kind, target_name = target
    return _body_pour_stage(obj_name, target_kind, target_name)


def _task_specs(task_id: int) -> list[StageSpec]:
    if task_id == 1:
        return [
            StageSpec("01_Place_Cookies_Basket", _in_container_body("cookies_1", "basket_1", 0.12, -0.05, 0.20)),
            StageSpec("02_Place_Tomato_Basket", _in_container_body("tomato_sauce_1", "basket_1", 0.12, -0.05, 0.20)),
        ]
    if task_id == 2:
        return [
            StageSpec("01_Place_Butter_Basket", _in_container_body("butter_1", "basket_1", 0.12, -0.05, 0.20)),
            StageSpec("02_Place_Popcorn_Basket", _in_container_body("popcorn_1", "basket_1", 0.12, -0.05, 0.20)),
        ]
    if task_id == 3:
        return [
            StageSpec("01_Place_Cream_Basket", _in_container_body("cream_cheese_1", "basket_1", 0.12, -0.05, 0.20)),
            StageSpec("02_Place_Pudding_Basket", _in_container_body("chocolate_pudding_1", "basket_1", 0.12, -0.05, 0.20)),
        ]
    if task_id == 4:
        return [
            StageSpec("01_Open_Top_Drawer", _drawer_open_abs("wooden_cabinet_1_top_region", None, 0.10)),
            StageSpec("02_Close_Top_Drawer", _drawer_closed_abs("wooden_cabinet_1_top_region", None, 0.08)),
            StageSpec("03_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("04_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
            StageSpec("05_Open_Bottom_Drawer", _drawer_open_abs("wooden_cabinet_1_bottom_region", None, 0.10)),
            StageSpec("06_Close_Bottom_Drawer", _drawer_closed_abs("wooden_cabinet_1_bottom_region", None, 0.08)),
            StageSpec("07_Open_Top_Drawer_Again", _drawer_open_abs("wooden_cabinet_1_top_region", None, 0.10)),
            StageSpec("08_Put_Butter_Top_Drawer", _in_drawer_radius("butter_1", "wooden_cabinet_1_top_region", 0.25, 0.15)),
            StageSpec("09_Close_Top_Drawer_Final", _drawer_closed_abs("wooden_cabinet_1_top_region", None, 0.08)),
        ]
    if task_id == 5:
        return [
            StageSpec("01_Open_Top_Drawer", _drawer_open_abs("wooden_cabinet_1_top_region", None, 0.10)),
            StageSpec("02_Close_Top_Drawer", _drawer_closed_abs("wooden_cabinet_1_top_region", None, 0.08)),
            StageSpec("03_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("04_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
            StageSpec("05_Open_Bottom_Drawer", _drawer_open_abs("wooden_cabinet_1_bottom_region", None, 0.10)),
            StageSpec("06_Close_Bottom_Drawer", _drawer_closed_abs("wooden_cabinet_1_bottom_region", None, 0.08)),
            StageSpec("07_Open_Middle_Drawer_Again", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("08_Put_Butter_Middle_Drawer", _in_drawer_radius("butter_1", "wooden_cabinet_1_middle_region", 0.25, 0.15)),
            StageSpec("09_Close_Middle_Drawer_Final", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
        ]
    if task_id == 6:
        return _counting_pour_stages("tomato_sauce_1", "Tomato_Sauce", "body", "cookies_1")
    if task_id == 7:
        return _counting_pour_stages("tomato_sauce_1", "Tomato_Sauce", "site", "frypan_1_default_site")
    if task_id == 8:
        return _counting_pour_stages("tomato_sauce_1", "Tomato_Sauce", "body", "chocolate_pudding_1")
    if task_id == 9:
        return _counting_pour_stages("tomato_sauce_1", "Tomato_Sauce", "body", "butter_1")
    if task_id == 10:
        return _counting_pour_stages("wine_bottle_1", "Wine_Bottle", "site", "white_yellow_mug_1_default_site")
    if task_id == 11:
        return [
            StageSpec("01_Open_Top_Drawer", _drawer_open_abs("wooden_cabinet_1_top_region", None, 0.10)),
            StageSpec("02_Place_Cookies_Top_Drawer", _in_container_site("cookies_1", "wooden_cabinet_1_top_region", 0.15, 0.15, -0.05, 0.15)),
            StageSpec("03_Close_Top_Drawer", _drawer_closed_abs("wooden_cabinet_1_top_region", None, 0.08)),
            StageSpec("04_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("05_Place_Butter_Middle_Drawer", _in_container_site("butter_1", "wooden_cabinet_1_middle_region", 0.15, 0.15, -0.05, 0.15)),
            StageSpec("06_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
        ]
    if task_id == 12:
        return [
            StageSpec("01_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("02_Place_Cookies_Middle_Drawer", _in_container_site("cookies_1", "wooden_cabinet_1_middle_region", 0.15, 0.15, -0.05, 0.15)),
            StageSpec("03_Place_Chocolate_Middle_Drawer", _in_container_site("chocolate_pudding_1", "wooden_cabinet_1_middle_region", 0.15, 0.15, -0.05, 0.15)),
            StageSpec("04_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
        ]
    if task_id == 13:
        return [
            StageSpec("01_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("02_Place_Cookies_Middle_Drawer", _in_drawer_y_window("cookies_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10)),
            StageSpec("03_Place_Butter_Middle_Drawer", _in_drawer_y_window("butter_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10)),
            StageSpec("04_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
        ]
    if task_id == 14:
        return [
            StageSpec("01_Open_Top_Drawer", _drawer_open_abs("wooden_cabinet_1_top_region", None, 0.10)),
            StageSpec("02_Place_Cookies_Top_Drawer", _in_drawer_y_window("cookies_1", "wooden_cabinet_1_top_region", 0.15, -0.20, 0.10, 0.10)),
            StageSpec("03_Close_Top_Drawer", _drawer_closed_abs("wooden_cabinet_1_top_region", None, 0.08)),
            StageSpec("04_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("05_Place_Chocolate_Middle_Drawer", _in_drawer_y_window("chocolate_pudding_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10)),
            StageSpec("06_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
        ]
    if task_id == 15:
        return _counting_pour_stages("milk_1", "Milk", "body", "butter_1")
    if task_id == 16:
        return _counting_pour_stages("milk_1", "Milk", "site", "red_coffee_mug_1_default_site")
    if task_id == 17:
        return [
            StageSpec("01_Open_Middle_Drawer", _drawer_open_abs("wooden_cabinet_1_middle_region", None, 0.10)),
            StageSpec("02_Place_Butter_Middle_Drawer", _in_drawer_y_window("butter_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10)),
            StageSpec("03_Place_Chocolate_Middle_Drawer", _in_drawer_y_window("chocolate_pudding_1", "wooden_cabinet_1_middle_region", 0.15, -0.20, 0.10, 0.10)),
            StageSpec("04_Close_Middle_Drawer", _drawer_closed_abs("wooden_cabinet_1_middle_region", None, 0.08)),
        ]
    if task_id == 18:
        return [
            StageSpec("01_Place_Chocolate_Cabinet2", _cabinet2("chocolate_pudding_1", 0.15, 0.10, 0.25)),
            StageSpec("02_Place_Butter_Cabinet2", _cabinet2("butter_1", 0.15, 0.10, 0.25)),
        ]
    if task_id == 19:
        return [
            StageSpec("01_Place_Tomato_Sauce_Cabinet2", _cabinet2("tomato_sauce_1", 0.30, 0.10, 0.30)),
            StageSpec("02_Place_Milk_Cabinet2", _cabinet2("milk_1", 0.30, 0.10, 0.30)),
            StageSpec("03_Place_Orange_Juice_Cabinet2", _cabinet2("orange_juice_1", 0.30, 0.10, 0.30)),
        ]
    if task_id == 20:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.30)),
            StageSpec("02_Place_Cookies_Microwave", _in_microwave("cookies_1")),
            StageSpec("03_Place_Chocolate_Microwave", _in_microwave("chocolate_pudding_1")),
        ]
    if task_id == 21:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Butter_Microwave", _in_microwave("butter_1")),
            StageSpec("03_Place_Chocolate_Microwave", _in_microwave("chocolate_pudding_1")),
        ]
    if task_id == 22:
        return [
            *_counting_pour_stages("tomato_sauce_1", "Tomato_Sauce", "body", "cookies_1"),
            StageSpec(
                "04_Place_Tomato_Aside",
                _near_fixed_position(
                    "tomato_sauce_1",
                    np.array([0.0, -0.2, 0.50], dtype=np.float32),
                    0.20,
                    0.20,
                ),
            ),
            StageSpec("05_Open_Microwave", _microwave_open(0.30)),
            StageSpec("06_Place_Cookies_Microwave", _in_microwave("cookies_1")),
            StageSpec(TASK22_OPTIONAL_FINAL_STAGE, _microwave_closed(0.05)),
        ]
    if task_id == 23:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Cream_Microwave", _in_microwave("cream_cheese_1")),
            StageSpec("03_Place_Popcorn_Microwave", _in_microwave("popcorn_1")),
        ]
    if task_id == 24:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Cookies_Microwave", _in_microwave("cookies_1")),
            StageSpec("03_Place_Popcorn_Microwave", _in_microwave("popcorn_1")),
        ]
    if task_id == 25:
        return [
            StageSpec("01_Place_Butter_Plate2", _on_plate("butter_1", "plate_2")),
            StageSpec("02_Place_Cream_Cheese_Plate2", _on_plate("cream_cheese_1", "plate_2")),
        ]
    if task_id == 26:
        return [
            StageSpec("01_Place_Chocolate_Pudding_Plate2", _on_plate("chocolate_pudding_1", "plate_2")),
            StageSpec("02_Place_Cream_Cheese_Plate2", _on_plate("cream_cheese_1", "plate_2")),
        ]
    raise ValueError(f"Unsupported task_id={task_id}")

def _goal_override_check(task_id: int) -> Callable[[Any, dict[str, bool]], bool] | None:
    if task_id == 22:
        return lambda env, stage_done: _stage_success_from_stage_done(task_id, stage_done)
    if task_id in {6, 7, 8, 9, 10, 15, 16, 18, 19}:
        # These tasks treat completing all ordered stages as goal success.
        return lambda env, stage_done: all(stage_done.values())
    return None
