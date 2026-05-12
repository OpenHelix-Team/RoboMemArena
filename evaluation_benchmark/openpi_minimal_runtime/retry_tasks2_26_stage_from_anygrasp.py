import json
import logging
import os
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

import eval_common as ec

try:
    from robosuite.utils.errors import RandomizationError as RobosuiteRandomizationError
except Exception:
    RobosuiteRandomizationError = None


OUT_ROOT = Path(os.environ["OUT_ROOT"])
VIDEO_DIR = Path(os.environ["VIDEO_DIR"])
SUMMARY_JSON = Path(os.environ["SUMMARY_JSON"])
SUMMARY_TSV = Path(os.environ["SUMMARY_TSV"])
PROMPT_TRACE_TSV = Path(os.environ.get("PROMPT_TRACE_TSV", str(OUT_ROOT / "prompt_trace.tsv")))

DEFAULT_TASKS = list(range(2, 27))


def _parse_tasks() -> list[int]:
    raw = os.environ.get("TASKS_JSON")
    if not raw:
        return DEFAULT_TASKS
    val = json.loads(raw)
    return [int(x) for x in val]


TASKS = _parse_tasks()
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "1"))
MEM_POLICY = os.environ.get("MEM_POLICY", "").strip().lower() in {"1", "true", "yes", "y", "on"}
MEM_OBS_STEPS = int(os.environ.get("MEM_OBS_STEPS", "4"))
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
RESIZE_SIZE = int(os.environ.get("RESIZE_SIZE", "256"))
REPLAN_STEPS = int(os.environ.get("REPLAN_STEPS", "5"))
NUM_STEPS_WAIT = int(os.environ.get("NUM_STEPS_WAIT", "10"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "2000"))
SEED = int(os.environ.get("SEED", "100"))
ENV_INIT_RETRIES = int(os.environ.get("ENV_INIT_RETRIES", "30"))
ENV_INIT_RETRY_SLEEP = float(os.environ.get("ENV_INIT_RETRY_SLEEP", "0.05"))
PROMPT_MODE = os.environ.get("PROMPT_MODE", "fixed")
PROMPT_POOL_NAME = os.environ.get("PROMPT_POOL_NAME")


@dataclass
class StageSpec:
    name: str
    check_fn: Callable[[Any, dict[str, Any], int], bool]


def _is_randomization_error(exc: Exception) -> bool:
    if RobosuiteRandomizationError is not None and isinstance(exc, RobosuiteRandomizationError):
        return True
    return type(exc).__name__ == "RandomizationError"


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
    fallback_x_thresh: float = 0.65,
) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        angle = _microwave_joint_angle(env)
        if angle is not None:
            return abs(angle) > joint_thresh
        handle_pos = _calc_microwave_handle_pos(env)
        if handle_pos is None:
            return False
        return float(handle_pos[0]) < fallback_x_thresh

    return check


def _microwave_closed(dist_thresh: float = 0.05) -> Callable[[Any, dict[str, Any], int], bool]:
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _calc_microwave_handle_pos(env)
        init = state.get("initial_microwave_handle_pos")
        if cur is not None and init is not None:
            return float(np.linalg.norm(cur - init)) < dist_thresh
        angle = _microwave_joint_angle(env)
        if angle is None:
            return False
        return abs(angle) < 0.15

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


def _task_specs(task_id: int) -> list[StageSpec]:
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
        return [
            StageSpec("01_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("02_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("03_Place_Bowl_Drainer", _in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ]
    if task_id == 7:
        return [
            StageSpec("01_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("02_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("03_Place_Bowl_Drainer", _in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ]
    if task_id == 8:
        return [
            StageSpec("01_Place_Pudding_Frypan", _in_container_body("chocolate_pudding_1", "frypan_1", 0.10, -0.05, 0.15)),
            StageSpec("02_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("03_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("04_Place_Bowl_Drainer", _in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ]
    if task_id == 9:
        return [
            StageSpec("01_Place_Butter_Frypan", _in_container_body("butter_1", "frypan_1", 0.10, -0.05, 0.15)),
            StageSpec("02_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("03_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("04_Place_Bowl_Drainer", _in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ]
    if task_id == 10:
        return [
            StageSpec("01_Pour_One", _pour_stage(0.78, 20, hold_angle=1.05, hold_frames=10)),
            StageSpec("02_Pour_Two", _pour_stage(0.78, 20, hold_angle=1.05, hold_frames=10)),
            StageSpec("03_Place_Wine_On_Table", _table_return("wine_bottle_1", 0.35)),
        ]
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
        return [
            StageSpec("01_Place_Butter_Frypan", _in_container_body("butter_1", "frypan_1", 0.12, -0.05, 0.15)),
            StageSpec("02_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("03_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("04_Place_Milk_Table", _table_return("milk_1", 0.40)),
        ]
    if task_id == 16:
        return [
            StageSpec("01_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("02_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("03_Place_Bowl_Drainer", _in_container_body("milk_1", "bowl_drainer_1", 0.15, -0.05, 0.20)),
        ]
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
            StageSpec("04_Close_Microwave", _microwave_closed(0.05)),
        ]
    if task_id == 21:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Butter_Microwave", _in_microwave("butter_1")),
            StageSpec("03_Place_Chocolate_Microwave", _in_microwave("chocolate_pudding_1")),
            StageSpec("04_Close_Microwave", _microwave_closed(0.05)),
        ]
    if task_id == 22:
        return [
            StageSpec("01_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("02_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("03_Place_Tomato_Aside", _near_fixed_position("tomato_sauce_1", np.array([0.0, -0.2, 0.50], dtype=np.float32), 0.20, 0.20)),
            StageSpec("04_Open_Microwave", _microwave_open(0.30)),
            StageSpec("05_Place_Cookies_Microwave", _in_microwave("cookies_1")),
            StageSpec("06_Close_Microwave", _microwave_closed(0.05)),
        ]
    if task_id == 23:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Cream_Microwave", _in_microwave("cream_cheese_1")),
            StageSpec("03_Place_Popcorn_Microwave", _in_microwave("popcorn_1")),
            StageSpec("04_Close_Microwave", _microwave_closed(0.05)),
        ]
    if task_id == 24:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Cookies_Microwave", _in_microwave("cookies_1")),
            StageSpec("03_Place_Popcorn_Microwave", _in_microwave("popcorn_1")),
            StageSpec("04_Close_Microwave", _microwave_closed(0.05)),
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
    if task_id in {10, 15, 18, 19}:
        #  goal 
        return lambda env, stage_done: all(stage_done.values())
    if task_id in {6, 7, 8, 9, 16}:
        #  +  bowl drainer，goal 
        place_bowl_drainer = _in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)
        return lambda env, stage_done: place_bowl_drainer(env, {}, 0)
    return None


def _ensure_prompt_trace_tsv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(
        "task_id\ttrial\tseed\tchosen_prompt\tgoal_success\tstage_pct\n",
        encoding="utf-8",
    )


def _append_prompt_trace_row(
    path: Path,
    *,
    task_id: int,
    trial: int,
    seed: int,
    chosen_prompt: str,
    goal_success: bool,
    stage_pct: float,
) -> None:
    _ensure_prompt_trace_tsv(path)
    with path.open("a", encoding="utf-8") as f:
        safe_prompt = chosen_prompt.replace("\t", " ").replace("\n", " ")
        f.write(
            f"{task_id}\t{trial}\t{seed}\t{safe_prompt}\t"
            f"{int(goal_success)}\t{stage_pct:.1f}\n"
        )


def run_episode_with_stateful_stages(
    env: Any,
    client: Any,
    prompt: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    stage_specs: list[StageSpec],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override: Callable[[Any, dict[str, bool]], bool] | None,
) -> tuple[float, dict[str, bool], bool, list[np.ndarray], list[np.ndarray]]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    all_stages_logged = False
    t = 0
    state: dict[str, Any] | None = None
    current_stage_start = 0
    goal_success = False
    build_element = ec.build_policy_input_builder(
        resize_size=resize_size,
        prompt=prompt,
        mem_policy=MEM_POLICY,
        mem_obs_steps=MEM_OBS_STEPS,
    )

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if state is None:
                state = _build_initial_state(env)
                current_stage_start = state["step_idx"]

            element, rgb, wrist_rgb = build_element(obs)
            replay.append(rgb)
            if wrist_rgb is not None:
                replay_wrist.append(wrist_rgb)

            if not action_plan:
                out = client.infer(element)
                actions = np.asarray(out["actions"])
                action_plan.extend(actions[:replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            _update_state(obs, state)

            if stage_idx < len(stage_specs):
                spec = stage_specs[stage_idx]
                if spec.check_fn(env, state, current_stage_start):
                    stage_done[spec.name] = True
                    logging.info(f"  [t={t}] : {spec.name}")
                    stage_idx += 1
                    current_stage_start = state["step_idx"]

            if stage_idx >= len(stage_specs) and not all_stages_logged:
                logging.info(f"  [t={t}] !")
                all_stages_logged = True

            if goal_check_override is not None:
                goal_success = goal_check_override(env, stage_done)
            else:
                goal_success = ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False
            if goal_success:
                logging.info(f"  [t={t}] BDDL goal ，!")
                break

            if done:
                break
            t += 1
    except Exception as exc:
        logging.exception(f"Episode failed: {exc}")

    num_done = sum(1 for ok in stage_done.values() if ok)
    score = 100.0 * num_done / max(1, len(stage_specs))
    if not goal_success:
        if goal_check_override is not None:
            goal_success = goal_check_override(env, stage_done)
        else:
            goal_success = ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False
    return score, stage_done, goal_success, replay, replay_wrist


def run_eval_task(
    task_id: int,
    num_trials_per_task: int,
    host: str,
    port: int,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    video_out_path: str,
    seed: int,
    prompt_mode: str = "fixed",
    prompt_pool_name: str | None = None,
    prompt_trace_tsv: str | None = None,
) -> None:
    tid, task_key = ec._resolve_task_id(task_id)
    bddl_path = ec._resolve_bddl_path(task_id)
    video_dir = Path(video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)
    prompt_trace_path = Path(prompt_trace_tsv) if prompt_trace_tsv else None

    logging.info(f"Using BDDL: {bddl_path}")
    if prompt_mode == "fixed":
        logging.info(f"Prompt mode: fixed ({ec.resolve_prompt(task_key, bddl_path.stem)})")
    else:
        logging.info(f"Prompt mode: {prompt_mode} (pool={prompt_pool_name or 'task2_4promptmix'})")
    logging.info(f"Video output: {video_dir}")

    OffScreenRenderEnv = ec._get_env_class()
    env = None
    last_exc: Exception | None = None
    for attempt in range(1, ENV_INIT_RETRIES + 1):
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_path),
                camera_heights=256,
                camera_widths=256,
                ignore_done=True,
                reward_shaping=True,
                control_freq=20,
                initialization_noise=None,
            )
            if attempt > 1:
                logging.info(f"Env init recovered at attempt {attempt}/{ENV_INIT_RETRIES}")
            break
        except Exception as exc:
            if not _is_randomization_error(exc):
                raise
            last_exc = exc
            logging.warning(
                f"Env init randomization failed ({attempt}/{ENV_INIT_RETRIES}): {exc}"
            )
            if attempt < ENV_INIT_RETRIES:
                time.sleep(ENV_INIT_RETRY_SLEEP)

    if env is None:
        raise RuntimeError(
            f"Env init failed after {ENV_INIT_RETRIES} retries (last error: {last_exc})"
        )

    client = ec._websocket_client_policy.WebsocketClientPolicy(host, port)

    stage_specs = _task_specs(task_id)
    goal_monitor_dict = ec._build_goal_monitor_dict(bddl_path)
    goal_check_override = _goal_override_check(task_id)

    total_score = 0.0
    stage_totals = {spec.name: 0 for spec in stage_specs}
    goal_succ_cnt = 0

    for ep in range(num_trials_per_task):
        current_seed = seed + ep
        np.random.seed(current_seed)
        try:
            env.seed(current_seed)
        except AttributeError:
            pass

        chosen_prompt = ec.resolve_prompt(
            task_key,
            bddl_path.stem,
            prompt_mode=prompt_mode,
            prompt_pool_name=prompt_pool_name,
            rng_seed=current_seed,
        )
        logging.info(f"Episode {ep} (seed={current_seed}) prompt: {chosen_prompt}")

        score, stage_done, goal_success, replay, replay_wrist = run_episode_with_stateful_stages(
            env=env,
            client=client,
            prompt=chosen_prompt,
            resize_size=resize_size,
            replan_steps=replan_steps,
            num_steps_wait=num_steps_wait,
            max_steps=max_steps,
            stage_specs=stage_specs,
            goal_monitor_dict=goal_monitor_dict,
            goal_check_override=goal_check_override,
        )
        total_score += score
        for name, ok in stage_done.items():
            stage_totals[name] += int(ok)
        goal_succ_cnt += int(goal_success)

        base_name = ec.get_video_basename(task_id, ep, current_seed, goal_success)
        if replay:
            imageio.mimwrite(video_dir / f"{base_name}.mp4", replay, fps=10)
        if replay_wrist:
            imageio.mimwrite(video_dir / f"{base_name}_wrist.mp4", replay_wrist, fps=10)

        stages_str = " | ".join(f"{n}={'Y' if stage_done[n] else 'N'}" for n in stage_done)
        logging.info(
            f"Episode {ep} (seed={current_seed}): score={score:.0f}% | prompt={chosen_prompt} | {stages_str} | goal={'Y' if goal_success else 'N'}"
        )
        if prompt_trace_path is not None:
            _append_prompt_trace_row(
                prompt_trace_path,
                task_id=task_id,
                trial=ep,
                seed=current_seed,
                chosen_prompt=chosen_prompt,
                goal_success=goal_success,
                stage_pct=score,
            )

    env.close()

    n = num_trials_per_task
    avg_score = total_score / max(1, n)
    logging.info("============================================================")
    logging.info(f" - :  = {avg_score:.1f}%")
    for name, cnt in stage_totals.items():
        logging.info(f"  {name}: {cnt}/{n} ({(cnt / max(1, n)) * 100:.0f}%)")
    if goal_monitor_dict:
        goal_pct = 100.0 * goal_succ_cnt / max(1, n)
        logging.info(f" - BDDL goal : {goal_succ_cnt}/{n} ({goal_pct:.1f}%)")
    logging.info(f": {video_dir}")
    logging.info("============================================================")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    _patch_env_resolution()

    results = []
    with SUMMARY_TSV.open("w", encoding="utf-8") as f:
        f.write("task_id\tstatus\terror\tvideo_dir\tnum_stage_checks\tduration_sec\n")

    for task_id in TASKS:
        task_video = VIDEO_DIR / f"task{task_id}"
        task_video.mkdir(parents=True, exist_ok=True)
        st = time.time()
        status = "completed"
        err = ""
        stage_specs = _task_specs(task_id)
        stage_num = len(stage_specs)
        print(f"[INFO] task={task_id} stage_checks={stage_num}")

        try:
            run_eval_task(
                task_id=task_id,
                num_trials_per_task=NUM_TRIALS,
                host=HOST,
                port=PORT,
                resize_size=RESIZE_SIZE,
                replan_steps=REPLAN_STEPS,
                num_steps_wait=NUM_STEPS_WAIT,
                max_steps=MAX_STEPS,
                video_out_path=str(task_video),
                seed=SEED,
                prompt_mode=PROMPT_MODE,
                prompt_pool_name=PROMPT_POOL_NAME,
                prompt_trace_tsv=str(PROMPT_TRACE_TSV),
            )
        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        dur = round(time.time() - st, 2)
        row = {
            "task_id": task_id,
            "status": status,
            "error": err,
            "video_dir": str(task_video),
            "num_stage_checks": stage_num,
            "duration_sec": dur,
        }
        results.append(row)
        with SUMMARY_TSV.open("a", encoding="utf-8") as f:
            f.write(
                f"{task_id}\t{status}\t{err.replace(chr(9), ' ')}\t{task_video}\t{stage_num}\t{dur}\n"
            )
        SUMMARY_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] done")
    print(f"[INFO] summary_json={SUMMARY_JSON}")
    print(f"[INFO] summary_tsv={SUMMARY_TSV}")
    print(f"[INFO] prompt_trace_tsv={PROMPT_TRACE_TSV}")


if __name__ == "__main__":
    main()
