from __future__ import annotations

import argparse
import dataclasses
import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

import eval_common as ec
from policy_adapter import BasePolicyAdapter, build_eval26_policy_input, ensure_action_chunk, load_policy_adapter


@dataclasses.dataclass
class StageSpec:
    name: str
    check_fn: Callable[[Any, dict[str, Any], int], bool]


def _patch_env_resolution() -> None:
    ec.patch_env_resolution(480, 640)


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


def _in_container_body(obj_name: str, target_name: str, xy_thresh: float, z_low: float, z_high: float):
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        obj_pos = _current_body_pos(env, obj_name)
        tgt_pos = _current_body_pos(env, target_name)
        if obj_pos is None or tgt_pos is None:
            return False
        xy_dist = float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2]))
        z_delta = float(obj_pos[2] - tgt_pos[2])
        return xy_dist < xy_thresh and z_low < z_delta < z_high

    return check


def _in_container_site(obj_name: str, site_name: str, x_thresh: float, y_thresh: float, z_low: float, z_high: float):
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


def _in_drawer_radius(obj_name: str, region_name: str, horizontal_thresh: float, z_thresh: float):
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        obj_pos = _current_body_pos(env, obj_name)
        region_pos = _current_site_pos(env, region_name)
        if obj_pos is None or region_pos is None:
            return False
        horizontal_dist = float(np.linalg.norm(obj_pos[:2] - region_pos[:2]))
        height_diff = abs(float(obj_pos[2] - region_pos[2]))
        return horizontal_dist < horizontal_thresh and height_diff < z_thresh

    return check


def _in_drawer_y_window(obj_name: str, region_name: str, x_thresh: float, y_low_offset: float, y_high_offset: float, z_thresh: float):
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


def _drawer_open_abs(region_name: str, initial_y: float | None, threshold: float):
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


def _drawer_closed_abs(region_name: str, initial_y: float | None, threshold: float):
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


def _microwave_open(joint_thresh: float, fallback_x_thresh: float = 0.65):
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        angle = _microwave_joint_angle(env)
        if angle is not None:
            return abs(angle) > joint_thresh
        handle_pos = _calc_microwave_handle_pos(env)
        if handle_pos is None:
            return False
        return float(handle_pos[0]) < fallback_x_thresh

    return check


def _microwave_closed(dist_thresh: float = 0.05):
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


def _in_microwave(obj_name: str, xy_thresh: float = 0.20):
    return _in_container_site(obj_name, "microwave_1_heating_region", xy_thresh, xy_thresh, -1.0, 1.0)


def _cabinet2(obj_name: str, xy_thresh: float, z_low: float, z_high: float):
    return _in_container_body(obj_name, "wooden_cabinet_2", xy_thresh, z_low, z_high)


def _on_plate(obj_name: str, plate_name: str = "plate_2"):
    return _in_container_body(obj_name, plate_name, 0.06, 0.01, 0.10)


def _table_return(obj_name: str, radius: float):
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _current_body_pos(env, obj_name)
        init = _initial_body_pos(state, obj_name)
        if cur is None or init is None:
            return False
        distance = float(np.linalg.norm(cur - init))
        return distance < radius and 0.0 < float(cur[2]) < 0.80

    return check


def _near_fixed_position(obj_name: str, target: np.ndarray, xy_thresh: float, z_thresh: float):
    def check(env: Any, state: dict[str, Any], stage_start: int) -> bool:
        cur = _current_body_pos(env, obj_name)
        if cur is None:
            return False
        xy_dist = float(np.linalg.norm(cur[:2] - target[:2]))
        z_diff = abs(float(cur[2] - target[2]))
        return xy_dist < xy_thresh and z_diff < z_thresh

    return check


def _pour_stage(range_thresh: float, min_steps: int, hold_angle: float | None = None, hold_frames: int | None = None):
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
        ]
    if task_id == 21:
        return [
            StageSpec("01_Open_Microwave", _microwave_open(0.50)),
            StageSpec("02_Place_Butter_Microwave", _in_microwave("butter_1")),
            StageSpec("03_Place_Chocolate_Microwave", _in_microwave("chocolate_pudding_1")),
        ]
    if task_id == 22:
        return [
            StageSpec("01_Pour_One", _pour_stage(0.30, 10)),
            StageSpec("02_Pour_Two", _pour_stage(0.30, 10)),
            StageSpec("03_Place_Tomato_Aside", _near_fixed_position("tomato_sauce_1", np.array([0.0, -0.2, 0.50], dtype=np.float32), 0.20, 0.20)),
            StageSpec("04_Open_Microwave", _microwave_open(0.30)),
            StageSpec("05_Place_Cookies_Microwave", _in_microwave("cookies_1")),
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


def _goal_override_check(task_id: int) -> Callable[[Any], bool] | None:
    if task_id == 6:
        place_bowl_drainer = _in_container_body("tomato_sauce_1", "bowl_drainer_1", 0.15, -0.05, 0.20)
        return lambda env: place_bowl_drainer(env, {}, 0)
    return None


def run_episode_with_stateful_stages(
    task_id: int,
    env: Any,
    adapter: BasePolicyAdapter,
    prompt: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    post_goal_steps: int,
    stage_specs: list[StageSpec],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override: Callable[[Any, dict[str, bool]], bool] | None,
    fail_on_extra_pour: bool,
    extra_pour_monitor_steps: int,
) -> tuple[float, dict[str, bool], bool | None, dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    t = 0
    state: dict[str, Any] | None = None
    current_stage_start = 0
    all_stages_logged = False
    goal_reached_t: int | None = None
    counting_pour_task = _is_counting_pour_task(task_id)
    extra_pour_check = _extra_pour_check(task_id)
    extra_monitor_start_state_idx: int | None = None
    extra_monitor_deadline_t: int | None = None
    extra_pour_detected = False
    pour_1_step: int | None = None
    pour_2_step: int | None = None

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                t += 1
                continue

            if state is None:
                state = _build_initial_state(env)
                current_stage_start = state["step_idx"]

            adapter_obs, processed_main, processed_wrist = build_eval26_policy_input(
                raw_obs=obs,
                prompt=prompt,
                resize_size=resize_size,
            )
            replay.append(processed_main)
            replay_wrist.append(processed_wrist)
            observe_fn = getattr(adapter, "observe", None)
            if callable(observe_fn):
                observe_fn(adapter_obs, prompt, resize_size)

            if not action_plan:
                actions = ensure_action_chunk(adapter.infer_actions(obs=adapter_obs, prompt=prompt, resize_size=resize_size))
                action_plan.extend(actions[:replan_steps])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            _update_state(obs, state)

            if stage_idx < len(stage_specs):
                spec = stage_specs[stage_idx]
                if spec.check_fn(env, state, current_stage_start):
                    stage_done[spec.name] = True
                    logging.info(f"  [t={t}] Stage completed: {spec.name}")
                    if spec.name.endswith("_Pour_One"):
                        pour_1_step = t
                    elif spec.name.endswith("_Pour_Two"):
                        pour_2_step = t
                        if counting_pour_task and fail_on_extra_pour:
                            extra_monitor_start_state_idx = int(state["step_idx"])
                            extra_monitor_deadline_t = t + extra_pour_monitor_steps
                            logging.info(
                                "  [t=%s] Extra-pour monitor started; deadline=%s.",
                                t,
                                extra_monitor_deadline_t,
                            )
                    stage_idx += 1
                    current_stage_start = state["step_idx"]

            if stage_idx >= len(stage_specs) and not all_stages_logged:
                logging.info(f"  [t={t}] All stages completed.")
                all_stages_logged = True

            if (
                counting_pour_task
                and fail_on_extra_pour
                and extra_pour_check is not None
                and extra_monitor_start_state_idx is not None
                and extra_monitor_deadline_t is not None
                and pour_2_step is not None
                and pour_2_step < t <= extra_monitor_deadline_t
                and extra_pour_check(env, state, extra_monitor_start_state_idx)
            ):
                extra_pour_detected = True
                logging.info(f"  [t={t}] Third pour detected; episode failed.")

            if (
                not counting_pour_task
                and goal_reached_t is None
                and _stage_success_from_stage_done(task_id, stage_done)
            ):
                goal_reached_t = t
                logging.info(
                    f"  [t={t}] Required stages completed. Continuing {post_goal_steps} more steps before exit."
                )

            required_stages_complete = _stage_success_from_stage_done(task_id, stage_done)
            extra_monitor_complete = (
                not fail_on_extra_pour
                or (
                    extra_monitor_deadline_t is not None
                    and t >= extra_monitor_deadline_t
                )
            )
            if done:
                break
            if not counting_pour_task and goal_reached_t is not None and (t - goal_reached_t) >= post_goal_steps:
                break
            if counting_pour_task:
                if extra_pour_detected or (required_stages_complete and extra_monitor_complete):
                    break
            t += 1
    except Exception as exc:
        logging.exception(f"Episode failed: {exc}")

    score = _stage_score_pct(task_id, stage_done)
    all_stages_complete = bool(stage_done) and all(stage_done.values())
    extra_monitor_complete = (
        not fail_on_extra_pour
        or (
            extra_monitor_deadline_t is not None
            and t >= extra_monitor_deadline_t
        )
    )
    required_stages_complete = _stage_success_from_stage_done(task_id, stage_done)
    stage_success = required_stages_complete and (
        not counting_pour_task
        or (extra_monitor_complete and not extra_pour_detected)
    )
    if extra_pour_detected:
        failure_reason = "extra_pour"
    elif not stage_success:
        failure_reason = "incomplete_stage"
    elif counting_pour_task and not extra_monitor_complete:
        failure_reason = "monitor_incomplete"
    else:
        failure_reason = None
    diagnostics = {
        "stage_success": bool(stage_success),
        "extra_pour_detected": bool(extra_pour_detected),
        "pour_1_step": pour_1_step,
        "pour_2_step": pour_2_step,
        "extra_monitor_end_step": (
            extra_monitor_deadline_t
            if extra_monitor_deadline_t is not None and t >= extra_monitor_deadline_t
            else None
        ),
        "failure_reason": failure_reason,
    }
    goal_success = stage_success
    return score, stage_done, goal_success, diagnostics, replay, replay_wrist


# Use the same Task2-26 stage/goal checker as the VLM/VLA reference
# evaluation path, while preserving the generic adapter interface.
from task2_26_reference_stage import (  # noqa: E402
    StageSpec,
    _build_initial_state,
    _extra_pour_check,
    _goal_override_check,
    _is_drawer_task,
    _is_counting_pour_task,
    _patch_env_resolution,
    _stage_score_pct,
    _stage_success_from_stage_done,
    _task_specs,
    _update_state,
)


def run_eval_task(
    task_id: int,
    num_trials_per_task: int,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    post_goal_steps: int,
    video_out_path: str,
    seed: int,
    adapter: BasePolicyAdapter | None = None,
    adapter_spec: str | None = None,
    adapter_kwargs: dict[str, Any] | None = None,
    fail_on_extra_pour: bool = True,
    extra_pour_monitor_steps: int = 30,
) -> dict[str, Any]:
    if task_id == 1:
        raise ValueError("Task 1 is intentionally excluded from eval_tasks2_26.py. Use eval_task1_only.py or reference_evaluation/task1_nomap_reference/eval_task1_nomap_reference.py.")

    if adapter is None:
        adapter = load_policy_adapter(adapter_spec or "", **(adapter_kwargs or {}))
        owns_adapter = True
    else:
        owns_adapter = False

    tid, task_key = ec._resolve_task_id(task_id)
    bddl_path = ec._resolve_bddl_path(task_id)
    prompt = ec.get_prompt(task_key, bddl_path.stem)
    video_dir = Path(video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Using BDDL: {bddl_path}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Video output: {video_dir}")

    OffScreenRenderEnv = ec._get_env_class()
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        initialization_noise=None,
    )

    stage_specs = _task_specs(task_id)
    counting_pour_task = _is_counting_pour_task(task_id)
    goal_monitor_dict = {} if counting_pour_task else ec._build_goal_monitor_dict(bddl_path)
    goal_check_override = _goal_override_check(task_id)

    total_score = 0.0
    stage_totals = {spec.name: 0 for spec in stage_specs}
    goal_succ_cnt = 0
    tsr_succ_cnt = 0
    episodes: list[dict[str, Any]] = []

    try:
        for ep in range(num_trials_per_task):
            current_seed = seed + ep
            np.random.seed(current_seed)
            try:
                env.seed(current_seed)
            except AttributeError:
                pass
            adapter.reset()

            score, stage_done, goal_success, diagnostics, replay, replay_wrist = run_episode_with_stateful_stages(
                task_id=task_id,
                env=env,
                adapter=adapter,
                prompt=prompt,
                resize_size=resize_size,
                replan_steps=replan_steps,
                num_steps_wait=num_steps_wait,
                max_steps=max_steps,
                post_goal_steps=post_goal_steps,
                stage_specs=stage_specs,
                goal_monitor_dict=goal_monitor_dict,
                goal_check_override=goal_check_override,
                fail_on_extra_pour=fail_on_extra_pour,
                extra_pour_monitor_steps=extra_pour_monitor_steps,
            )
            total_score += score
            for name, ok in stage_done.items():
                stage_totals[name] += int(ok)
            tsr_success = bool(diagnostics["stage_success"])
            tsr_succ_cnt += int(tsr_success)
            goal_succ_cnt += score / 100.0

            base_name = ec.get_video_basename(task_id, ep, current_seed, tsr_success)
            if replay:
                imageio.mimwrite(video_dir / f"{base_name}.mp4", replay, fps=10)
            if replay_wrist:
                imageio.mimwrite(video_dir / f"{base_name}_wrist.mp4", replay_wrist, fps=10)

            stages_str = " | ".join(f"{n}={'Y' if stage_done[n] else 'N'}" for n in stage_done)
            logging.info(
                f"Episode {ep} (seed={current_seed}): TSR={100.0 if tsr_success else 0.0:.0f}% | "
                f"CSR={score:.0f}% | {stages_str} | "
                f"failure_reason={diagnostics['failure_reason']}"
            )
            episode_diagnostics = {k: v for k, v in diagnostics.items() if k != "stage_success"}
            episodes.append(
                {
                    "ep": ep,
                    "seed": current_seed,
                    "TSR": 100.0 if tsr_success else 0.0,
                    "CSR": float(score),
                    "stage_done": stage_done,
                    **episode_diagnostics,
                }
            )
    finally:
        env.close()
        if owns_adapter:
            close_fn = getattr(adapter, "close", None)
            if callable(close_fn):
                close_fn()

    n = num_trials_per_task
    avg_score = total_score / max(1, n)
    logging.info("============================================================")
    logging.info(f"Final result - average stage success rate = {avg_score:.1f}%")
    for name, cnt in stage_totals.items():
        logging.info(f"  {name}: {cnt}/{n} ({(cnt / max(1, n)) * 100:.0f}%)")
    tsr_pct = 100.0 * tsr_succ_cnt / max(1, n)
    logging.info(f"Final result - TSR all-stage success rate: {tsr_succ_cnt}/{n} ({tsr_pct:.1f}%)")
    goal_pct = 100.0 * goal_succ_cnt / max(1, n)
    logging.info(f"Final result - CSR stage completion rate: {goal_pct:.1f}%")
    logging.info(f"Video output: {video_dir}")
    logging.info("============================================================")

    return {
        "task_id": tid if tid is not None else task_id,
        "task_key": task_key,
        "prompt": prompt,
        "bddl_path": str(bddl_path),
        "video_dir": str(video_dir),
        "TSR": float(tsr_pct),
        "CSR": float(goal_pct),
        "episodes": episodes,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one RoboMemArena task from task2..26 with a custom policy adapter.")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--adapter-spec", required=True)
    parser.add_argument("--adapter-kwargs", default="")
    parser.add_argument("--resize-size", type=int, default=256)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--post-goal-steps", type=int, default=200)
    parser.add_argument(
        "--fail-on-extra-pour",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--extra-pour-monitor-steps",
        "--post-stage-steps",
        dest="extra_pour_monitor_steps",
        type=int,
        default=30,
    )
    parser.add_argument("--video-out-path", default="outputs/tasks2_26_eval")
    parser.add_argument("--seed", type=int, default=100)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = build_argparser().parse_args()
    _patch_env_resolution()
    run_eval_task(
        task_id=args.task_id,
        num_trials_per_task=args.num_trials_per_task,
        adapter_spec=args.adapter_spec,
        adapter_kwargs=ec.parse_adapter_kwargs(args.adapter_kwargs),
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        post_goal_steps=args.post_goal_steps,
        fail_on_extra_pour=args.fail_on_extra_pour,
        extra_pour_monitor_steps=args.extra_pour_monitor_steps,
        video_out_path=args.video_out_path,
        seed=args.seed,
    )
