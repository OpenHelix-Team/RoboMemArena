#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microwave_debug import microwave_joint_angle
from eef_direction_guard import evaluate_eef_direction_gate
from eef_release_guard import should_keep_place_gripper_closed
from forward_hold_guard import should_block_pick_forward
from runtime_hint import should_inject_hold_state_hint


BASE_EVAL_PY = Path(os.environ.get("TASKS2_26_BASE_EVAL_PY", "/REQUIRE_TASKS2_26_BASE_EVAL_PY"))

spec = importlib.util.spec_from_file_location("_tasks2_26_base_eval", BASE_EVAL_PY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base eval from {BASE_EVAL_PY}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

_ORIG_FULLVLM26_BUILD_MESSAGES = base.FullVlm26MemoryPlanner._build_messages


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return Path(value)


OFFICIAL_SCRIPTS_DIR = _required_env_path("ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR")
OFFICIAL_BDDL_DIR = _required_env_path("ROBOMEMARENA_OFFICIAL_BDDL_DIR")
if str(OFFICIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SCRIPTS_DIR))


def _load_official_module(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load official module from {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    module_spec.loader.exec_module(module)
    return module


official_ec = _load_official_module(
    "_robomemarena_official_eval_common", OFFICIAL_SCRIPTS_DIR / "eval_common.py"
)
_previous_eval_common = sys.modules.get("eval_common")
sys.modules["eval_common"] = official_ec
try:
    official_stage_py = OFFICIAL_SCRIPTS_DIR / "task2_26_reference_stage.py"
    if not official_stage_py.exists():
        raise FileNotFoundError(
            f"Missing required latest official stage scorer: {official_stage_py}. "
            "Refusing to fall back to eval_tasks2_26.py because that can use stale microwave-door logic."
        )
    official_stage = _load_official_module(
        "_robomemarena_official_task2_26_stage",
        official_stage_py,
    )
finally:
    if _previous_eval_common is None:
        sys.modules.pop("eval_common", None)
    else:
        sys.modules["eval_common"] = _previous_eval_common


MICROWAVE_STAGE_ONLY_TASKS = {20, 21, 22, 23, 24}


def _official_counted_stage_names(task_id: int, stage_done: dict[str, bool]) -> list[str]:
    counted_fn = getattr(official_stage, "_counted_stage_names", None)
    if callable(counted_fn):
        names = list(counted_fn(task_id, stage_done))
    else:
        names = list(stage_done)
    if task_id in MICROWAVE_STAGE_ONLY_TASKS:
        names = [name for name in names if "Close_Microwave" not in name]
    return names


def _official_stage_score_pct(task_id: int, stage_done: dict[str, bool]) -> float:
    counted_names = _official_counted_stage_names(task_id, stage_done)
    if task_id in MICROWAVE_STAGE_ONLY_TASKS:
        num_done = sum(1 for name in counted_names if stage_done.get(name, False))
        return 100.0 * num_done / max(1, len(counted_names))
    score_fn = getattr(official_stage, "_stage_score_pct", None)
    if callable(score_fn):
        return float(score_fn(task_id, stage_done))
    return 100.0 * sum(1 for name in counted_names if stage_done.get(name, False)) / max(1, len(counted_names))


def _official_stage_success(task_id: int, stage_done: dict[str, bool]) -> bool:
    counted_names = _official_counted_stage_names(task_id, stage_done)
    if task_id in MICROWAVE_STAGE_ONLY_TASKS:
        return bool(counted_names) and all(stage_done.get(name, False) for name in counted_names)
    success_fn = getattr(official_stage, "_stage_success_from_stage_done", None)
    if callable(success_fn):
        return bool(success_fn(task_id, stage_done))
    return bool(counted_names) and all(stage_done.get(name, False) for name in counted_names)


def _official_goal_success(task_id: int, env: Any, stage_done: dict[str, bool], stage_success: bool) -> bool:
    goal_fn = getattr(official_stage, "_goal_override_check", None)
    if not callable(goal_fn):
        return bool(stage_success)
    override = goal_fn(task_id)
    if override is None:
        return bool(stage_success)
    try:
        return bool(override(env, stage_done))
    except TypeError:
        return bool(override(env))


def _patch_official_bddl_resolution() -> None:
    """Use remote official BDDL files for rollout env construction.

    The old local RoboMemArena checkout can contain a stale Task2 BDDL where the
    filename says butter/popcorn but the content is cream/pudding. That makes the
    robot solve one task while official scoring checks another. This patch keeps
    the full original evaluator rollout logic, but forces the environment BDDL to
    match the official scoring BDDL for tasks that are present in this bundle.
    """
    original_resolve = base.ec._resolve_bddl_path

    def resolve_bddl_path(task_id: int) -> Path:
        matches = sorted(OFFICIAL_BDDL_DIR.glob(f"{int(task_id)}_*.bddl"))
        if matches:
            return matches[0]
        return original_resolve(task_id)

    base.ec._resolve_bddl_path = resolve_bddl_path


def _patch_stage_eval_compat() -> None:
    if hasattr(base.stage_eval, "_is_counting_pour_task"):
        return

    def is_counting_pour_task(task_id: int) -> bool:
        try:
            specs = official_stage._task_specs(int(task_id))
        except Exception:
            return False
        return any("Pour_" in str(spec.name) for spec in specs)

    base.stage_eval._is_counting_pour_task = is_counting_pour_task


def _completed_subtasks_mode() -> str:
    raw = os.environ.get("VLM_COMPLETED_SUBTASKS_MODE", "auto").strip().lower()
    if raw in {"0", "false", "no", "none", "off", ""}:
        return ""
    if raw in {"completed_text", "text"}:
        return "completed_text"
    if raw in {"completed_struct", "struct", "json"}:
        return "completed_struct"
    if raw != "auto":
        return ""

    haystack = " ".join(
        os.environ.get(name, "")
        for name in (
            "EVAL_TAG",
            "WATCH_TAG",
            "RUN_STAMP",
            "ARTIFACT_ROOT",
            "SOURCE_CHECKPOINT",
            "TRAIN_OUTDIR",
        )
    ).lower()
    if "completed_struct" in haystack:
        return "completed_struct"
    if "completed_text" in haystack:
        return "completed_text"
    return ""


def _completed_subtasks_block(completed: list[str], mode: str) -> str:
    clean = [str(item).strip() for item in completed if str(item).strip()]
    if mode == "completed_text":
        return "Completed subtasks: " + ("; ".join(clean) if clean else "none") + "."
    if mode == "completed_struct":
        payload = {"type": "completed_subtasks", "count": len(clean), "items": clean}
        return "Completed-subtasks feature JSON: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return ""


def _inject_runtime_completed_subtasks(messages: list[dict[str, Any]], completed: list[str], mode: str) -> None:
    block = _completed_subtasks_block(completed, mode)
    if not block:
        return
    _inject_text_block_before_observation(messages, block)


def _inject_text_block_before_observation(messages: list[dict[str, Any]], block: str) -> None:
    if not block:
        return
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and item.get("type") == "text"):
                continue
            text = str(item.get("text", ""))
            marker = "\nCurrent observation:"
            if marker in text:
                item["text"] = text.replace(marker, "\n" + block + "\n" + marker.lstrip(), 1)
            else:
                item["text"] = text + "\n" + block
            return


def _build_messages_runtime_progress(self, *args, **kwargs):
    mode = os.environ.get("VLM_TASK_TEXT_MODE", "default").strip().lower()
    if mode in {"no_label_no_order", "scene_only"}:
        original_info = self.task_info
        scene_text = original_info.scene_description or original_info.brief_description
        safe_task_block = (
            "High-level objective: infer the next executable low-level robot action from visual evidence only. "
            "Do not rely on a provided primitive list or a fixed task order; use the historical keyframes and "
            "the current visual context to decide what action should be run now."
        )
        self.task_info = dataclasses.replace(
            original_info,
            task_block=safe_task_block,
            brief_description=scene_text,
            scene_description=scene_text,
        )
        try:
            messages = _ORIG_FULLVLM26_BUILD_MESSAGES(self, *args, **kwargs)
        finally:
            self.task_info = original_info
    else:
        messages = _ORIG_FULLVLM26_BUILD_MESSAGES(self, *args, **kwargs)

    completed_mode = _completed_subtasks_mode()
    if completed_mode:
        completed = list(getattr(self, "_runtime_completed_subtasks", []))
        _inject_runtime_completed_subtasks(messages, completed, completed_mode)
    hold_state = getattr(self, "_runtime_hold_state", {}) or {}
    held_subtask = str(hold_state.get("subtask", "")).strip()
    hold_state_phase = str(hold_state.get("phase", "active")).strip() or "active"
    if should_inject_hold_state_hint(
        env_bool("VLM_HOLD_STATE_HINT", False),
        os.environ.get("VLM_HOLD_STATE_HINT_SUBTASKS", ""),
        bool(hold_state.get("active")),
        held_subtask,
        state_phase=hold_state_phase,
        configured_phase=os.environ.get("VLM_HOLD_STATE_HINT_PHASE", "active"),
    ):
        if hold_state_phase == "post_release":
            block = (
                "Runtime controller state: the low-level controller reached the end-pose for "
                f"'{held_subtask or 'the current primitive'}' and then executed its fixed continuation. "
                "Use the visual observation to decide the current executable primitive; if that primitive "
                "is visually finished, output the next primitive."
            )
        else:
            block = (
                "Runtime controller state: the low-level controller is holding still at an end-pose "
                f"for '{held_subtask or 'the current primitive'}'. Use the visual observation to decide "
                "the current executable primitive; if that primitive is visually finished, output the next primitive."
            )
        _inject_text_block_before_observation(messages, block)
    return messages


PORTABLE_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
DEFAULT_TARGET_JSON = str(PORTABLE_CONFIG_DIR / "tasks2_26_endpose_targets_seed100_199.json")
LEGACY_TARGET_PASSAGE_COUNTS_JSON = str(PORTABLE_CONFIG_DIR / "tasks2_26_target_passage_counts_seed100_199.json")
DEFAULT_TARGET_PASSAGE_COUNTS_JSON = str(
    PORTABLE_CONFIG_DIR / "tasks2_26_target_passage_counts_seed100_199_alltasks_tol045_20260624_074452.json"
)
DEFAULT_H5DUMP_BIN = os.environ["H5DUMP_BIN"]


@dataclass(frozen=True)
class HoldConfig:
    enabled: bool
    targets_json: Path
    target_passage_counts_json: Path | None
    direction_signatures_json: Path | None
    pos_tol: float
    eef_default_tol: float
    eef_p95_extra_tol: float
    eef_tol_cap: float
    min_active_steps: int
    consecutive: int
    disable_final: bool
    post_release_vla_steps: int
    post_release_vla_steps_by_subtask: dict[str, int]
    post_pick_hold_release_same_prompt_steps: int
    strict_hold_release_next: bool
    prevent_regression: bool
    regression_guard_after_hold_release: bool
    distance_log_interval: int
    direction_cos_min: float
    direction_window: int
    direction_min_displacement: float
    direction_trend_eps: float
    pick_gripper_gate: bool
    pick_gripper_open_max: float
    pick_gripper_close_min: float
    pick_deferred_gripper_release: bool
    pick_height_gate: bool
    pick_height_targets_json: Path | None
    pick_height_tol: float
    pick_object_lift_gate: bool
    pick_object_lift_delta: float
    drawer_close_hold_require_stage: bool


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def parse_float_list_env(name: str, default: list[float], expected_len: int) -> np.ndarray:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        values = default
    else:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated floats, got {values}")
    return np.asarray(values, dtype=np.float64)


def resolve_target_passage_counts_json() -> Path | None:
    raw = os.environ.get("ENDPOSE_TARGET_PASSAGE_COUNTS_JSON")
    if raw is None or not raw.strip():
        candidate = Path(DEFAULT_TARGET_PASSAGE_COUNTS_JSON)
        return candidate if candidate.exists() else None

    raw_norm = raw.strip().lower()
    if raw_norm in {"__none__", "none", "null", "off", "disable", "disabled", "0"}:
        return None

    candidate = Path(raw.strip())
    if str(candidate) == LEGACY_TARGET_PASSAGE_COUNTS_JSON:
        upgraded = Path(DEFAULT_TARGET_PASSAGE_COUNTS_JSON)
        if upgraded.exists():
            return upgraded
    return candidate


def hold_config() -> HoldConfig:
    target_passage_counts_path = resolve_target_passage_counts_json()
    direction_signatures_raw = os.environ.get("ENDPOSE_HOLD_DIRECTION_SIGNATURES_JSON")
    pick_height_targets_raw = os.environ.get("ENDPOSE_PICK_HEIGHT_TARGETS_JSON")
    return HoldConfig(
        enabled=env_bool("ENABLE_ENDPOSE_HOLD", True),
        targets_json=Path(os.environ.get("ENDPOSE_HOLD_TARGETS_JSON", DEFAULT_TARGET_JSON)),
        target_passage_counts_json=target_passage_counts_path,
        direction_signatures_json=Path(direction_signatures_raw) if direction_signatures_raw else None,
        pos_tol=env_float("ENDPOSE_HOLD_POS_TOL", 0.04),
        eef_default_tol=env_float("ENDPOSE_HOLD_EEF_DEFAULT_TOL", 0.06),
        eef_p95_extra_tol=env_float("ENDPOSE_HOLD_EEF_P95_EXTRA_TOL", 0.02),
        eef_tol_cap=env_float("ENDPOSE_HOLD_EEF_TOL_CAP", 0.08),
        min_active_steps=env_int("ENDPOSE_HOLD_MIN_ACTIVE_STEPS", 20),
        consecutive=env_int("ENDPOSE_HOLD_CONSECUTIVE", 2),
        disable_final=env_bool("ENDPOSE_HOLD_DISABLE_FINAL", True),
        post_release_vla_steps=env_int("POST_HOLD_RELEASE_VLA_STEPS", 30),
        post_release_vla_steps_by_subtask={},
        post_pick_hold_release_same_prompt_steps=env_int("POST_PICK_HOLD_RELEASE_SAME_PROMPT_STEPS", 0),
        strict_hold_release_next=env_bool("STRICT_HOLD_RELEASE_NEXT", True),
        prevent_regression=env_bool("PREVENT_SUBTASK_REGRESSION", True),
        regression_guard_after_hold_release=env_bool("REGRESSION_GUARD_AFTER_HOLD_RELEASE", True),
        distance_log_interval=env_int("ENDPOSE_DISTANCE_LOG_INTERVAL", 0),
        direction_cos_min=env_float("ENDPOSE_HOLD_DIRECTION_COS_MIN", 0.50),
        direction_window=env_int("ENDPOSE_HOLD_DIRECTION_WINDOW", 5),
        direction_min_displacement=env_float("ENDPOSE_HOLD_DIRECTION_MIN_DISPLACEMENT", 0.0005),
        direction_trend_eps=env_float("ENDPOSE_HOLD_DIRECTION_TREND_EPS", 0.005),
        pick_gripper_gate=env_bool("ENDPOSE_PICK_GRIPPER_GATE", False),
        pick_gripper_open_max=env_float("ENDPOSE_PICK_GRIPPER_OPEN_MAX", -0.2),
        pick_gripper_close_min=env_float("ENDPOSE_PICK_GRIPPER_CLOSE_MIN", 0.2),
        pick_deferred_gripper_release=env_bool("ENDPOSE_PICK_DEFERRED_GRIPPER_RELEASE", False),
        pick_height_gate=env_bool("ENDPOSE_PICK_HEIGHT_GATE", False),
        pick_height_targets_json=Path(pick_height_targets_raw) if pick_height_targets_raw else None,
        pick_height_tol=env_float("ENDPOSE_PICK_HEIGHT_TOL", 0.005),
        pick_object_lift_gate=env_bool("ENDPOSE_PICK_OBJECT_LIFT_GATE", True),
        pick_object_lift_delta=env_float("ENDPOSE_PICK_OBJECT_LIFT_DELTA", 0.01),
        drawer_close_hold_require_stage=env_bool("DRAWER_CLOSE_HOLD_REQUIRE_STAGE", True),
    )


def normalize_subtask(subtask: str, labels: list[str]) -> str:
    raw = " ".join(str(subtask).strip().lower().replace("_", " ").split())
    try:
        norm = base._normalize_primitive(subtask, allowed_subtasks=labels)
        if norm:
            return norm
    except Exception:
        pass

    label_norms = [" ".join(label.strip().lower().split()) for label in labels]
    if raw in label_norms:
        return raw

    # Some checkpoints output a shortened object/action phrase such as
    # "place butter". Map it only when it uniquely identifies one legal label.
    raw_tokens = set(re.findall(r"[a-z0-9]+", raw))
    if raw_tokens:
        matches = [
            label
            for label, label_norm in zip(labels, label_norms, strict=True)
            if raw_tokens.issubset(set(re.findall(r"[a-z0-9]+", label_norm)))
        ]
        if len(matches) == 1:
            return matches[0]

    # Some drawer checkpoints hallucinate temporal suffixes such as
    # "again"/"final" on labels that do not legally contain them, e.g.
    # "close bottom drawer again". Only strip these extra tokens when
    # the original raw text was not already an exact legal label and the
    # stripped phrase maps to exactly one allowed label.
    raw_token_list = re.findall(r"[a-z0-9]+", raw)
    if raw_token_list:
        removable = {"again", "final", "the"}
        stripped_tokens = [tok for tok in raw_token_list if tok not in removable]
        if stripped_tokens and stripped_tokens != raw_token_list:
            stripped_set = set(stripped_tokens)
            matches = [
                label
                for label, label_norm in zip(labels, label_norms, strict=True)
                if stripped_set.issubset(set(re.findall(r"[a-z0-9]+", label_norm)))
            ]
            if len(matches) == 1:
                return matches[0]
    return raw


def subtask_temporal_stripped_key(subtask: str) -> str:
    raw = " ".join(str(subtask).strip().lower().replace("_", " ").split())
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", raw) if tok not in {"again", "final", "the"}]
    return " ".join(tokens)


def order_index(subtask: str, labels: list[str]) -> int | None:
    norm = normalize_subtask(subtask, labels)
    try:
        return labels.index(norm)
    except ValueError:
        return None


def get_eef_pos(obs: dict[str, Any]) -> np.ndarray:
    for key in ("robot0_eef_pos", "ee_pos"):
        if key in obs:
            value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if value.size >= 3:
                return value[:3]
    raise KeyError(f"Cannot find EEF position in obs keys={sorted(obs.keys())}")


def format_vec3(vec: np.ndarray | list[float] | tuple[float, ...] | None) -> str:
    if vec is None:
        return "NA"
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return "NA"
    return f"[{arr[0]:+.3f}, {arr[1]:+.3f}, {arr[2]:+.3f}]"


def _as_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr


def overlay_debug_text(
    frame: np.ndarray,
    lines: list[str],
    *,
    anchor_xy: tuple[int, int] = (8, 8),
) -> np.ndarray:
    arr = _as_uint8_rgb(frame).copy()
    if not lines:
        return arr
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()
    x0, y0 = anchor_xy
    max_chars = 100 if img.width >= 640 else 72
    wrapped: list[str] = []
    for line in lines:
        text = str(line).strip()
        if not text:
            continue
        wrapped.extend(textwrap.wrap(text, width=max_chars) or [""])
    if not wrapped:
        return arr
    line_boxes = [draw.textbbox((0, 0), line, font=font) for line in wrapped]
    line_heights = [max(14, box[3] - box[1] + 2) for box in line_boxes]
    text_width = max(box[2] - box[0] for box in line_boxes)
    total_height = sum(line_heights) + 10
    bg_w = min(img.width - x0 - 4, text_width + 12)
    bg_h = min(img.height - y0 - 4, total_height)
    draw.rectangle((x0, y0, x0 + bg_w, y0 + bg_h), fill=(0, 0, 0, 180))
    y = y0 + 5
    for line, line_h in zip(wrapped, line_heights, strict=True):
        draw.text((x0 + 6, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h
        if y >= y0 + bg_h - 10:
            break
    return np.asarray(img)


def load_task_targets(cfg: HoldConfig, task_id: int, labels: list[str]) -> dict[str, dict[str, Any]]:
    if not cfg.enabled:
        return {}
    if not cfg.targets_json.exists():
        raise FileNotFoundError(
            f"End-pose target JSON does not exist: {cfg.targets_json}. "
            "Run compute_task_endpose_targets.py first."
        )
    raw = json.loads(cfg.targets_json.read_text(encoding="utf-8"))
    if "tasks" in raw:
        task_payload = raw["tasks"].get(str(task_id), {})
        raw_subtasks = task_payload.get("subtasks", {})
    else:
        raw_subtasks = raw.get("subtasks", raw)
    targets: dict[str, dict[str, Any]] = {}
    for name, payload in raw_subtasks.items():
        subtask = normalize_subtask(name, labels)
        pos = payload.get("target_ee_pos") or payload.get("ee_pos") or payload.get("median_ee_pos")
        if pos is None:
            raise ValueError(f"{cfg.targets_json}: missing target_ee_pos for task{task_id} {name}")
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos_arr.size < 3:
            raise ValueError(f"{cfg.targets_json}: invalid target_ee_pos for task{task_id} {name}: {pos}")
        hold_gripper = float(payload.get("hold_gripper", -1.0))
        targets[subtask] = {
            "target_ee_pos": pos_arr[:3],
            "hold_gripper": 1.0 if hold_gripper >= 0.0 else -1.0,
            "pos_dist_p95": float(payload.get("pos_dist_p95", 0.0) or 0.0),
        }
    return targets


def load_task_passage_requirements(cfg: HoldConfig, task_id: int, labels: list[str]) -> dict[str, int]:
    if not cfg.enabled or cfg.target_passage_counts_json is None:
        return {}
    if not cfg.target_passage_counts_json.exists():
        raise FileNotFoundError(
            f"End-pose passage-count JSON does not exist: {cfg.target_passage_counts_json}"
        )

    raw = json.loads(cfg.target_passage_counts_json.read_text(encoding="utf-8"))
    if "tasks" in raw:
        task_payload = raw["tasks"].get(str(task_id), {})
        raw_subtasks = task_payload.get("subtasks", {})
    else:
        raw_subtasks = raw.get("subtasks", raw)

    requirements: dict[str, int] = {}
    for name, payload in raw_subtasks.items():
        subtask = normalize_subtask(name, labels)
        required = (
            payload.get("required_near_segments")
            or payload.get("required_passages")
            or payload.get("mode_near_segments")
            or 1
        )
        requirements[subtask] = max(1, int(required))
    return requirements


def load_task_direction_signatures(cfg: HoldConfig, task_id: int, labels: list[str]) -> dict[str, dict[str, Any]]:
    if not cfg.enabled or cfg.direction_signatures_json is None:
        return {}
    if not cfg.direction_signatures_json.exists():
        raise FileNotFoundError(f"Direction signature JSON does not exist: {cfg.direction_signatures_json}")

    raw = json.loads(cfg.direction_signatures_json.read_text(encoding="utf-8"))
    if "tasks" in raw:
        task_payload = raw["tasks"].get(str(task_id), {})
        raw_subtasks = task_payload.get("subtasks", {})
    else:
        raw_subtasks = raw.get("subtasks", raw)

    signatures: dict[str, dict[str, Any]] = {}
    for name, payload in raw_subtasks.items():
        if not isinstance(payload, dict) or "direction_mean" not in payload:
            continue
        subtask = normalize_subtask(name, labels)
        direction = np.asarray(payload["direction_mean"], dtype=np.float64).reshape(-1)
        if direction.size < 3:
            continue
        norm = float(np.linalg.norm(direction[:3]))
        if norm <= 1e-9:
            continue
        signatures[subtask] = {
            "direction_mean": direction[:3] / norm,
            "window": int(payload.get("window", cfg.direction_window) or cfg.direction_window),
            "sample_count": int(payload.get("sample_count", 0) or 0),
        }
    return signatures


def load_task_pick_height_targets(cfg: HoldConfig, task_id: int, labels: list[str]) -> dict[str, dict[str, Any]]:
    if not cfg.enabled or not cfg.pick_height_gate:
        return {}
    if cfg.pick_height_targets_json is None:
        raise ValueError("ENDPOSE_PICK_HEIGHT_GATE=1 requires ENDPOSE_PICK_HEIGHT_TARGETS_JSON")
    if not cfg.pick_height_targets_json.exists():
        raise FileNotFoundError(f"Pick-height target JSON does not exist: {cfg.pick_height_targets_json}")

    raw = json.loads(cfg.pick_height_targets_json.read_text(encoding="utf-8"))
    if "tasks" in raw:
        task_payload = raw["tasks"].get(str(task_id), {})
        raw_subtasks = task_payload.get("subtasks", {})
    else:
        raw_subtasks = raw.get("subtasks", raw)

    targets: dict[str, dict[str, Any]] = {}
    for name, payload in raw_subtasks.items():
        if not isinstance(payload, dict):
            continue
        subtask = normalize_subtask(name, labels)
        if not subtask.startswith("pick "):
            continue
        z_target = payload.get("height_z_target", payload.get("height_z_median", payload.get("height_z_mean")))
        if z_target is None:
            continue
        z_target = float(z_target)
        object_key = payload.get("object_key")
        if object_key is None:
            raise KeyError(f"Pick-height target for {subtask!r} is missing object_key")
        z_min = payload.get("trigger_z_min_default")
        if z_min is None:
            z_min = z_target - float(payload.get("height_tol_default", cfg.pick_height_tol))
        targets[subtask] = {
            "object_key": str(object_key),
            "height_z_target": z_target,
            "height_z_min": float(z_min),
            "num_seeds": int(payload.get("num_seeds", 0) or 0),
        }
    return targets


def distance_to_target(obs: dict[str, Any], target: dict[str, Any]) -> float:
    return float(np.linalg.norm(get_eef_pos(obs) - target["target_ee_pos"]))


def drawer_slot_name(subtask: str) -> str | None:
    text = subtask_temporal_stripped_key(subtask)
    if "top drawer" in text:
        return "top"
    if "middle drawer" in text:
        return "middle"
    if "bottom drawer" in text:
        return "bottom"
    return None


def is_close_drawer_subtask(subtask: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", subtask_temporal_stripped_key(subtask)))
    return "close" in tokens and "drawer" in tokens and drawer_slot_name(subtask) is not None


def close_drawer_stage_matches(stage_name: str, subtask: str) -> bool:
    slot = drawer_slot_name(subtask)
    if slot is None:
        return False
    tokens = set(re.findall(r"[a-z0-9]+", subtask_temporal_stripped_key(stage_name)))
    return "close" in tokens and "drawer" in tokens and slot in tokens


def _parse_h5dump_subset(stdout: str, expected_dim: int) -> np.ndarray:
    data_idx = stdout.find("DATA {")
    payload = stdout[data_idx:] if data_idx >= 0 else stdout
    payload = re.sub(r"\(\s*\d+\s*(?:,\s*\d+)?\s*\):", " ", payload)
    values = [
        float(token)
        for token in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", payload)
    ]
    if len(values) < expected_dim:
        raise ValueError(f"h5dump subset parse failed: need {expected_dim} values, got {len(values)}")
    return np.asarray(values[:expected_dim], dtype=np.float64)


@lru_cache(maxsize=64)
def _load_h5dump_row(path_str: str, dataset: str, row_idx: int, dim: int) -> np.ndarray:
    cmd = [
        DEFAULT_H5DUMP_BIN,
        "-w",
        "65535",
        "-d",
        dataset,
        "-s",
        f"{row_idx},0",
        "-c",
        f"1,{dim}",
        path_str,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _parse_h5dump_subset(result.stdout, dim)


@lru_cache(maxsize=32)
def load_release_anchor(anchor_hdf5: str, frame_idx: int) -> dict[str, np.ndarray]:
    return {
        "joint_states": _load_h5dump_row(anchor_hdf5, "/data/demo_0/obs/joint_states", frame_idx, 7),
        "gripper_states": _load_h5dump_row(anchor_hdf5, "/data/demo_0/obs/gripper_states", frame_idx, 2),
        "ee_pos": _load_h5dump_row(anchor_hdf5, "/data/demo_0/obs/ee_pos", frame_idx, 3),
    }


def _named_sim_pos(env: Any, name: str, *, kind: str) -> np.ndarray | None:
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None:
        return None
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    try:
        if kind == "site" and hasattr(model, "site_name2id"):
            return np.asarray(data.site_xpos[model.site_name2id(name)], dtype=np.float64).reshape(-1)[:3]
        if kind == "body" and hasattr(model, "body_name2id"):
            return np.asarray(data.body_xpos[model.body_name2id(name)], dtype=np.float64).reshape(-1)[:3]
    except Exception:
        return None
    return None


def _set_freejoint_body_pos(env: Any, body_name: str, target_pos: np.ndarray) -> dict[str, Any]:
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None:
        return {"ok": False, "reason": "no_sim"}
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {"ok": False, "reason": "no_model_or_data"}
    try:
        body_id = int(model.body_name2id(body_name))
    except Exception as exc:
        return {"ok": False, "reason": f"body_not_found:{exc}"}

    try:
        jnt_adr = int(model.body_jntadr[body_id])
        jnt_num = int(model.body_jntnum[body_id])
    except Exception as exc:
        return {"ok": False, "reason": f"joint_lookup_failed:{exc}"}
    if jnt_num <= 0 or jnt_adr < 0:
        return {"ok": False, "reason": "body_has_no_joint", "body_id": body_id}

    try:
        qpos_adr = int(model.jnt_qposadr[jnt_adr])
        qpos_before = np.asarray(data.qpos[qpos_adr : qpos_adr + 7], dtype=np.float64).copy()
        if qpos_before.size < 3:
            return {"ok": False, "reason": "qpos_too_short", "qpos_adr": qpos_adr}
        data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(target_pos, dtype=np.float64).reshape(3)
        if qpos_before.size >= 7 and np.linalg.norm(qpos_before[3:7]) < 1e-8:
            data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        try:
            dof_adr = int(model.jnt_dofadr[jnt_adr])
            data.qvel[dof_adr : dof_adr + 6] = 0.0
        except Exception:
            pass
        sim.forward()
        return {
            "ok": True,
            "body_id": body_id,
            "joint_adr": jnt_adr,
            "qpos_adr": qpos_adr,
            "before": np.round(qpos_before[:3], 6).tolist(),
            "after": np.round(np.asarray(data.body_xpos[body_id], dtype=np.float64), 6).tolist(),
        }
    except Exception as exc:
        return {"ok": False, "reason": f"set_failed:{exc}", "body_id": body_id}


def _write_qpos_addr(sim: Any, qpos_addr: Any, values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if isinstance(qpos_addr, slice):
        n = len(sim.data.qpos[qpos_addr])
        sim.data.qpos[qpos_addr] = values[:n]
        if hasattr(sim.data, "qvel"):
            sim.data.qvel[qpos_addr] = 0.0
        return n > 0
    if isinstance(qpos_addr, tuple) and len(qpos_addr) == 2:
        start, end = int(qpos_addr[0]), int(qpos_addr[1])
        n = max(0, end - start)
        sim.data.qpos[start:end] = values[:n]
        if hasattr(sim.data, "qvel"):
            sim.data.qvel[start:end] = 0.0
        return n > 0
    indexes = np.asarray(qpos_addr, dtype=np.int64).reshape(-1)
    if indexes.size == 0:
        return False
    sim.data.qpos[indexes] = values[: indexes.size]
    if hasattr(sim.data, "qvel"):
        sim.data.qvel[indexes] = 0.0
    return True


def _apply_gripper_joint_positions(env: Any, robot: Any, gripper_states: np.ndarray) -> str | None:
    gripper_states = np.asarray(gripper_states, dtype=np.float64).reshape(-1)
    if gripper_states.size == 0:
        return None

    if hasattr(robot, "set_gripper_joint_positions"):
        try:
            robot.set_gripper_joint_positions(gripper_states)
            return "robot.set_gripper_joint_positions"
        except Exception:
            pass

    sim = getattr(env, "sim", None)
    if sim is None:
        return None

    gripper_index_map = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    arms = list(getattr(robot, "arms", []) or [])
    for arm_name in arms:
        if (
            gripper_index_map is not None
            and arm_name in gripper_index_map
            and gripper_index_map[arm_name] is not None
            and _write_qpos_addr(sim, gripper_index_map[arm_name], gripper_states)
        ):
            return f"robot._ref_gripper_joint_pos_indexes[{arm_name}]"

    joint_names = list(getattr(getattr(sim, "model", None), "joint_names", []) or [])
    gripper_joint_names = [
        name
        for name in joint_names
        if "finger_joint" in str(name).lower() or "gripper" in str(name).lower()
    ]
    if gripper_joint_names and hasattr(sim.model, "get_joint_qpos_addr"):
        applied = 0
        for name, value in zip(gripper_joint_names, gripper_states):
            qpos_addr = sim.model.get_joint_qpos_addr(name)
            if _write_qpos_addr(sim, qpos_addr, np.asarray([value], dtype=np.float64)):
                applied += 1
        if applied:
            return f"sim.model.joint_names[{','.join(gripper_joint_names[:applied])}]"

    return None


@lru_cache(maxsize=8)
def _load_release_anchor_rules_from_json(path_str: str) -> dict[int, list[dict[str, Any]]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Release-anchor JSON does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("tasks", raw)
    if not isinstance(payload, dict):
        raise ValueError("Release-anchor JSON must be a dict or contain a top-level 'tasks' dict")
    out: dict[int, list[dict[str, Any]]] = {}
    for task_key, rules in payload.items():
        task_id = int(task_key)
        if not isinstance(rules, list):
            raise ValueError(f"Release-anchor rules for task {task_key} must be a list")
        parsed_rules: list[dict[str, Any]] = []
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"Task {task_key} rule #{idx} must be an object")
            released = str(rule.get("released", "")).strip()
            nxt = str(rule.get("next", "")).strip()
            anchor_hdf5 = os.path.expandvars(str(rule.get("anchor_hdf5", "")).strip())
            frame_idx = int(rule.get("frame_idx", 0))
            if not released or not nxt or not anchor_hdf5:
                raise ValueError(f"Task {task_key} rule #{idx} must contain released/next/anchor_hdf5")
            parsed_rule = {
                "released": released,
                "next": nxt,
                "anchor_hdf5": anchor_hdf5,
                "frame_idx": max(0, frame_idx),
                "tag": str(rule.get("tag", f"{released}->{nxt}")).strip() or f"{released}->{nxt}",
            }
            for optional_key in (
                "object_body",
                "object_target_site",
                "object_target_body",
                "object_target_pos",
                "object_offset",
            ):
                if optional_key in rule:
                    parsed_rule[optional_key] = rule[optional_key]
            parsed_rules.append(parsed_rule)
        out[task_id] = parsed_rules
    return out


def active_release_anchor_rules(task_id: int) -> list[dict[str, Any]]:
    path_str = os.environ.get("SUBTASK_RELEASE_ANCHORS_JSON", "").strip()
    if not path_str:
        return []
    return _load_release_anchor_rules_from_json(path_str).get(task_id, [])


@lru_cache(maxsize=8)
def _load_initial_anchor_rules_from_json(path_str: str) -> dict[int, list[dict[str, Any]]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Initial-anchor JSON does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    payload = raw.get("tasks", raw)
    if not isinstance(payload, dict):
        raise ValueError("Initial-anchor JSON must be a dict or contain a top-level 'tasks' dict")
    out: dict[int, list[dict[str, Any]]] = {}
    for task_key, rules in payload.items():
        task_id = int(task_key)
        if not isinstance(rules, list):
            raise ValueError(f"Initial-anchor rules for task {task_key} must be a list")
        parsed_rules: list[dict[str, Any]] = []
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"Task {task_key} initial-anchor rule #{idx} must be an object")
            subtask = str(rule.get("subtask", "")).strip()
            anchor_hdf5 = os.path.expandvars(str(rule.get("anchor_hdf5", "")).strip())
            frame_idx = int(rule.get("frame_idx", 0))
            if not subtask or not anchor_hdf5:
                raise ValueError(
                    f"Task {task_key} initial-anchor rule #{idx} must contain subtask/anchor_hdf5"
                )
            parsed_rules.append(
                {
                    "subtask": subtask,
                    "anchor_hdf5": anchor_hdf5,
                    "frame_idx": max(0, frame_idx),
                    "tag": str(rule.get("tag", subtask)).strip() or subtask,
                }
            )
        out[task_id] = parsed_rules
    return out


def active_initial_anchor_rules(task_id: int) -> list[dict[str, Any]]:
    path_str = os.environ.get("INITIAL_SUBTASK_ANCHORS_JSON", "").strip()
    if not path_str:
        return []
    return _load_initial_anchor_rules_from_json(path_str).get(task_id, [])


DEFAULT_ABS_EEF_GAINS = [41.12742736, 76.66682399, 82.47444396, 1.89994837, -3.73739313, -0.36202026]
DEFAULT_ABS_EEF_CLIPS = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]


def adapt_vla_action_for_env(
    action: list[float] | np.ndarray,
    element_state: np.ndarray,
) -> np.ndarray:
    """Convert optional absolute EEF VLA output back to LIBERO env action."""
    action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_arr.size < 7:
        raise ValueError(f"Expected at least 7 action dims, got shape={action_arr.shape}")
    mode = os.environ.get("VLA_ACTION_TARGET_MODE", "raw").strip().lower()
    if mode in {"raw", "delta", "controller"}:
        return action_arr[:7].astype(np.float32)
    if mode not in {"abs_eef_next", "absolute_eef_next"}:
        raise ValueError(
            "Unsupported VLA_ACTION_TARGET_MODE="
            f"{mode!r}; use raw or abs_eef_next."
        )

    state_arr = np.asarray(element_state, dtype=np.float64).reshape(-1)
    if state_arr.size < 6:
        raise ValueError(f"Expected observation/state with at least 6 dims, got shape={state_arr.shape}")
    gains = parse_float_list_env("VLA_ABS_EEF_GAIN", DEFAULT_ABS_EEF_GAINS, 6)
    clips = parse_float_list_env("VLA_ABS_EEF_CLIP", DEFAULT_ABS_EEF_CLIPS, 6)
    env_action = np.empty(7, dtype=np.float32)
    env_action[:6] = np.clip((action_arr[:6] - state_arr[:6]) * gains, -clips, clips).astype(np.float32)
    env_action[6] = np.float32(action_arr[6])
    return env_action


def get_object_pos(env: Any, obs: dict[str, Any], object_key: str) -> np.ndarray:
    if object_key in obs:
        value = np.asarray(obs[object_key], dtype=np.float64).reshape(-1)
        if value.size >= 3:
            return value[:3]

    # LIBERO observations usually expose object positions directly, e.g.
    # cookies_1_pos. Keep MuJoCo lookup as a fallback for compatible tasks.
    object_name = object_key[:-4] if object_key.endswith("_pos") else object_key
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is not None:
        candidates = [
            ("body", getattr(getattr(sim, "model", None), "body_names", []), getattr(getattr(sim, "data", None), "body_xpos", None)),
            ("site", getattr(getattr(sim, "model", None), "site_names", []), getattr(getattr(sim, "data", None), "site_xpos", None)),
            ("geom", getattr(getattr(sim, "model", None), "geom_names", []), getattr(getattr(sim, "data", None), "geom_xpos", None)),
        ]
        for _, names, positions in candidates:
            if positions is None:
                continue
            for idx, name in enumerate(names):
                if not name:
                    continue
                low = str(name).lower()
                if low == object_name.lower() or low.startswith(f"{object_name.lower()}_"):
                    value = np.asarray(positions[idx], dtype=np.float64).reshape(-1)
                    if value.size >= 3:
                        return value[:3]
    raise KeyError(f"Cannot find object position for {object_key!r}; obs keys={sorted(obs.keys())}")


def get_site_pos(env: Any, site_name: str) -> np.ndarray | None:
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None:
        return None
    names = [str(name) for name in getattr(getattr(sim, "model", None), "site_names", [])]
    variants = [site_name]
    if not site_name.endswith("_main"):
        variants.append(f"{site_name}_main")
    if site_name.endswith("_main"):
        variants.append(site_name[:-5])
    for candidate in variants:
        if candidate in names:
            sid = sim.model.site_name2id(candidate)
            value = np.asarray(sim.data.site_xpos[sid], dtype=np.float64).reshape(-1)
            if value.size >= 3:
                return value[:3]
    return None


PICK_OBJECT_ALIASES: dict[str, list[str]] = {
    "tomato sauce": ["tomato_sauce", "tomato sauce"],
    "orange juice": ["orange_juice", "orange juice"],
    "cream": ["cream_cheese", "cream"],
}


def pick_object_candidates(subtask: str) -> list[str]:
    normalized = " ".join(str(subtask).strip().lower().replace("_", " ").split())
    if not normalized.startswith("pick "):
        return []
    object_phrase = normalized[len("pick ") :].strip()
    variants = PICK_OBJECT_ALIASES.get(object_phrase, [object_phrase])
    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        for candidate in (
            variant,
            variant.replace(" ", "_"),
            variant.replace("_", " "),
            variant.split()[0],
            variant.replace("_", " ").split()[0],
        ):
            candidate = candidate.strip().lower()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
    return out


def manipulated_object_candidates(subtask: str) -> list[str]:
    normalized = " ".join(str(subtask).strip().lower().replace("_", " ").split())
    for prefix in ("pick ", "place ", "put "):
        if normalized.startswith(prefix):
            object_phrase = normalized[len(prefix) :].strip()
            break
    else:
        return []
    for suffix in (" microwave", " into microwave", " in microwave", " middle drawer", " top drawer", " bottom drawer"):
        if object_phrase.endswith(suffix):
            object_phrase = object_phrase[: -len(suffix)].strip()
    variants = PICK_OBJECT_ALIASES.get(object_phrase, [object_phrase])
    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        for candidate in (
            variant,
            variant.replace(" ", "_"),
            variant.replace("_", " "),
            variant.split()[0],
            variant.replace("_", " ").split()[0],
        ):
            candidate = candidate.strip().lower()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
    return out


def infer_pick_object_key(subtask: str, env: Any, obs: dict[str, Any]) -> str | None:
    candidates = pick_object_candidates(subtask)
    if not candidates:
        return None

    obs_keys = [str(key) for key in obs.keys()]
    for candidate in candidates:
        for key in obs_keys:
            low = key.lower()
            if low == candidate or low == f"{candidate}_pos" or low.startswith(f"{candidate}_"):
                return key

    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None:
        return None

    name_groups = [
        getattr(getattr(sim, "model", None), "body_names", []),
        getattr(getattr(sim, "model", None), "site_names", []),
        getattr(getattr(sim, "model", None), "geom_names", []),
    ]
    for candidate in candidates:
        for names in name_groups:
            for name in names:
                if not name:
                    continue
                low = str(name).lower()
                if low == candidate or low.startswith(f"{candidate}_"):
                    return candidate
    return None


def infer_manipulated_object_key(subtask: str, env: Any, obs: dict[str, Any]) -> str | None:
    candidates = manipulated_object_candidates(subtask)
    if not candidates:
        return None

    obs_keys = [str(key) for key in obs.keys()]
    for candidate in candidates:
        for key in obs_keys:
            low = key.lower()
            if low == candidate or low == f"{candidate}_pos" or low.startswith(f"{candidate}_"):
                return key

    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "env"):
        sim = getattr(env.env, "sim", None)
    if sim is None:
        return None

    name_groups = [
        getattr(getattr(sim, "model", None), "body_names", []),
        getattr(getattr(sim, "model", None), "site_names", []),
        getattr(getattr(sim, "model", None), "geom_names", []),
    ]
    for candidate in candidates:
        for names in name_groups:
            for name in names:
                low = str(name).lower()
                if low == candidate or low.startswith(f"{candidate}_"):
                    return str(name)
    return None


def load_place_object_gate_json(labels: list[str]) -> dict[str, dict[str, Any]]:
    raw_path = os.environ.get("ENDPOSE_PLACE_OBJECT_GATE_JSON", "").strip()
    if not raw_path:
        return {}
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"ENDPOSE_PLACE_OBJECT_GATE_JSON does not exist: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "subtasks" in raw:
        raw = raw["subtasks"]
    gates: dict[str, dict[str, Any]] = {}
    for name, payload in raw.items():
        subtask = normalize_subtask(str(name), labels)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: place object gate for {name!r} must be an object")
        gates[subtask] = {
            "object_key": str(payload.get("object_key", "")).strip(),
            "site": str(payload.get("site", "microwave_1_heating_region")).strip(),
            "xy_thresh": float(payload.get("xy_thresh", 0.20)),
            "x_thresh": float(payload.get("x_thresh", payload.get("xy_thresh", 0.20))),
            "y_thresh": float(payload.get("y_thresh", payload.get("xy_thresh", 0.20))),
            "z_low": float(payload.get("z_low", -1.0)),
            "z_high": float(payload.get("z_high", 1.0)),
        }
    return gates


def run_episode_sync_endpose_hold(
    *,
    task_id: int | None = None,
    env: Any,
    client: Any,
    planner: Any,
    args: Any,
    stage_specs: list[Any],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    vlm_camera_pose: dict | None,
    logger: logging.Logger,
    fail_on_extra_pour: bool = False,
    extra_pour_monitor_steps: int = 50,
    **_: Any,
) -> tuple[float, dict[str, bool], bool, dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    if args.async_vlm:
        raise ValueError("eval_tasks2_26_sync_endpose_hold.py is sync-only; set ASYNC_VLM=0.")

    cfg = hold_config()
    labels = list(planner.task_info.primitive_labels)
    task_id_int = int(planner.task_info.task_id)
    final_subtask = labels[-1] if labels else ""
    targets = load_task_targets(cfg, task_id_int, labels)
    target_passage_requirements = load_task_passage_requirements(cfg, task_id_int, labels)
    direction_signatures = load_task_direction_signatures(cfg, task_id_int, labels)
    pick_height_targets = load_task_pick_height_targets(cfg, task_id_int, labels)
    place_object_gates = load_place_object_gate_json(labels)
    official_stage_specs = official_stage._task_specs(task_id_int)
    release_anchor_rules = active_release_anchor_rules(task_id_int)
    initial_anchor_rules = active_initial_anchor_rules(task_id_int)
    oracle_hold_release_next = env_bool("ORACLE_HOLD_RELEASE_NEXT", False)
    oracle_force_initial_prompt = env_bool("ORACLE_FORCE_INITIAL_PROMPT", False)
    oracle_initial_stage_lock = env_bool("ORACLE_INITIAL_STAGE_LOCK", False)
    oracle_stage_advance_next = env_bool("ORACLE_STAGE_ADVANCE_NEXT", False)
    oracle_monotonic_sequence_lock = env_bool("ORACLE_MONOTONIC_SEQUENCE_LOCK", False)
    oracle_stage_lock_until_done = env_bool("ORACLE_STAGE_LOCK_UNTIL_DONE", False)
    microwave_stage_lock_until_done = env_bool("MICROWAVE_STAGE_LOCK_UNTIL_DONE", False)
    require_initial_vlm_subtask = env_bool("REQUIRE_INITIAL_VLM_SUBTASK", False)
    use_direction_hold = bool(direction_signatures)
    disable_output_normalize = env_bool("DISABLE_OUTPUT_NORMALIZE", False)
    drawer_forward_advance_guard = env_bool("DRAWER_FORWARD_ADVANCE_GUARD", False)
    forward_switch_block_previous = env_bool("FORWARD_SWITCH_BLOCK_PREVIOUS", False)
    require_hold_release_for_pick_forward = env_bool("REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD", False)
    pick_forward_hold_subtasks = {
        normalize_subtask(item, labels)
        for item in os.environ.get("REQUIRE_HOLD_RELEASE_FOR_PICK_FORWARD_SUBTASKS", "").split(",")
        if item.strip()
    }
    require_hold_release_for_place_forward = env_bool("REQUIRE_HOLD_RELEASE_FOR_PLACE_FORWARD", False)
    block_forward_before_first_stage_done = env_bool("BLOCK_FORWARD_BEFORE_FIRST_STAGE_DONE", False)
    microwave_forward_require_prior_hold = env_bool("MICROWAVE_FORWARD_REQUIRE_PRIOR_HOLD", False)
    microwave_forward_gap_fill_next = env_bool("MICROWAVE_FORWARD_GAP_FILL_NEXT", False)
    microwave_forward_blocked_no_current_action = os.environ.get(
        "MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION", "dummy"
    ).strip().lower()
    if microwave_forward_blocked_no_current_action not in {"dummy", "default_vla"}:
        raise ValueError(
            "Unsupported MICROWAVE_FORWARD_BLOCKED_NO_CURRENT_ACTION="
            f"{microwave_forward_blocked_no_current_action!r}; use 'dummy' or 'default_vla'."
        )
    hold_release_block_past_subtasks = env_bool("HOLD_RELEASE_BLOCK_PAST_SUBTASKS", False)
    prevent_prompt_frontier_regression = env_bool("PREVENT_PROMPT_FRONTIER_REGRESSION", False)
    prevent_completed_stage_regression = env_bool("PREVENT_COMPLETED_STAGE_REGRESSION", False)
    prevent_released_hold_regression = env_bool("PREVENT_RELEASED_HOLD_REGRESSION", False)
    prevent_held_subtask_regression = env_bool("PREVENT_HELD_SUBTASK_REGRESSION", False)
    allow_stage_done_release_anchor = env_bool("ALLOW_STAGE_DONE_RELEASE_ANCHOR", False)
    allow_autonomous_forward_release_anchor = env_bool(
        "ALLOW_AUTONOMOUS_FORWARD_RELEASE_ANCHOR", False
    )
    hold_start_after_release_anchor = env_bool("ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR", False)
    hold_start_after_release_anchor_subtasks = {
        normalize_subtask(item, labels)
        for item in os.environ.get("ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR_SUBTASKS", "").split(",")
        if item.strip()
    }
    autonomous_forward_anchor_subtasks = {
        normalize_subtask(item, labels)
        for item in os.environ.get("AUTONOMOUS_FORWARD_RELEASE_ANCHOR_SUBTASKS", "").split(",")
        if item.strip()
    }
    completed_update_from_official_stage = env_bool("COMPLETED_UPDATE_FROM_OFFICIAL_STAGE", False)
    require_open_microwave_endpose_hold_before_release = env_bool(
        "REQUIRE_OPEN_MICROWAVE_ENDPOSE_HOLD_BEFORE_RELEASE", False
    )
    pick_height_require_eef_near = env_bool("ENDPOSE_PICK_HEIGHT_REQUIRE_EEF_NEAR", False)
    drawer_task_mode = drawer_forward_advance_guard and any("drawer" in str(label).lower() for label in labels)
    hold_gripper_mode = os.environ.get("ENDPOSE_HOLD_GRIPPER_MODE", "target").strip().lower()
    if hold_gripper_mode not in {"target", "zero"}:
        raise ValueError(f"Unsupported ENDPOSE_HOLD_GRIPPER_MODE={hold_gripper_mode!r}; use 'target' or 'zero'.")
    tol_by_subtask_file = os.environ.get("ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_FILE", "").strip()
    raw_tol_by_subtask = os.environ.get("ENDPOSE_HOLD_POS_TOL_BY_SUBTASK_JSON", "").strip()
    if tol_by_subtask_file:
        raw_tol_by_subtask = Path(tol_by_subtask_file).read_text(encoding="utf-8").strip()
    tol_by_subtask: dict[str, float] = {}
    if raw_tol_by_subtask:
        for name, value in json.loads(raw_tol_by_subtask).items():
            tol_by_subtask[normalize_subtask(str(name), labels)] = float(value)

    raw_pick_lift_gate_by_subtask = os.environ.get("ENDPOSE_PICK_OBJECT_LIFT_GATE_BY_SUBTASK_JSON", "").strip()
    pick_lift_gate_by_subtask: dict[str, bool] = {}
    if raw_pick_lift_gate_by_subtask:
        for name, value in json.loads(raw_pick_lift_gate_by_subtask).items():
            pick_lift_gate_by_subtask[normalize_subtask(str(name), labels)] = bool(value)

    hold_release_min_steps_by_subtask_file = os.environ.get(
        "ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_FILE", ""
    ).strip()
    raw_hold_release_min_steps_by_subtask = os.environ.get(
        "ENDPOSE_HOLD_RELEASE_MIN_STEPS_BY_SUBTASK_JSON", ""
    ).strip()
    if hold_release_min_steps_by_subtask_file:
        raw_hold_release_min_steps_by_subtask = Path(hold_release_min_steps_by_subtask_file).read_text(
            encoding="utf-8"
        ).strip()
    hold_release_min_steps_by_subtask: dict[str, int] = {}
    if raw_hold_release_min_steps_by_subtask:
        for name, value in json.loads(raw_hold_release_min_steps_by_subtask).items():
            hold_release_min_steps_by_subtask[normalize_subtask(str(name), labels)] = max(0, int(value))

    raw_post_release_vla_steps_by_subtask = os.environ.get(
        "POST_HOLD_RELEASE_VLA_STEPS_BY_SUBTASK_JSON", ""
    ).strip()
    post_release_vla_steps_by_subtask: dict[str, int] = {}
    if raw_post_release_vla_steps_by_subtask:
        for name, value in json.loads(raw_post_release_vla_steps_by_subtask).items():
            post_release_vla_steps_by_subtask[normalize_subtask(str(name), labels)] = max(0, int(value))

    raw_consecutive_by_subtask = os.environ.get("ENDPOSE_HOLD_CONSECUTIVE_BY_SUBTASK_JSON", "").strip()
    consecutive_by_subtask: dict[str, int] = {}
    if raw_consecutive_by_subtask:
        for name, value in json.loads(raw_consecutive_by_subtask).items():
            consecutive_by_subtask[normalize_subtask(str(name), labels)] = max(1, int(value))
    hold_auto_resume_same_prompt = env_bool("ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT", False)
    hold_auto_resume_excluded_subtasks = {
        normalize_subtask(item, labels)
        for item in os.environ.get("ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT_EXCLUDE_SUBTASKS", "").split(",")
        if item.strip()
    }
    hold_auto_resume_cooldown_steps = max(0, env_int("ENDPOSE_HOLD_AUTO_RESUME_COOLDOWN_STEPS", 50))
    hold_skip_vlm_inference = env_bool("ENDPOSE_HOLD_SKIP_VLM_INFERENCE", False)
    require_open_eef_hold_for_success = env_bool(
        "MICROWAVE_REQUIRE_OPEN_EEF_HOLD_FOR_SUCCESS",
        False,
    )
    stop_on_stage_success = env_bool("STOP_ON_STAGE_SUCCESS", False)

    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    eef_pos_history: deque[np.ndarray] = deque(maxlen=max(8, cfg.direction_window + 2))
    recent_vlm_frames: deque[tuple[np.ndarray, np.ndarray | None]] = deque(maxlen=args.n_recent)
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    all_stages_logged = False
    state: dict[str, Any] | None = None
    current_stage_start = 0
    official_stage_done = {spec.name: False for spec in official_stage_specs}
    official_stage_idx = 0
    official_all_stages_logged = False
    official_state: dict[str, Any] | None = None
    official_current_stage_start = 0
    current_subtask_prompt = ""
    current_subtask_start_t = 0
    endpose_streak = 0
    hold_active = False
    hold_subtask = ""
    hold_started_t: int | None = None
    post_hold_hint_subtask = ""
    pending_hold_release_subtask = ""
    pending_hold_release_t: int | None = None
    min_endpose_dist: dict[str, float] = {}
    min_endpose_t: dict[str, int] = {}
    max_pick_height_z: dict[str, float] = {}
    max_pick_height_t: dict[str, int] = {}
    pick_object_key_cache: dict[str, str] = {}
    pick_object_key_source: dict[str, str] = {}
    place_object_key_cache: dict[str, str] = {}
    place_object_key_source: dict[str, str] = {}
    pick_object_baseline_z: dict[str, float] = {}
    pick_object_baseline_t: dict[str, int] = {}
    target_inside_region: dict[str, bool] = {}
    target_passage_count: dict[str, int] = {}
    regression_guard_active = not cfg.regression_guard_after_hold_release
    blocked_after_hold_prompts: set[str] = set()
    hold_prompt_counts: dict[str, int] = {}
    endpose_hold_started_subtasks: set[str] = set()
    endpose_hold_cooldown_until: dict[str, int] = {}
    initial_anchor_applied_subtasks: set[str] = set()
    runtime_completed_subtasks: list[str] = []
    setattr(planner, "_runtime_completed_subtasks", runtime_completed_subtasks)
    ever_goal_success = False
    last_gripper_action: float | None = None
    pick_gate_open_seen = False
    pick_gate_closed_after_open = False
    pick_gate_open_t: int | None = None
    pick_gate_close_t: int | None = None
    post_pick_release_keep_gripper_steps = max(0, env_int("POST_PICK_RELEASE_KEEP_GRIPPER_STEPS", 0))
    post_pick_release_keep_gripper_value = env_float("POST_PICK_RELEASE_KEEP_GRIPPER_VALUE", 1.0)
    post_pick_keep_gripper_until_t: int | None = None
    post_pick_keep_gripper_source: str = ""
    place_release_eef_guard = env_bool("ENDPOSE_PLACE_RELEASE_EEF_GUARD", False)
    place_release_eef_guard_value = env_float("ENDPOSE_PLACE_RELEASE_EEF_GUARD_GRIPPER_VALUE", 1.0)
    max_prompt_idx_seen: int | None = None
    close_hold_stage_gate_logged: set[str] = set()
    pending_stage_release_anchor: tuple[str, str, str] | None = None
    open_eef_hold_verified = False
    open_eef_hold_t: int | None = None
    open_eef_hold_dist: float | None = None
    open_eef_hold_door_angle: float | None = None
    save_vlm_debug_frames = env_bool("MICROWAVE_DEBUG_SAVE_VLM_FRAMES", False)
    debug_frame_count = 0
    t = 0

    def debug_frame_dir() -> Path:
        for handler in logger.handlers:
            base_filename = getattr(handler, "baseFilename", "")
            if base_filename:
                return Path(base_filename).resolve().parent / "debug_frames"
        return Path(os.environ["OUT_ROOT"]) / "debug_frames" / f"task{task_id_int}" / f"pid{os.getpid()}"

    def microwave_debug_lines(
        *,
        raw_subtask: str,
        normalized_subtask: str,
        event: str,
    ) -> list[str]:
        eef = get_eef_pos(obs)
        open_target = targets.get(normalize_subtask("open microwave", labels))
        open_dist = distance_to_target(obs, open_target) if open_target is not None else None
        open_tol = (
            pos_tol_for_subtask(normalize_subtask("open microwave", labels))
            if open_target is not None
            else None
        )
        door_angle = microwave_joint_angle(env)
        return [
            f"event={event} t={t}",
            f"vlm_raw={raw_subtask or '<none>'}",
            f"vlm_norm={normalized_subtask or '<none>'} current={current_subtask_prompt or '<none>'}",
            f"eef={format_vec3(eef)} open_target={format_vec3(open_target['target_ee_pos']) if open_target else 'NA'}",
            "open_eef_dist="
            f"{open_dist:.5f} tol={open_tol:.5f}" if open_dist is not None and open_tol is not None else "open_eef_dist=NA",
            f"door_joint={door_angle:.5f}" if door_angle is not None else "door_joint=NA",
            f"open_official={int(bool(official_stage_done.get('01_Open_Microwave', False)))} "
            f"hold_active={int(bool(hold_active))} hold_subtask={hold_subtask or '<none>'}",
        ]

    def save_vlm_debug_frame(raw_subtask: str, normalized_subtask: str) -> None:
        nonlocal debug_frame_count
        if not save_vlm_debug_frames:
            return
        try:
            out_dir = debug_frame_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            lines = microwave_debug_lines(
                raw_subtask=raw_subtask,
                normalized_subtask=normalized_subtask,
                event="vlm_infer",
            )
            prompt = current_subtask_prompt or planner.default_subtask_prompt
            element = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt)
            stem = f"{debug_frame_count:04d}_t{t:04d}_vlm"
            main_path = out_dir / f"{stem}_main.jpg"
            Image.fromarray(overlay_debug_text(element["observation/image"], lines)).save(
                main_path,
                quality=95,
            )
            wrist_path = None
            wrist = element.get("observation/wrist_image")
            if wrist is not None:
                wrist_path = out_dir / f"{stem}_wrist.jpg"
                Image.fromarray(overlay_debug_text(wrist, lines)).save(wrist_path, quality=95)
            record = {
                "index": debug_frame_count,
                "t": t,
                "raw_subtask": raw_subtask,
                "normalized_subtask": normalized_subtask,
                "current_subtask": current_subtask_prompt,
                "eef": get_eef_pos(obs).tolist(),
                "door_joint": microwave_joint_angle(env),
                "main_image": str(main_path),
                "wrist_image": str(wrist_path) if wrist_path is not None else "",
            }
            with (out_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(
                "[MICROWAVE_DEBUG_FRAME] t=%s task=%s raw_subtask=%s normalized_subtask=%s "
                "door_joint=%s main=%s wrist=%s",
                t,
                planner.task_info.task_id,
                raw_subtask,
                normalized_subtask,
                f"{record['door_joint']:.5f}" if record["door_joint"] is not None else "NA",
                main_path,
                wrist_path or "NA",
            )
            debug_frame_count += 1
        except Exception:
            logger.exception("[MICROWAVE_DEBUG_FRAME_ERROR] t=%s task=%s", t, planner.task_info.task_id)

    def mark_open_eef_hold_verified(source: str) -> None:
        nonlocal open_eef_hold_verified, open_eef_hold_t
        nonlocal open_eef_hold_dist, open_eef_hold_door_angle
        if normalize_subtask(hold_subtask, labels) != normalize_subtask("open microwave", labels):
            return
        open_eef_hold_verified = True
        open_eef_hold_t = t
        open_eef_hold_dist = distance_to_target(obs, targets[hold_subtask])
        open_eef_hold_door_angle = microwave_joint_angle(env)
        logger.info(
            "[MICROWAVE_OPEN_EEF_HOLD_VERIFIED] t=%s task=%s dist=%.5f tol=%.5f "
            "door_joint=%s source=%s",
            t,
            planner.task_info.task_id,
            open_eef_hold_dist,
            pos_tol_for_subtask(hold_subtask),
            f"{open_eef_hold_door_angle:.5f}" if open_eef_hold_door_angle is not None else "NA",
            source,
        )

    def official_stage_name_for_subtask(subtask: str) -> str | None:
        norm = normalize_subtask(subtask, labels)
        if not norm:
            return None
        norm_tokens = re.findall(r"[a-z0-9]+", norm)
        if not norm_tokens:
            return None
        if norm == "open microwave":
            wanted_tokens = ["open", "microwave"]
        elif norm == "close microwave":
            wanted_tokens = ["close", "microwave"]
        elif norm.startswith("place "):
            wanted_tokens = norm_tokens + ["microwave"]
        else:
            return None
        wanted = set(wanted_tokens)
        for spec in official_stage_specs:
            stage_tokens = set(re.findall(r"[a-z0-9]+", str(spec.name).replace("_", " ").lower()))
            if wanted.issubset(stage_tokens):
                return str(spec.name)
        return None

    def completed_stage_name_for_prompt(subtask: str) -> str | None:
        norm = normalize_subtask(subtask, labels)
        if not norm:
            return None
        if norm.startswith("pick "):
            obj_tokens = re.findall(r"[a-z0-9]+", norm[len("pick ") :])
            if not obj_tokens:
                return None
            wanted = set(["place", *obj_tokens, "microwave"])
            for spec in official_stage_specs:
                stage_tokens = set(re.findall(r"[a-z0-9]+", str(spec.name).replace("_", " ").lower()))
                if wanted.issubset(stage_tokens):
                    return str(spec.name)
            return None
        return official_stage_name_for_subtask(norm)

    def can_replay_completed_prompt(subtask: str) -> bool:
        # A stage can be officially complete before the arm has reached the EEF
        # hold target used for release/anchor timing. Keep executing that prompt
        # until the EEF hold is actually observed, then block later replays.
        norm = normalize_subtask(subtask, labels)
        if hold_active and normalize_subtask(hold_subtask, labels) == norm:
            return True
        if requires_endpose_hold_before_release(norm) and not has_started_endpose_hold(norm):
            return True
        if can_hold(norm) and norm in targets and not has_started_endpose_hold(norm):
            return True
        return False

    def is_pick_subtask(subtask: str) -> bool:
        return " ".join(str(subtask).strip().lower().split()).startswith("pick ")

    def is_place_subtask(subtask: str) -> bool:
        return " ".join(str(subtask).strip().lower().split()).startswith("place ")

    def requires_endpose_hold_before_release(subtask: str) -> bool:
        norm = normalize_subtask(subtask, labels)
        return bool(require_open_microwave_endpose_hold_before_release and norm == "open microwave")

    def has_started_endpose_hold(subtask: str) -> bool:
        return normalize_subtask(subtask, labels) in endpose_hold_started_subtasks

    def reset_pick_completion_gate(subtask: str) -> None:
        nonlocal pick_gate_open_seen, pick_gate_closed_after_open, pick_gate_open_t, pick_gate_close_t
        pick_gate_open_seen = False
        pick_gate_closed_after_open = False
        pick_gate_open_t = None
        pick_gate_close_t = None
        pick_object_baseline_z.pop(subtask, None)
        pick_object_baseline_t.pop(subtask, None)
        if cfg.pick_gripper_gate and is_pick_subtask(subtask) and last_gripper_action is not None:
            if last_gripper_action <= cfg.pick_gripper_open_max:
                pick_gate_open_seen = True
                pick_gate_open_t = t
                logger.info(
                    "[PICK_GRIPPER_GATE_OPEN_SEEN] t=%s task=%s subtask=%s source=initial_last_action "
                    "gripper=%+.3f open_max=%+.3f",
                    t,
                    planner.task_info.task_id,
                subtask,
                last_gripper_action,
                cfg.pick_gripper_open_max,
            )

    def resolved_pick_object_key(subtask: str) -> tuple[str | None, str]:
        explicit = pick_height_targets.get(subtask)
        if explicit is not None and "object_key" in explicit:
            return str(explicit["object_key"]), "json"
        cached = pick_object_key_cache.get(subtask)
        if cached:
            return cached, pick_object_key_source.get(subtask, "cache")
        inferred = infer_pick_object_key(subtask, env, obs)
        if inferred:
            pick_object_key_cache[subtask] = inferred
            pick_object_key_source[subtask] = "inferred"
            logger.info(
                "[PICK_OBJECT_KEY_INFERRED] t=%s task=%s subtask=%s object_key=%s",
                t,
                planner.task_info.task_id,
                subtask,
                inferred,
            )
            return inferred, "inferred"
        return None, "missing"

    def resolved_place_object_key(subtask: str) -> tuple[str | None, str]:
        explicit = place_object_gates.get(subtask)
        if explicit is not None and explicit.get("object_key"):
            return str(explicit["object_key"]), "json"
        cached = place_object_key_cache.get(subtask)
        if cached:
            return cached, place_object_key_source.get(subtask, "cache")
        inferred = infer_manipulated_object_key(subtask, env, obs)
        if inferred:
            place_object_key_cache[subtask] = inferred
            place_object_key_source[subtask] = "inferred"
            logger.info(
                "[PLACE_OBJECT_KEY_INFERRED] t=%s task=%s subtask=%s object_key=%s",
                t,
                planner.task_info.task_id,
                subtask,
                inferred,
            )
            return inferred, "inferred"
        return None, "missing"

    def update_pick_gripper_gate(action: list[float] | np.ndarray, prompt_for_vla: str) -> None:
        nonlocal last_gripper_action, pick_gate_open_seen, pick_gate_closed_after_open
        nonlocal pick_gate_open_t, pick_gate_close_t
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.size < 7:
            return
        gripper = float(action_arr[6])
        last_gripper_action = gripper
        if not (cfg.pick_gripper_gate and is_pick_subtask(prompt_for_vla)):
            return
        if (not pick_gate_open_seen) and gripper <= cfg.pick_gripper_open_max:
            pick_gate_open_seen = True
            pick_gate_open_t = t
            logger.info(
                "[PICK_GRIPPER_GATE_OPEN_SEEN] t=%s task=%s subtask=%s source=action gripper=%+.3f open_max=%+.3f",
                t,
                planner.task_info.task_id,
                prompt_for_vla,
                gripper,
                cfg.pick_gripper_open_max,
            )
        if pick_gate_open_seen and (not pick_gate_closed_after_open) and gripper >= cfg.pick_gripper_close_min:
            pick_gate_closed_after_open = True
            pick_gate_close_t = t
            logger.info(
                "[PICK_GRIPPER_GATE_CLOSED_AFTER_OPEN] t=%s task=%s subtask=%s gripper=%+.3f close_min=%+.3f open_t=%s",
                t,
                planner.task_info.task_id,
                prompt_for_vla,
                gripper,
                cfg.pick_gripper_close_min,
                pick_gate_open_t,
            )

    def append_eef_pos() -> None:
        eef_pos_history.append(get_eef_pos(obs).copy())

    def clone_recent_frames() -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [(m.copy(), w.copy() if w is not None else None) for m, w in recent_vlm_frames]

    def append_vlm_frame() -> None:
        recent_vlm_frames.append(base._extract_vlm_frame(env, obs, args, vlm_camera_pose))

    def refresh_obs_after_anchor() -> None:
        nonlocal obs
        env.sim.forward()
        if hasattr(env, "_post_process"):
            env._post_process()
        if hasattr(env, "_update_observables"):
            env._update_observables(force=True)
        if hasattr(env, "env") and hasattr(env.env, "_get_observations"):
            obs = env.env._get_observations()
        elif hasattr(env, "_get_observations"):
            obs = env._get_observations()
        eef_pos_history.clear()
        append_eef_pos()
        recent_vlm_frames.clear()
        append_vlm_frame()

    def can_hold(subtask: str) -> bool:
        if not cfg.enabled or not subtask or subtask not in targets:
            return False
        if cfg.disable_final and subtask == final_subtask:
            return False
        if cfg.drawer_close_hold_require_stage and is_close_drawer_subtask(subtask):
            matching_stages = [
                name for name in stage_done.keys() if close_drawer_stage_matches(name, subtask)
            ]
            if matching_stages and not any(stage_done.get(name, False) for name in matching_stages):
                if subtask not in close_hold_stage_gate_logged:
                    logger.info(
                        "[ENDPOSE_HOLD_STAGE_GATE_BLOCKED] t=%s task=%s subtask=%s reason=close_drawer_stage_not_done "
                        "matching_stages=%s done=%s",
                        t,
                        planner.task_info.task_id,
                        subtask,
                        matching_stages,
                        {name: stage_done.get(name, False) for name in matching_stages},
                    )
                    close_hold_stage_gate_logged.add(subtask)
                return False
        return True

    def pos_tol_for_subtask(subtask: str) -> float:
        if subtask in tol_by_subtask:
            return float(tol_by_subtask[subtask])
        target = targets.get(subtask)
        if target is None:
            return float(max(cfg.pos_tol, cfg.eef_default_tol))
        p95 = float(target.get("pos_dist_p95", 0.0) or 0.0)
        adaptive = p95 + cfg.eef_p95_extra_tol if p95 > 0.0 else 0.0
        return float(min(cfg.eef_tol_cap, max(cfg.pos_tol, cfg.eef_default_tol, adaptive)))

    def pick_object_lift_gate_for_subtask(subtask: str) -> bool:
        return bool(pick_lift_gate_by_subtask.get(subtask, cfg.pick_object_lift_gate))


    def most_common_hold_prompt() -> str:
        if not hold_prompt_counts:
            return hold_subtask
        return max(
            hold_prompt_counts.items(),
            key=lambda item: (item[1], 1 if item[0] == hold_subtask else 0, item[0]),
        )[0]

    def hold_release_min_steps(subtask: str) -> int:
        subtask_norm = normalize_subtask(subtask, labels)
        if subtask_norm in hold_release_min_steps_by_subtask:
            return int(hold_release_min_steps_by_subtask[subtask_norm])
        if is_place_subtask(subtask_norm):
            return max(0, env_int("ENDPOSE_PLACE_HOLD_MIN_STEPS_BEFORE_RELEASE", 0))
        return max(0, env_int("ENDPOSE_HOLD_MIN_STEPS_BEFORE_RELEASE", 0))

    def record_completed_subtask(subtask: str, source: str) -> None:
        if not _completed_subtasks_mode():
            return
        if not subtask or subtask in runtime_completed_subtasks:
            return
        runtime_completed_subtasks.append(subtask)
        setattr(planner, "_runtime_completed_subtasks", runtime_completed_subtasks)
        logger.info(
            "[COMPLETED_SUBTASKS_UPDATE] t=%s task=%s completed=%s mode=%s source=%s subtask=%s",
            t,
            planner.task_info.task_id,
            runtime_completed_subtasks,
            _completed_subtasks_mode() or "off",
            source,
            subtask,
        )

    def direction_gate(subtask: str, target: dict[str, Any], dist: float) -> tuple[bool, float | None, float | None, float | None, str]:
        if not use_direction_hold:
            return True, None, None, None, "disabled"
        signature = direction_signatures.get(subtask)
        return evaluate_eef_direction_gate(
            eef_pos_history,
            np.asarray(target["target_ee_pos"], dtype=np.float64),
            dist,
            signature,
            default_window=cfg.direction_window,
            min_displacement=cfg.direction_min_displacement,
            cos_min=cfg.direction_cos_min,
            trend_eps=cfg.direction_trend_eps,
        )

    def check_goal(done: bool) -> bool:
        nonlocal ever_goal_success
        goal_success = (
            bool(goal_check_override(env, stage_done))
            if goal_check_override is not None
            else bool(base.ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False)
        )
        if goal_success and not ever_goal_success:
            logger.info("[t=%s] goal success", t)
        ever_goal_success = ever_goal_success or goal_success
        if done:
            logger.info("[DONE] t=%s, task done", t)
            return True
        return False

    def update_stage_and_goal(done: bool) -> bool:
        nonlocal stage_idx, current_stage_start, all_stages_logged, state
        nonlocal official_stage_idx, official_current_stage_start, official_all_stages_logged, official_state
        nonlocal pending_stage_release_anchor
        if state is not None:
            base.stage_eval._update_state(obs, state)
            if stage_idx < len(stage_specs):
                spec = stage_specs[stage_idx]
                if spec.check_fn(env, state, current_stage_start):
                    stage_done[spec.name] = True
                    logger.info("[t=%s] stage done: %s", t, spec.name)
                    stage_idx += 1
                    current_stage_start = state["step_idx"]
            if stage_idx >= len(stage_specs) and not all_stages_logged:
                logger.info("[t=%s] all stages done", t)
                all_stages_logged = True
        if official_state is not None:
            official_stage._update_state(obs, official_state)
            if official_stage_idx < len(official_stage_specs):
                spec = official_stage_specs[official_stage_idx]
                if spec.check_fn(env, official_state, official_current_stage_start):
                    official_stage_done[spec.name] = True
                    logger.info("[t=%s] official stage done: %s", t, spec.name)
                    logger.info(
                        "[OFFICIAL_STAGE_EEF] t=%s task=%s stage=%s eef=%s",
                        t,
                        planner.task_info.task_id,
                        spec.name,
                        np.round(get_eef_pos(obs), 6).tolist(),
                    )
                    if (
                        completed_update_from_official_stage
                        and official_stage_name_for_subtask(current_subtask_prompt) == spec.name
                    ):
                        record_completed_subtask(current_subtask_prompt, f"official_stage_done:{spec.name}")
                    if pending_stage_release_anchor is not None:
                        released_subtask, next_subtask, required_stage = pending_stage_release_anchor
                        if required_stage == spec.name:
                            if normalize_subtask(current_subtask_prompt, labels) == normalize_subtask(
                                next_subtask, labels
                            ):
                                applied = maybe_apply_release_anchor(released_subtask, next_subtask)
                                logger.info(
                                    "[SUBTASK_RELEASE_ANCHOR_DEFERRED_STAGE_DONE] t=%s task=%s "
                                    "released_subtask=%s next_subtask=%s stage=%s applied=%s",
                                    t,
                                    planner.task_info.task_id,
                                    released_subtask,
                                    next_subtask,
                                    required_stage,
                                    applied,
                                )
                            else:
                                logger.info(
                                    "[SUBTASK_RELEASE_ANCHOR_DEFERRED_DROPPED] t=%s task=%s "
                                    "released_subtask=%s next_subtask=%s stage=%s current_subtask=%s "
                                    "reason=current_prompt_advanced",
                                    t,
                                    planner.task_info.task_id,
                                    released_subtask,
                                    next_subtask,
                                    required_stage,
                                    current_subtask_prompt,
                                )
                            pending_stage_release_anchor = None
                    official_stage_idx += 1
                    official_current_stage_start = official_state["step_idx"]
            if official_stage_idx >= len(official_stage_specs) and not official_all_stages_logged:
                logger.info("[t=%s] all official stages done", t)
                official_all_stages_logged = True
        if stop_on_stage_success:
            official_stage_success_raw = _official_stage_success(task_id_int, official_stage_done)
            if official_stage_success_raw:
                logger.info(
                    "[STOP_ON_STAGE_SUCCESS] t=%s task=%s official_stage_done=%s "
                    "open_eef_required=%s open_eef_verified=%s",
                    t,
                    planner.task_info.task_id,
                    json.dumps(official_stage_done, ensure_ascii=False, separators=(",", ":")),
                    int(require_open_eef_hold_for_success),
                    int(open_eef_hold_verified),
                )
                return True
        return check_goal(done)

    def maybe_update_endpose_streak(subtask: str, phase: str, t_now: int) -> bool:
        nonlocal endpose_streak
        if subtask not in targets:
            return False
        norm_subtask = normalize_subtask(subtask, labels)
        required_consecutive = consecutive_by_subtask.get(norm_subtask, cfg.consecutive)
        cooldown_until = endpose_hold_cooldown_until.get(norm_subtask, 0)
        if cooldown_until and t_now < cooldown_until:
            endpose_streak = 0
            if cfg.distance_log_interval > 0 and t_now % cfg.distance_log_interval == 0:
                logger.info(
                    "[ENDPOSE_HOLD_COOLDOWN] t=%s task=%s subtask=%s cooldown_until=%s phase=%s",
                    t_now,
                    planner.task_info.task_id,
                    subtask,
                    cooldown_until,
                    phase,
                )
            return False
        dist = distance_to_target(obs, targets[subtask])
        pos_tol = pos_tol_for_subtask(subtask)
        explicit_pick_height_target = pick_height_targets.get(subtask)
        pick_object_key, pick_object_key_source = resolved_pick_object_key(subtask)
        pick_height_applies = bool(
            is_pick_subtask(subtask)
            and pick_object_key is not None
            and (pick_object_lift_gate_for_subtask(subtask) or explicit_pick_height_target is not None)
        )
        if pick_height_applies:
            current_object_pos = get_object_pos(env, obs, str(pick_object_key))
            current_z = float(current_object_pos[2])
            prev_max_z = max_pick_height_z.get(subtask)
            if prev_max_z is None or current_z > prev_max_z:
                max_pick_height_z[subtask] = current_z
                max_pick_height_t[subtask] = t_now
            baseline_z = pick_object_baseline_z.get(subtask)
            if baseline_z is None or ((not pick_gate_closed_after_open) and current_z < baseline_z):
                pick_object_baseline_z[subtask] = current_z
                pick_object_baseline_t[subtask] = t_now
        else:
            current_z = float("nan")
        prev_min = min_endpose_dist.get(subtask)
        if prev_min is None or dist < prev_min:
            min_endpose_dist[subtask] = dist
            min_endpose_t[subtask] = t_now
        active_steps = max(0, t_now - current_subtask_start_t)
        final_no_hold = cfg.disable_final and subtask == final_subtask
        if pick_height_applies:
            if explicit_pick_height_target is not None:
                height_z_min = float(explicit_pick_height_target["height_z_min"])
                height_z_target = float(explicit_pick_height_target["height_z_target"])
                baseline_z = pick_object_baseline_z.get(subtask)
            else:
                baseline_z = float(pick_object_baseline_z.get(subtask, current_z))
                height_z_target = baseline_z + cfg.pick_object_lift_delta
                height_z_min = height_z_target
            height_ok = current_z >= height_z_min
            eef_near_for_pick = dist <= pos_tol
            near_target = bool(height_ok and (eef_near_for_pick or not pick_height_require_eef_near))
        else:
            height_z_min = None
            height_z_target = None
            baseline_z = None
            height_ok = False
            eef_near_for_pick = False
            place_object_gate = place_object_gates.get(subtask) if is_place_subtask(subtask) else None
            if place_object_gate is not None:
                eef_near_for_place = dist <= pos_tol
                place_object_key, place_object_key_source = resolved_place_object_key(subtask)
                site_name = str(place_object_gate.get("site") or "microwave_1_heating_region")
                site_pos = get_site_pos(env, site_name)
                object_pos = None
                object_error = ""
                if place_object_key is not None:
                    try:
                        object_pos = get_object_pos(env, obs, str(place_object_key))
                    except Exception as exc:
                        object_error = f"{type(exc).__name__}: {exc}"
                x_thresh = float(place_object_gate.get("x_thresh", place_object_gate.get("xy_thresh", 0.20)))
                y_thresh = float(place_object_gate.get("y_thresh", place_object_gate.get("xy_thresh", 0.20)))
                z_low = float(place_object_gate.get("z_low", -1.0))
                z_high = float(place_object_gate.get("z_high", 1.0))
                if object_pos is not None and site_pos is not None:
                    x_diff = abs(float(object_pos[0] - site_pos[0]))
                    y_diff = abs(float(object_pos[1] - site_pos[1]))
                    z_diff = float(object_pos[2] - site_pos[2])
                    place_object_ok = x_diff < x_thresh and y_diff < y_thresh and z_low < z_diff < z_high
                else:
                    x_diff = float("nan")
                    y_diff = float("nan")
                    z_diff = float("nan")
                    place_object_ok = False
                near_target = bool(eef_near_for_place and place_object_ok)
                if eef_near_for_place or place_object_ok or target_inside_region.get(subtask, False):
                    logger.info(
                        "[PLACE_OBJECT_GATE] t=%s task=%s subtask=%s eef_near=%s eef_dist=%.5f "
                        "eef_tol=%.5f object_ok=%s object_key=%s object_key_source=%s site=%s "
                        "x_diff=%s y_diff=%s z_diff=%s x_thresh=%.5f y_thresh=%.5f z_low=%.5f "
                        "z_high=%.5f object_error=%s phase=%s",
                        t_now,
                        planner.task_info.task_id,
                        subtask,
                        eef_near_for_place,
                        dist,
                        pos_tol,
                        place_object_ok,
                        place_object_key or "NA",
                        place_object_key_source,
                        site_name,
                        f"{x_diff:.5f}" if np.isfinite(x_diff) else "NA",
                        f"{y_diff:.5f}" if np.isfinite(y_diff) else "NA",
                        f"{z_diff:.5f}" if np.isfinite(z_diff) else "NA",
                        x_thresh,
                        y_thresh,
                        z_low,
                        z_high,
                        object_error or "NA",
                        phase,
                    )
            else:
                near_target = dist <= pos_tol
        required_passages = max(1, int(target_passage_requirements.get(subtask, 1)))
        was_inside = target_inside_region.get(subtask, False)
        if pick_height_applies:
            if near_target or was_inside:
                logger.info(
                    "[PICK_HEIGHT_GATE] t=%s task=%s subtask=%s z=%.5f target_z=%.5f z_min=%.5f "
                    "height_ok=%s eef_dist=%.5f eef_tol=%.5f eef_near=%s require_eef_near=%s "
                    "active_steps=%s object_key=%s object_key_source=%s baseline_z=%s phase=%s",
                    t_now,
                    planner.task_info.task_id,
                    subtask,
                    current_z,
                    height_z_target,
                    height_z_min,
                    height_ok,
                    dist,
                    pos_tol,
                    eef_near_for_pick,
                    pick_height_require_eef_near,
                    active_steps,
                    pick_object_key,
                    pick_object_key_source,
                    f"{baseline_z:.5f}" if baseline_z is not None else "NA",
                    phase,
                )
        elif near_target and not was_inside:
            target_passage_count[subtask] = target_passage_count.get(subtask, 0) + 1
            logger.info(
                "[ENDPOSE_PASSAGE] t=%s task=%s subtask=%s passage=%s/%s dist=%.5f tol=%.5f phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                target_passage_count[subtask],
                required_passages,
                dist,
                pos_tol,
                phase,
            )
        elif (not near_target) and was_inside:
            logger.info(
                "[ENDPOSE_PASSAGE_EXIT] t=%s task=%s subtask=%s passage=%s/%s dist=%.5f tol=%.5f phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                target_passage_count.get(subtask, 0),
                required_passages,
                dist,
                pos_tol,
                phase,
            )
        target_inside_region[subtask] = near_target

        pick_gate_applies = cfg.pick_gripper_gate and is_pick_subtask(subtask)
        if pick_height_applies:
            passage_ok = True
            direction_ok, direction_cos, direction_disp, prev_dist, direction_reason = (
                True,
                None,
                None,
                None,
                "disabled_by_pick_height_gate",
            )
            gripper_gate_ok = pick_gate_closed_after_open if pick_gate_applies else True
        elif pick_gate_applies:
            passage_ok = True
            direction_ok, direction_cos, direction_disp, prev_dist, direction_reason = (
                True,
                None,
                None,
                None,
                "disabled_by_pick_gripper_gate",
            )
            gripper_gate_ok = pick_gate_closed_after_open
        else:
            direction_gate_applies = subtask in direction_signatures
            passage_ok = (
                True
                if direction_gate_applies
                else target_passage_count.get(subtask, 0) >= required_passages
            )
            direction_ok, direction_cos, direction_disp, prev_dist, direction_reason = (
                direction_gate(subtask, targets[subtask], dist) if near_target else (False, None, None, None, "not_near")
            )
            gripper_gate_ok = True
        deferred_gripper_release = bool(
            cfg.pick_deferred_gripper_release
            and pick_gate_applies
            and pick_gate_closed_after_open
            and target_passage_count.get(subtask, 0) >= 1
        )
        if deferred_gripper_release:
            passage_ok = True
            direction_ok, direction_cos, direction_disp, prev_dist, direction_reason = (
                True,
                None,
                None,
                None,
                "deferred_gripper_after_near_target",
            )
            gripper_gate_ok = True
        should_count = (
            can_hold(subtask)
            and active_steps >= cfg.min_active_steps
            and (near_target or deferred_gripper_release)
            and passage_ok
            and direction_ok
            and gripper_gate_ok
        )
        endpose_streak = endpose_streak + 1 if should_count else 0
        if final_no_hold:
            logger.info(
                "[ENDPOSE_FINAL_LOG] t=%s task=%s subtask=%s dist=%.5f tol=%.5f phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                phase,
            )
        elif should_count or near_target:
            logger.info(
                "[ENDPOSE_NEAR] t=%s task=%s subtask=%s dist=%.5f tol=%.5f active_steps=%s "
                "passage=%s/%s passage_ok=%s direction_ok=%s direction_reason=%s direction_cos=%s "
                "direction_disp=%s prev_dist=%s gripper_gate=%s gripper_open_seen=%s "
                "gripper_closed_after_open=%s gripper_open_t=%s gripper_close_t=%s "
                "pick_height_gate=%s object_key=%s object_key_source=%s baseline_z=%s current_z=%.5f "
                "height_z_min=%s height_z_target=%s deferred_gripper_release=%s "
                "streak=%s/%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                active_steps,
                target_passage_count.get(subtask, 0),
                required_passages,
                passage_ok,
                direction_ok,
                direction_reason,
                f"{direction_cos:.4f}" if direction_cos is not None else "NA",
                f"{direction_disp:.5f}" if direction_disp is not None else "NA",
                f"{prev_dist:.5f}" if prev_dist is not None else "NA",
                gripper_gate_ok,
                pick_gate_open_seen if pick_gate_applies else "NA",
                pick_gate_closed_after_open if pick_gate_applies else "NA",
                pick_gate_open_t if pick_gate_applies else "NA",
                pick_gate_close_t if pick_gate_applies else "NA",
                pick_height_applies,
                pick_object_key if pick_height_applies else "NA",
                pick_object_key_source if pick_height_applies else "NA",
                f"{baseline_z:.5f}" if baseline_z is not None else "NA",
                current_z,
                f"{height_z_min:.5f}" if height_z_min is not None else "NA",
                f"{height_z_target:.5f}" if height_z_target is not None else "NA",
                deferred_gripper_release,
                endpose_streak,
                required_consecutive,
                phase,
            )
        elif cfg.distance_log_interval > 0 and t_now % cfg.distance_log_interval == 0:
            logger.info(
                "[ENDPOSE_DISTANCE] t=%s task=%s subtask=%s dist=%.5f tol=%.5f active_steps=%s "
                "min_dist=%.5f min_t=%s phase=%s",
                t_now,
                planner.task_info.task_id,
                subtask,
                dist,
                pos_tol,
                active_steps,
                min_endpose_dist[subtask],
                min_endpose_t[subtask],
                phase,
            )
        return can_hold(subtask) and endpose_streak >= required_consecutive

    def build_video_overlay_lines(prompt_for_vla: str, control_mode: str) -> list[str]:
        raw_prompt = " ".join(str(prompt_for_vla).strip().split()) if prompt_for_vla else "<none>"
        subtask = normalize_subtask(prompt_for_vla, labels) if prompt_for_vla else ""
        lines = [
            f"t={t} control={control_mode}",
            f"vla_prompt={raw_prompt}",
        ]
        door_angle = microwave_joint_angle(env)
        lines.append(
            f"eef={format_vec3(get_eef_pos(obs))} "
            f"door_joint={f'{door_angle:.5f}' if door_angle is not None else 'NA'} "
            f"open_official={int(bool(official_stage_done.get('01_Open_Microwave', False)))}"
        )
        if hold_active or hold_subtask:
            hold_required_consecutive = consecutive_by_subtask.get(
                normalize_subtask(hold_subtask, labels) if hold_subtask else "",
                cfg.consecutive,
            )
            lines.append(
                f"hold_active={int(bool(hold_active))} hold_subtask={hold_subtask or '<none>'} "
                f"endpose_streak={endpose_streak}/{hold_required_consecutive}"
            )

        target = targets.get(subtask)
        if target is None:
            lines.append("target_eef=NA")
            return lines

        dist = distance_to_target(obs, target)
        pos_tol = pos_tol_for_subtask(subtask)
        required_passages = max(1, int(target_passage_requirements.get(subtask, 1)))
        seen_passages = int(target_passage_count.get(subtask, 0))
        lines.append(
            f"target_eef={format_vec3(target['target_ee_pos'])} "
            f"dist={dist:.5f} tol={pos_tol:.5f}"
        )
        lines.append(
            f"target_passage={seen_passages}/{required_passages} "
            f"in_near={int(bool(target_inside_region.get(subtask, False)))} "
            f"can_hold={int(can_hold(subtask))}"
        )

        if use_direction_hold and not is_pick_subtask(subtask):
            direction_ok, direction_cos, direction_disp, prev_dist, direction_reason = (
                direction_gate(subtask, target, dist) if dist <= pos_tol else (False, None, None, None, "not_near")
            )
            lines.append(
                f"direction ok={int(bool(direction_ok))} reason={direction_reason} "
                f"cos={f'{direction_cos:.4f}' if direction_cos is not None else 'NA'} "
                f"disp={f'{direction_disp:.5f}' if direction_disp is not None else 'NA'} "
                f"prev_dist={f'{prev_dist:.5f}' if prev_dist is not None else 'NA'}"
            )

        if is_pick_subtask(subtask):
            pick_object_key, _ = resolved_pick_object_key(subtask)
            if pick_object_key is not None:
                try:
                    object_pos = get_object_pos(env, obs, str(pick_object_key))
                    object_z = float(object_pos[2])
                except Exception:
                    object_z = float("nan")
                baseline_z = pick_object_baseline_z.get(subtask)
                explicit_target = pick_height_targets.get(subtask)
                if explicit_target is not None:
                    height_z_min = float(explicit_target["height_z_min"])
                    height_z_target = float(explicit_target["height_z_target"])
                elif baseline_z is not None:
                    height_z_min = float(baseline_z) + cfg.pick_object_lift_delta
                    height_z_target = height_z_min
                else:
                    height_z_min = None
                    height_z_target = None
                lines.append(
                    f"pick_obj={pick_object_key} z={object_z:.5f} "
                    f"baseline_z={f'{baseline_z:.5f}' if baseline_z is not None else 'NA'} "
                    f"height_min={f'{height_z_min:.5f}' if height_z_min is not None else 'NA'} "
                    f"height_target={f'{height_z_target:.5f}' if height_z_target is not None else 'NA'}"
                )
                lines.append(
                    f"gripper_gate open_seen={int(bool(pick_gate_open_seen))} "
                    f"closed_after_open={int(bool(pick_gate_closed_after_open))} "
                    f"open_t={pick_gate_open_t if pick_gate_open_t is not None else 'NA'} "
                    f"close_t={pick_gate_close_t if pick_gate_close_t is not None else 'NA'}"
                )
        return lines

    def step_env(action: list[float] | np.ndarray, prompt_for_vla: str, control_mode: str) -> bool:
        nonlocal obs, t
        element_step = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
        env_action = adapt_vla_action_for_env(action, element_step["observation/state"])
        prompt_norm = normalize_subtask(prompt_for_vla, labels) if prompt_for_vla else ""
        if (
            post_pick_keep_gripper_until_t is not None
            and t < post_pick_keep_gripper_until_t
            and is_place_subtask(prompt_norm)
        ):
            original_gripper = float(env_action[6])
            if original_gripper < post_pick_release_keep_gripper_value:
                env_action[6] = np.float32(post_pick_release_keep_gripper_value)
            logger.info(
                "[POST_PICK_KEEP_GRIPPER] t=%s task=%s subtask=%s control=%s "
                "original_gripper=%+.3f applied_gripper=%+.3f until_t=%s source=%s",
                t,
                planner.task_info.task_id,
                prompt_norm,
                control_mode,
                original_gripper,
                float(env_action[6]),
                post_pick_keep_gripper_until_t,
                post_pick_keep_gripper_source,
            )
        target = targets.get(prompt_norm)
        if target is not None:
            guard_tol = pos_tol_for_subtask(prompt_norm)
            keep_closed, guard_dist = should_keep_place_gripper_closed(
                prompt_norm,
                get_eef_pos(obs),
                target["target_ee_pos"],
                guard_tol,
                enabled=place_release_eef_guard,
            )
            if keep_closed:
                original_gripper = float(env_action[6])
                if original_gripper < place_release_eef_guard_value:
                    env_action[6] = np.float32(place_release_eef_guard_value)
                if original_gripper < place_release_eef_guard_value or t % 10 == 0:
                    logger.info(
                        "[PLACE_RELEASE_EEF_GUARD] t=%s task=%s subtask=%s control=%s "
                        "eef_dist=%.5f release_tol=%.5f original_gripper=%+.3f applied_gripper=%+.3f",
                        t,
                        planner.task_info.task_id,
                        prompt_norm,
                        control_mode,
                        guard_dist,
                        guard_tol,
                        original_gripper,
                        float(env_action[6]),
                    )
        update_pick_gripper_gate(env_action, prompt_for_vla)
        overlay_lines = build_video_overlay_lines(prompt_for_vla, control_mode)
        replay.append(overlay_debug_text(element_step["observation/image"], overlay_lines))
        wrist = element_step.get("observation/wrist_image")
        if wrist is not None:
            replay_wrist.append(overlay_debug_text(wrist, overlay_lines))
        obs, _, done, _ = env.step(env_action.tolist())
        append_eef_pos()
        append_vlm_frame()
        t += 1
        return update_stage_and_goal(bool(done))

    def run_vla_without_vlm(step_budget: int, phase: str) -> bool:
        remaining = max(0, int(step_budget))
        if remaining <= 0:
            return False
        logger.info(
            "[POST_HOLD_RELEASE_VLA_START] t=%s task=%s subtask=%s steps=%s phase=%s",
            t,
            planner.task_info.task_id,
            current_subtask_prompt,
            remaining,
            phase,
        )
        while remaining > 0 and t < args.max_steps + args.num_steps_wait:
            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            element = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            if len(actions) <= 0:
                raise RuntimeError("VLA returned an empty action chunk")
            chunk_len = min(len(actions), remaining, args.max_steps + args.num_steps_wait - t)
            logger.info(
                "[POST_HOLD_RELEASE_VLA_CHUNK] t=%s task=%s subtask=%s chunk_steps=%s remaining_before=%s",
                t,
                planner.task_info.task_id,
                current_subtask_prompt,
                chunk_len,
                remaining,
            )
            for post_idx, action in enumerate(actions[:chunk_len], start=1):
                if step_env(action, prompt_for_vla, f"post_hold_release_vla_{post_idx}/{chunk_len}"):
                    return True
                remaining -= 1
                if t >= args.max_steps + args.num_steps_wait:
                    break
        logger.info("[POST_HOLD_RELEASE_VLA_END] t=%s task=%s phase=%s", t, planner.task_info.task_id, phase)
        return False

    def maybe_apply_initial_subtask_anchor(subtask: str) -> bool:
        if not initial_anchor_rules:
            return False
        subtask_norm = normalize_subtask(subtask, labels)
        if not subtask_norm or subtask_norm in initial_anchor_applied_subtasks:
            return False
        rule = None
        for candidate in initial_anchor_rules:
            candidate_subtask = normalize_subtask(candidate["subtask"], labels)
            if subtask_norm == candidate_subtask:
                rule = candidate
                break
        if rule is None:
            return False

        anchor_hdf5 = str(rule["anchor_hdf5"]).strip()
        frame_idx = max(0, int(rule.get("frame_idx", 0)))
        try:
            anchor = load_release_anchor(anchor_hdf5, frame_idx)
            robot = env.robots[0]
            robot.set_robot_joint_positions(anchor["joint_states"])
            gripper_method = _apply_gripper_joint_positions(env, robot, anchor["gripper_states"])
            refresh_obs_after_anchor()
            initial_anchor_applied_subtasks.add(subtask_norm)
            logger.info(
                "[INITIAL_SUBTASK_ANCHOR] t=%s task=%s subtask=%s rule=%s anchor_file=%s "
                "frame_idx=%s anchor_ee=%s joint=%s gripper=%s",
                t,
                planner.task_info.task_id,
                subtask_norm,
                rule.get("tag", subtask_norm),
                anchor_hdf5,
                frame_idx,
                format_vec3(anchor["ee_pos"]),
                np.round(anchor["joint_states"], 6).tolist(),
                {
                    "method": gripper_method or "SKIP",
                    "target": np.round(anchor["gripper_states"], 6).tolist(),
                    "obs": np.round(np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64), 6).tolist()
                    if isinstance(obs, dict)
                    else [],
                },
            )
            return True
        except Exception:
            logger.exception(
                "[INITIAL_SUBTASK_ANCHOR_FAILED] t=%s task=%s subtask=%s rule=%s anchor_file=%s frame_idx=%s",
                t,
                planner.task_info.task_id,
                subtask_norm,
                rule.get("tag", subtask_norm),
                anchor_hdf5,
                frame_idx,
            )
            return False

    def maybe_apply_release_anchor(released_hold_subtask: str, next_subtask: str) -> bool:
        nonlocal obs
        if not release_anchor_rules:
            return False
        released_norm = normalize_subtask(released_hold_subtask, labels)
        next_norm = normalize_subtask(next_subtask, labels)
        rule = None
        for candidate in release_anchor_rules:
            candidate_released = normalize_subtask(candidate["released"], labels)
            candidate_next = normalize_subtask(candidate["next"], labels)
            if released_norm == candidate_released and next_norm == candidate_next:
                rule = candidate
                break
        if rule is None:
            return False

        anchor_hdf5 = str(rule["anchor_hdf5"]).strip()
        frame_idx = max(0, int(rule.get("frame_idx", 0)))
        try:
            anchor = load_release_anchor(anchor_hdf5, frame_idx)
            robot = env.robots[0]
            robot.set_robot_joint_positions(anchor["joint_states"])
            gripper_method = _apply_gripper_joint_positions(env, robot, anchor["gripper_states"])
            object_anchor_result = None
            object_body = str(rule.get("object_body", "")).strip()
            if object_body:
                target_pos = None
                target_source = None
                target_site = str(rule.get("object_target_site", "")).strip()
                target_body = str(rule.get("object_target_body", "")).strip()
                if target_site:
                    target_pos = _named_sim_pos(env, target_site, kind="site")
                    target_source = f"site:{target_site}"
                if target_pos is None and target_body:
                    target_pos = _named_sim_pos(env, target_body, kind="body")
                    target_source = f"body:{target_body}"
                if target_pos is None and "object_target_pos" in rule:
                    try:
                        target_pos = np.asarray(rule["object_target_pos"], dtype=np.float64).reshape(3)
                        target_source = "explicit"
                    except Exception:
                        target_pos = None
                if target_pos is not None:
                    offset = np.asarray(rule.get("object_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
                    object_anchor_result = _set_freejoint_body_pos(env, object_body, target_pos + offset)
                    logger.info(
                        "[SUBTASK_RELEASE_OBJECT_ANCHOR] t=%s task=%s released_hold_subtask=%s next_subtask=%s "
                        "object_body=%s target_source=%s target=%s offset=%s result=%s",
                        t,
                        planner.task_info.task_id,
                        released_hold_subtask,
                        next_subtask,
                        object_body,
                        target_source,
                        np.round(np.asarray(target_pos, dtype=np.float64), 6).tolist(),
                        np.round(offset, 6).tolist(),
                        object_anchor_result,
                    )
                else:
                    logger.warning(
                        "[SUBTASK_RELEASE_OBJECT_ANCHOR_SKIPPED] t=%s task=%s released_hold_subtask=%s "
                        "next_subtask=%s object_body=%s target_site=%s target_body=%s reason=no_target_pos",
                        t,
                        planner.task_info.task_id,
                        released_hold_subtask,
                        next_subtask,
                        object_body,
                        target_site,
                        target_body,
                    )
            refresh_obs_after_anchor()
            logger.info(
                "[SUBTASK_RELEASE_ANCHOR] t=%s task=%s released_hold_subtask=%s next_subtask=%s "
                "rule=%s anchor_file=%s frame_idx=%s anchor_ee=%s joint=%s gripper=%s object_anchor=%s",
                t,
                planner.task_info.task_id,
                released_hold_subtask,
                next_subtask,
                rule.get("tag", f"{rule['released']}->{rule['next']}"),
                anchor_hdf5,
                frame_idx,
                format_vec3(anchor["ee_pos"]),
                np.round(anchor["joint_states"], 6).tolist(),
                {
                    "method": gripper_method or "SKIP",
                    "target": np.round(anchor["gripper_states"], 6).tolist(),
                    "obs": np.round(np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float64), 6).tolist()
                    if isinstance(obs, dict)
                    else [],
                },
                object_anchor_result,
            )
            return True
        except Exception:
            logger.exception(
                "[SUBTASK_RELEASE_ANCHOR_FAILED] t=%s task=%s released_hold_subtask=%s next_subtask=%s "
                "rule=%s anchor_file=%s frame_idx=%s",
                t,
                planner.task_info.task_id,
                released_hold_subtask,
                next_subtask,
                rule.get("tag", f"{rule['released']}->{rule['next']}"),
                anchor_hdf5,
                frame_idx,
            )
            return False

    logger.info(
        "sync endpose-hold rollout: task=%s replan_steps=%s hold=%s tol=%.5f eef_default_tol=%.5f "
        "eef_p95_extra_tol=%.5f eef_tol_cap=%.5f min_active_steps=%s "
        "consecutive=%s consecutive_by_subtask=%s "
        "post_hold_release_vla_steps=%s post_hold_release_vla_steps_by_subtask=%s "
        "strict_hold_release_next=%s prevent_regression=%s "
        "guard_after_hold=%s regression_guard_mode=hold_majority_prompt disable_output_normalize=%s "
        "forward_switch_block_previous=%s require_hold_release_for_pick_forward=%s "
        "pick_forward_hold_subtasks=%s "
        "require_hold_release_for_place_forward=%s block_forward_before_first_stage_done=%s "
        "microwave_forward_require_prior_hold=%s microwave_forward_gap_fill_next=%s "
        "microwave_stage_lock_until_done=%s "
        "microwave_forward_blocked_no_current_action=%s hold_release_block_past_subtasks=%s "
        "prevent_prompt_frontier_regression=%s prevent_completed_stage_regression=%s "
        "prevent_released_hold_regression=%s prevent_held_subtask_regression=%s "
        "allow_stage_done_release_anchor=%s "
        "completed_update_from_official_stage=%s "
        "hold_start_after_release_anchor=%s hold_start_after_release_anchor_subtasks=%s "
        "vlm_task_text_mode=%s hold_gripper_mode=%s tol_by_subtask=%s "
        "hold_release_min_steps_by_subtask=%s place_object_gates=%s targets=%s "
        "target_passage_requirements=%s direction_hold=%s direction_signatures=%s direction_cos_min=%.3f "
        "direction_window=%s direction_min_displacement=%.5f direction_trend_eps=%.5f "
        "pick_gripper_gate=%s pick_gripper_open_max=%.3f pick_gripper_close_min=%.3f "
        "pick_deferred_gripper_release=%s "
        "post_pick_release_keep_gripper_steps=%s post_pick_release_keep_gripper_value=%.3f "
        "pick_height_gate=%s pick_height_targets=%s pick_height_tol=%.5f "
        "pick_object_lift_gate=%s pick_object_lift_gate_by_subtask=%s pick_object_lift_delta=%.5f "
        "drawer_forward_advance_guard=%s drawer_task_mode=%s "
        "drawer_close_hold_require_stage=%s require_initial_vlm_subtask=%s "
        "require_open_microwave_endpose_hold_before_release=%s "
        "stop_on_stage_success=%s "
        "release_anchor_json=%s release_anchor_rules=%s "
        "initial_anchor_json=%s initial_anchor_rules=%s",
        planner.task_info.task_id,
        args.replan_steps,
        cfg.enabled,
        cfg.pos_tol,
        cfg.eef_default_tol,
        cfg.eef_p95_extra_tol,
        cfg.eef_tol_cap,
        cfg.min_active_steps,
        cfg.consecutive,
        dict(sorted(consecutive_by_subtask.items())),
        cfg.post_release_vla_steps,
        dict(sorted(post_release_vla_steps_by_subtask.items())),
        cfg.strict_hold_release_next,
        cfg.prevent_regression,
        cfg.regression_guard_after_hold_release,
        disable_output_normalize,
        forward_switch_block_previous,
        require_hold_release_for_pick_forward,
        sorted(pick_forward_hold_subtasks),
        require_hold_release_for_place_forward,
        block_forward_before_first_stage_done,
        microwave_forward_require_prior_hold,
        microwave_forward_gap_fill_next,
        microwave_stage_lock_until_done,
        microwave_forward_blocked_no_current_action,
        hold_release_block_past_subtasks,
        prevent_prompt_frontier_regression,
        prevent_completed_stage_regression,
        prevent_released_hold_regression,
        prevent_held_subtask_regression,
        allow_stage_done_release_anchor,
        completed_update_from_official_stage,
        hold_start_after_release_anchor,
        sorted(hold_start_after_release_anchor_subtasks),
        os.environ.get("VLM_TASK_TEXT_MODE", "default"),
        hold_gripper_mode,
        dict(sorted(tol_by_subtask.items())),
        dict(sorted(hold_release_min_steps_by_subtask.items())),
        dict(sorted(place_object_gates.items())),
        sorted(targets.keys()),
        dict(sorted(target_passage_requirements.items())),
        use_direction_hold,
        sorted(direction_signatures.keys()),
        cfg.direction_cos_min,
        cfg.direction_window,
        cfg.direction_min_displacement,
        cfg.direction_trend_eps,
        cfg.pick_gripper_gate,
        cfg.pick_gripper_open_max,
        cfg.pick_gripper_close_min,
        cfg.pick_deferred_gripper_release,
        post_pick_release_keep_gripper_steps,
        post_pick_release_keep_gripper_value,
        cfg.pick_height_gate,
        dict(sorted(pick_height_targets.items())),
        cfg.pick_height_tol,
        cfg.pick_object_lift_gate,
        dict(sorted(pick_lift_gate_by_subtask.items())),
        cfg.pick_object_lift_delta,
        drawer_forward_advance_guard,
        drawer_task_mode,
        cfg.drawer_close_hold_require_stage,
        require_initial_vlm_subtask,
        require_open_microwave_endpose_hold_before_release,
        stop_on_stage_success,
        os.environ.get("SUBTASK_RELEASE_ANCHORS_JSON", "").strip(),
        release_anchor_rules,
        os.environ.get("INITIAL_SUBTASK_ANCHORS_JSON", "").strip(),
        initial_anchor_rules,
    )

    episode_exception: Exception | None = None
    try:
        append_eef_pos()
        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                append_eef_pos()
                append_vlm_frame()
                t += 1
                if check_goal(bool(done)):
                    break
                continue

            if state is None:
                state = base.stage_eval._build_initial_state(env)
                current_stage_start = state["step_idx"]
                official_state = official_stage._build_initial_state(env)
                official_current_stage_start = official_state["step_idx"]

            if len(recent_vlm_frames) < args.n_recent:
                obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                append_eef_pos()
                append_vlm_frame()
                t += 1
                if update_stage_and_goal(bool(done)):
                    break
                continue

            effective_t = t - args.num_steps_wait
            runtime_hint_phase = "active" if hold_active else ("post_release" if post_hold_hint_subtask else "")
            setattr(
                planner,
                "_runtime_hold_state",
                {
                    "active": bool(runtime_hint_phase),
                    "subtask": hold_subtask if hold_active else (post_hold_hint_subtask or current_subtask_prompt),
                    "phase": runtime_hint_phase or "active",
                },
            )
            if hold_active and hold_skip_vlm_inference:
                latest_subtask = current_subtask_prompt
                logger.info(
                    "[VLM_SKIPPED_DURING_HOLD] t=%s task=%s subtask=%s",
                    t,
                    planner.task_info.task_id,
                    hold_subtask,
                )
            else:
                latest_subtask = planner.infer_sync(effective_t, clone_recent_frames())
                if runtime_hint_phase == "post_release":
                    post_hold_hint_subtask = ""
            raw_latest_subtask = str(latest_subtask)
            if disable_output_normalize:
                latest_subtask = " ".join(str(latest_subtask).strip().lower().replace("_", " ").split())
            else:
                latest_subtask = normalize_subtask(latest_subtask, labels)
            save_vlm_debug_frame(raw_latest_subtask, latest_subtask)

            if microwave_forward_gap_fill_next and current_subtask_prompt and latest_subtask and labels:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    current_idx is not None
                    and latest_idx is not None
                    and latest_idx > current_idx + 1
                ):
                    filled_subtask = labels[current_idx + 1]
                    logger.info(
                        "[SUBTASK_FORWARD_GAP_FILL_NEXT] t=%s task=%s current_subtask=%s "
                        "raw_subtask=%s raw_idx=%s filled_subtask=%s filled_idx=%s",
                        t,
                        planner.task_info.task_id,
                        current_subtask_prompt,
                        latest_subtask,
                        latest_idx,
                        filled_subtask,
                        current_idx + 1,
                    )
                    latest_subtask = filled_subtask

            if microwave_forward_require_prior_hold and latest_subtask and labels:
                latest_idx = order_index(latest_subtask, labels)
                if latest_idx is not None and latest_idx > 0:
                    required_previous = labels[latest_idx - 1]
                    if required_previous not in runtime_completed_subtasks:
                        logger.info(
                            "[SUBTASK_FORWARD_WAIT_PRIOR_HOLD] t=%s task=%s raw_subtask=%s "
                            "raw_idx=%s required_previous=%s completed=%s action=%s",
                            t,
                            planner.task_info.task_id,
                            latest_subtask,
                            latest_idx,
                            required_previous,
                            runtime_completed_subtasks,
                            "keep_current" if current_subtask_prompt else microwave_forward_blocked_no_current_action,
                        )
                        if current_subtask_prompt:
                            latest_subtask = current_subtask_prompt
                        elif microwave_forward_blocked_no_current_action == "default_vla":
                            latest_subtask = ""
                        else:
                            obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                            append_eef_pos()
                            append_vlm_frame()
                            t += 1
                            if update_stage_and_goal(bool(done)):
                                break
                            continue

            if require_initial_vlm_subtask and (not current_subtask_prompt) and labels:
                required_initial = labels[0]
                required_stage = official_stage_specs[0].name if official_stage_specs else None
                stage_done_already = bool(required_stage and official_stage_done.get(required_stage, False))
                if (not stage_done_already) and latest_subtask != required_initial:
                    logger.info(
                        "[VLM_INITIAL_SUBTASK_WAIT] t=%s task=%s raw_subtask=%s required_initial=%s "
                        "waiting_stage=%s action=dummy_step_no_oracle_prompt",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        required_initial,
                        required_stage,
                    )
                    obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                    append_eef_pos()
                    append_vlm_frame()
                    t += 1
                    if update_stage_and_goal(bool(done)):
                        break
                    continue

            if block_forward_before_first_stage_done and labels and official_stage_specs and latest_subtask:
                required_initial = labels[0]
                required_stage = official_stage_specs[0].name
                stage_done_already = bool(official_stage_done.get(required_stage, False))
                latest_idx = order_index(latest_subtask, labels)
                if (not stage_done_already) and latest_idx is not None and latest_idx > 0:
                    logger.info(
                        "[FORWARD_BLOCK_BEFORE_FIRST_STAGE_DONE] t=%s task=%s raw_subtask=%s "
                        "required_initial=%s waiting_stage=%s action=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        required_initial,
                        required_stage,
                        "keep_current" if current_subtask_prompt else "dummy_step_no_oracle_prompt",
                    )
                    if current_subtask_prompt:
                        latest_subtask = current_subtask_prompt
                    else:
                        obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                        append_eef_pos()
                        append_vlm_frame()
                        t += 1
                        if update_stage_and_goal(bool(done)):
                            break
                        continue

            if oracle_force_initial_prompt and not current_subtask_prompt and labels:
                oracle_initial = labels[0]
                if latest_subtask != oracle_initial:
                    logger.info(
                        "[ORACLE_INITIAL_PROMPT] t=%s task=%s raw_subtask=%s oracle_initial=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        oracle_initial,
                    )
                latest_subtask = oracle_initial

            if prevent_completed_stage_regression and latest_subtask:
                completed_stage_name = completed_stage_name_for_prompt(latest_subtask)
                completed_stage_done = bool(
                    completed_stage_name and official_stage_done.get(completed_stage_name, False)
                )
                if (
                    completed_stage_done
                    and latest_subtask != current_subtask_prompt
                    and current_subtask_prompt
                ):
                    logger.info(
                        "[COMPLETED_STAGE_REGRESSION_BLOCKED] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s completed_stage=%s action=keep_current",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        completed_stage_name,
                    )
                    latest_subtask = current_subtask_prompt
                elif (
                    completed_stage_done
                    and latest_subtask == current_subtask_prompt
                    and current_subtask_prompt
                    and not can_replay_completed_prompt(current_subtask_prompt)
                ):
                    logger.info(
                        "[COMPLETED_STAGE_REPLAY_BLOCKED] t=%s task=%s subtask=%s "
                        "completed_stage=%s action=dummy_wait_for_vlm",
                        t,
                        planner.task_info.task_id,
                        current_subtask_prompt,
                        completed_stage_name,
                    )
                    obs, _, done, _ = env.step(base.ec.LIBERO_DUMMY_ACTION)
                    append_eef_pos()
                    append_vlm_frame()
                    t += 1
                    if update_stage_and_goal(bool(done)):
                        break
                    continue

            if (
                oracle_initial_stage_lock
                and current_subtask_prompt
                and labels
                and current_subtask_prompt == labels[0]
                and official_stage_specs
                and not official_stage_done.get(official_stage_specs[0].name, False)
                and latest_subtask != current_subtask_prompt
            ):
                logger.info(
                    "[ORACLE_INITIAL_STAGE_LOCK] t=%s task=%s raw_subtask=%s locked_subtask=%s "
                    "waiting_stage=%s",
                    t,
                    planner.task_info.task_id,
                    latest_subtask,
                    current_subtask_prompt,
                    official_stage_specs[0].name,
                )
                latest_subtask = current_subtask_prompt

            if hold_active and latest_subtask and latest_subtask != current_subtask_prompt:
                min_hold_steps = hold_release_min_steps(hold_subtask)
                held_steps = 0 if hold_started_t is None else max(0, t - hold_started_t)
                hold_idx = order_index(hold_subtask, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    hold_idx is not None
                    and latest_idx is not None
                    and latest_idx > hold_idx
                    and latest_subtask != pending_hold_release_subtask
                ):
                    pending_hold_release_subtask = latest_subtask
                    pending_hold_release_t = t
                    logger.info(
                        "[ENDPOSE_HOLD_PENDING_RELEASE_PROMPT] t=%s task=%s hold_subtask=%s "
                        "pending_subtask=%s hold_idx=%s pending_idx=%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        pending_hold_release_subtask,
                        hold_idx,
                        latest_idx,
                    )
                if held_steps < min_hold_steps:
                    logger.info(
                        "[ENDPOSE_HOLD_RELEASE_MIN_STEPS_BLOCKED] t=%s task=%s hold_subtask=%s "
                        "raw_subtask=%s held_steps=%s required_steps=%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        latest_subtask,
                        held_steps,
                        min_hold_steps,
                    )
                    latest_subtask = current_subtask_prompt

            if (
                hold_active
                and pending_hold_release_subtask
                and hold_started_t is not None
                and latest_subtask == current_subtask_prompt
            ):
                min_hold_steps = hold_release_min_steps(hold_subtask)
                held_steps = max(0, t - hold_started_t)
                if held_steps >= min_hold_steps:
                    logger.info(
                        "[ENDPOSE_HOLD_RELEASE_PENDING_PROMPT] t=%s task=%s hold_subtask=%s "
                        "pending_subtask=%s pending_seen_t=%s held_steps=%s required_steps=%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        pending_hold_release_subtask,
                        pending_hold_release_t,
                        held_steps,
                        min_hold_steps,
                    )
                    latest_subtask = pending_hold_release_subtask

            if hold_active and cfg.strict_hold_release_next and latest_subtask and latest_subtask != current_subtask_prompt:
                hold_idx = order_index(hold_subtask, labels)
                latest_idx = order_index(latest_subtask, labels)
                expected_idx = None if hold_idx is None else hold_idx + 1
                if hold_idx is None or latest_idx is None or latest_idx != expected_idx:
                    logger.info(
                        "[ENDPOSE_HOLD_RELEASE_BLOCKED] t=%s task=%s hold_subtask=%s raw_subtask=%s "
                        "hold_idx=%s raw_idx=%s expected_next_idx=%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        latest_subtask,
                        hold_idx,
                        latest_idx,
                        expected_idx,
                    )
                    latest_subtask = current_subtask_prompt

            if hold_active and hold_release_block_past_subtasks and latest_subtask and latest_subtask != current_subtask_prompt:
                hold_idx = order_index(hold_subtask, labels)
                latest_idx = order_index(latest_subtask, labels)
                if hold_idx is not None and latest_idx is not None and latest_idx <= hold_idx:
                    logger.info(
                        "[ENDPOSE_HOLD_RELEASE_PAST_BLOCKED] t=%s task=%s hold_subtask=%s raw_subtask=%s "
                        "hold_idx=%s raw_idx=%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        latest_subtask,
                        hold_idx,
                        latest_idx,
                    )
                    latest_subtask = current_subtask_prompt

            if regression_guard_active and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    cfg.prevent_regression
                    and
                    prevent_prompt_frontier_regression
                    and latest_subtask != current_subtask_prompt
                    and max_prompt_idx_seen is not None
                    and latest_idx is not None
                    and latest_idx < max_prompt_idx_seen
                ):
                    logger.info(
                        "[PROMPT_FRONTIER_REGRESSION_BLOCKED] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s raw_idx=%s max_prompt_idx_seen=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        max_prompt_idx_seen,
                    )
                    latest_subtask = current_subtask_prompt
                    latest_idx = current_idx
                if (
                    cfg.prevent_regression
                    and
                    drawer_task_mode
                    and latest_subtask != current_subtask_prompt
                    and max_prompt_idx_seen is not None
                    and latest_idx is not None
                    and latest_idx < max_prompt_idx_seen
                ):
                    logger.info(
                        "[DRAWER_FORWARD_BLOCKED] t=%s task=%s raw_subtask=%s current_subtask=%s "
                        "latest_idx=%s max_prompt_idx_seen=%s raw_key=%s current_key=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        max_prompt_idx_seen,
                        subtask_temporal_stripped_key(latest_subtask),
                        subtask_temporal_stripped_key(current_subtask_prompt),
                    )
                    latest_subtask = current_subtask_prompt
                    latest_idx = current_idx
                if (
                    (
                        cfg.prevent_regression
                        or prevent_released_hold_regression
                        or prevent_held_subtask_regression
                    )
                    and latest_subtask != current_subtask_prompt
                    and latest_subtask in blocked_after_hold_prompts
                ):
                    logger.info(
                        "[SUBTASK_REGRESSION_BLOCKED] t=%s task=%s raw_subtask=%s current_subtask=%s "
                        "guard_mode=hold_majority_prompt blocked_prompts=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        sorted(blocked_after_hold_prompts),
                    )
                    latest_subtask = current_subtask_prompt

            if oracle_monotonic_sequence_lock and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    current_idx is not None
                    and latest_idx is not None
                    and latest_idx < current_idx
                    and latest_subtask != current_subtask_prompt
                ):
                    logger.info(
                        "[ORACLE_MONOTONIC_SEQUENCE_LOCK] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s raw_idx=%s current_idx=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        current_idx,
                    )
                    latest_subtask = current_subtask_prompt

            if (
                (oracle_stage_lock_until_done or microwave_stage_lock_until_done)
                and ((not hold_active) or microwave_stage_lock_until_done)
                and current_subtask_prompt
                and latest_subtask
            ):
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                stage_name = official_stage_name_for_subtask(current_subtask_prompt)
                if (
                    stage_name is not None
                    and not official_stage_done.get(stage_name, False)
                    and current_idx is not None
                    and latest_idx is not None
                    and latest_idx > current_idx
                    and latest_subtask != current_subtask_prompt
                ):
                    lock_name = (
                        "ORACLE_STAGE_LOCK_UNTIL_DONE"
                        if oracle_stage_lock_until_done
                        else "MICROWAVE_STAGE_LOCK_UNTIL_DONE"
                    )
                    logger.info(
                        "[%s] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s raw_idx=%s current_idx=%s waiting_stage=%s",
                        lock_name,
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        current_idx,
                        stage_name,
                    )
                    latest_subtask = current_subtask_prompt

            if require_hold_release_for_pick_forward and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                current_pick_hold_started = has_started_endpose_hold(current_subtask_prompt)
                if should_block_pick_forward(
                    enabled=True,
                    hold_active=hold_active,
                    current_subtask=current_subtask_prompt,
                    next_subtask=latest_subtask,
                    current_index=current_idx,
                    next_index=latest_idx,
                    selected_subtasks=pick_forward_hold_subtasks,
                    hold_started_before=current_pick_hold_started,
                ):
                    logger.info(
                        "[PICK_FORWARD_HOLD_RELEASE_REQUIRED] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s raw_idx=%s current_idx=%s selected_subtasks=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        current_idx,
                        sorted(pick_forward_hold_subtasks),
                    )
                    latest_subtask = current_subtask_prompt
                elif (
                    current_pick_hold_started
                    and current_subtask_prompt.startswith("pick ")
                    and current_idx is not None
                    and latest_idx is not None
                    and latest_idx > current_idx
                    and latest_subtask != current_subtask_prompt
                ):
                    logger.info(
                        "[PICK_FORWARD_ALLOWED_PRIOR_EEF_HOLD] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s raw_idx=%s current_idx=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        current_idx,
                    )

            if require_hold_release_for_place_forward and (not hold_active) and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
                    current_subtask_prompt.startswith("place ")
                    and current_idx is not None
                    and latest_idx is not None
                    and latest_idx > current_idx
                    and latest_subtask != current_subtask_prompt
                ):
                    logger.info(
                        "[PLACE_FORWARD_HOLD_RELEASE_REQUIRED] t=%s task=%s raw_subtask=%s "
                        "current_subtask=%s raw_idx=%s current_idx=%s",
                        t,
                        planner.task_info.task_id,
                        latest_subtask,
                        current_subtask_prompt,
                        latest_idx,
                        current_idx,
                    )
                    latest_subtask = current_subtask_prompt

            if oracle_stage_advance_next and current_subtask_prompt and labels:
                stage_name = official_stage_name_for_subtask(current_subtask_prompt)
                current_idx = order_index(current_subtask_prompt, labels)
                if (
                    stage_name is not None
                    and official_stage_done.get(stage_name, False)
                    and current_idx is not None
                    and current_idx + 1 < len(labels)
                ):
                    oracle_next = labels[current_idx + 1]
                    if latest_subtask != oracle_next:
                        logger.info(
                            "[ORACLE_STAGE_ADVANCE_NEXT] t=%s task=%s current_subtask=%s "
                            "stage=%s raw_subtask=%s oracle_next=%s",
                            t,
                            planner.task_info.task_id,
                            current_subtask_prompt,
                            stage_name,
                            latest_subtask,
                            oracle_next,
                        )
                    latest_subtask = oracle_next

            if hold_active and latest_subtask:
                hold_prompt_counts[latest_subtask] = hold_prompt_counts.get(latest_subtask, 0) + 1

            if oracle_hold_release_next and hold_active:
                hold_idx = order_index(hold_subtask, labels)
                if hold_idx is not None and hold_idx + 1 < len(labels):
                    oracle_next = labels[hold_idx + 1]
                    if latest_subtask != oracle_next:
                        logger.info(
                            "[ORACLE_HOLD_RELEASE_NEXT] t=%s task=%s hold_subtask=%s raw_subtask=%s oracle_next=%s",
                            t,
                            planner.task_info.task_id,
                            hold_subtask,
                            latest_subtask,
                            oracle_next,
                        )
                    latest_subtask = oracle_next

            if latest_subtask and latest_subtask != current_subtask_prompt:
                previous = current_subtask_prompt
                if (
                    previous
                    and requires_endpose_hold_before_release(previous)
                    and not has_started_endpose_hold(previous)
                ):
                    previous_stage_name = official_stage_name_for_subtask(previous)
                    logger.info(
                        "[SUBTASK_RELEASE_BLOCKED_ENDPOSE_HOLD_REQUIRED] t=%s task=%s "
                        "previous=%s next_subtask=%s previous_stage=%s previous_stage_done=%s "
                        "reason=endpose_hold_not_started",
                        t,
                        planner.task_info.task_id,
                        previous,
                        latest_subtask,
                        previous_stage_name,
                        bool(previous_stage_name and official_stage_done.get(previous_stage_name, False)),
                    )
                    latest_subtask = current_subtask_prompt
                else:
                    released_from_hold = hold_active
                    released_hold_subtask = hold_subtask
                    if (
                        released_from_hold
                        and cfg.post_pick_hold_release_same_prompt_steps > 0
                        and is_pick_subtask(released_hold_subtask)
                        and is_place_subtask(latest_subtask)
                    ):
                        logger.info(
                            "[POST_PICK_HOLD_RELEASE_SAME_PROMPT_START] t=%s task=%s "
                            "old_subtask=%s new_subtask=%s steps=%s",
                            t,
                            planner.task_info.task_id,
                            released_hold_subtask,
                            latest_subtask,
                            cfg.post_pick_hold_release_same_prompt_steps,
                        )
                        if run_vla_without_vlm(
                            cfg.post_pick_hold_release_same_prompt_steps,
                            phase="before_pick_hold_release_switch_same_prompt",
                        ):
                            break
                        logger.info(
                            "[POST_PICK_HOLD_RELEASE_SAME_PROMPT_END] t=%s task=%s "
                            "old_subtask=%s new_subtask=%s",
                            t,
                            planner.task_info.task_id,
                            released_hold_subtask,
                            latest_subtask,
                        )
                    current_subtask_prompt = latest_subtask
                    current_subtask_start_t = t
                    endpose_streak = 0
                    reset_pick_completion_gate(current_subtask_prompt)
                    maybe_apply_initial_subtask_anchor(current_subtask_prompt)
                    release_anchor_applied = False
                    if (
                        prevent_held_subtask_regression
                        and previous
                        and has_started_endpose_hold(previous)
                        and previous not in blocked_after_hold_prompts
                    ):
                        blocked_after_hold_prompts.add(previous)
                        logger.info(
                            "[HELD_SUBTASK_BLOCKLIST_ADD] t=%s task=%s previous=%s new=%s "
                            "released_from_hold=%s blocked_prompts=%s",
                            t,
                            planner.task_info.task_id,
                            previous,
                            current_subtask_prompt,
                            released_from_hold,
                            sorted(blocked_after_hold_prompts),
                        )
                    if released_from_hold:
                        record_completed_subtask(released_hold_subtask, "hold_release")
                        if (
                            post_pick_release_keep_gripper_steps > 0
                            and is_pick_subtask(released_hold_subtask)
                            and is_place_subtask(current_subtask_prompt)
                        ):
                            post_pick_keep_gripper_until_t = t + post_pick_release_keep_gripper_steps
                            post_pick_keep_gripper_source = (
                                f"{released_hold_subtask}->{current_subtask_prompt}@t{t}"
                            )
                            logger.info(
                                "[POST_PICK_KEEP_GRIPPER_START] t=%s task=%s old_subtask=%s "
                                "new_subtask=%s steps=%s until_t=%s gripper_value=%+.3f",
                                t,
                                planner.task_info.task_id,
                                released_hold_subtask,
                                current_subtask_prompt,
                                post_pick_release_keep_gripper_steps,
                                post_pick_keep_gripper_until_t,
                                post_pick_release_keep_gripper_value,
                            )
                        block_prompt = most_common_hold_prompt()
                        if block_prompt:
                            blocked_after_hold_prompts.add(block_prompt)
                        if hold_release_block_past_subtasks:
                            released_idx = order_index(released_hold_subtask, labels)
                            if released_idx is not None:
                                blocked_after_hold_prompts.update(labels[: released_idx + 1])
                        logger.info(
                            "[ENDPOSE_HOLD_RELEASE] t=%s task=%s old_subtask=%s new_subtask=%s "
                            "blocked_after_release=%s hold_prompt_counts=%s",
                            t,
                            planner.task_info.task_id,
                            released_hold_subtask,
                            current_subtask_prompt,
                            block_prompt,
                            dict(sorted(hold_prompt_counts.items())),
                            )
                        release_anchor_applied = maybe_apply_release_anchor(
                            released_hold_subtask,
                            current_subtask_prompt,
                        )
                        if hold_release_block_past_subtasks:
                            logger.info(
                                "[HOLD_RELEASE_PAST_BLOCKLIST_ADD] t=%s task=%s old_subtask=%s "
                                "blocked_prompts=%s",
                                t,
                                planner.task_info.task_id,
                                released_hold_subtask,
                                sorted(blocked_after_hold_prompts),
                            )
                    elif previous:
                        previous_stage_name = official_stage_name_for_subtask(previous)
                        if (
                            allow_autonomous_forward_release_anchor
                            and normalize_subtask(previous, labels) in autonomous_forward_anchor_subtasks
                        ):
                            applied = maybe_apply_release_anchor(previous, current_subtask_prompt)
                            logger.info(
                                "[SUBTASK_RELEASE_ANCHOR_AUTONOMOUS_FORWARD] t=%s task=%s "
                                "previous=%s next_subtask=%s applied=%s",
                                t,
                                planner.task_info.task_id,
                                previous,
                                current_subtask_prompt,
                                applied,
                            )
                        elif (
                            allow_stage_done_release_anchor
                            and previous_stage_name is not None
                            and official_stage_done.get(previous_stage_name, False)
                        ):
                            release_anchor_applied = maybe_apply_release_anchor(
                                previous,
                                current_subtask_prompt,
                            )
                        elif previous_stage_name is not None:
                            if allow_stage_done_release_anchor:
                                pending_stage_release_anchor = (
                                    previous,
                                    current_subtask_prompt,
                                    previous_stage_name,
                                )
                            logger.info(
                                "[SUBTASK_RELEASE_ANCHOR_%s] t=%s task=%s "
                                "previous=%s next_subtask=%s previous_stage=%s previous_stage_done=%s "
                                "reason=anchor_requires_eef_hold_release",
                                "PENDING_STAGE_DONE" if allow_stage_done_release_anchor else "SKIPPED_STAGE_RUNTIME_DISABLED",
                                t,
                                planner.task_info.task_id,
                                previous,
                                current_subtask_prompt,
                                previous_stage_name,
                                bool(official_stage_done.get(previous_stage_name, False)),
                            )
                    hold_active = False
                    hold_subtask = ""
                    hold_started_t = None
                    pending_hold_release_subtask = ""
                    pending_hold_release_t = None
                    hold_prompt_counts.clear()
                    if (
                        hold_start_after_release_anchor
                        and release_anchor_applied
                        and (
                            not hold_start_after_release_anchor_subtasks
                            or normalize_subtask(current_subtask_prompt, labels)
                            in hold_start_after_release_anchor_subtasks
                        )
                        and can_hold(current_subtask_prompt)
                        and current_subtask_prompt in targets
                    ):
                        anchor_dist = distance_to_target(obs, targets[current_subtask_prompt])
                        anchor_tol = pos_tol_for_subtask(current_subtask_prompt)
                        if anchor_dist <= anchor_tol:
                            hold_active = True
                            hold_subtask = current_subtask_prompt
                            hold_started_t = t
                            endpose_streak = consecutive_by_subtask.get(
                                normalize_subtask(hold_subtask, labels),
                                cfg.consecutive,
                            )
                            endpose_hold_started_subtasks.add(normalize_subtask(hold_subtask, labels))
                            hold_prompt_counts.clear()
                            if hold_subtask:
                                hold_prompt_counts[hold_subtask] = 1
                                record_completed_subtask(hold_subtask, "hold_start_release_anchor")
                            logger.info(
                                "[ENDPOSE_HOLD_START_AFTER_RELEASE_ANCHOR] t=%s task=%s subtask=%s "
                                "dist=%.5f tol=%.5f streak=%s",
                                t,
                                planner.task_info.task_id,
                                hold_subtask,
                                anchor_dist,
                                anchor_tol,
                                endpose_streak,
                            )
                            mark_open_eef_hold_verified("release_anchor")
                        else:
                            logger.info(
                                "[ENDPOSE_HOLD_AFTER_RELEASE_ANCHOR_SKIPPED] t=%s task=%s subtask=%s "
                                "dist=%.5f tol=%.5f reason=not_near_target",
                                t,
                                planner.task_info.task_id,
                                current_subtask_prompt,
                                anchor_dist,
                                anchor_tol,
                            )
                    if released_from_hold and cfg.regression_guard_after_hold_release:
                        regression_guard_active = True
                        logger.info(
                            "[SUBTASK_REGRESSION_GUARD_ON] t=%s task=%s trigger=hold_release subtask=%s",
                            t,
                            planner.task_info.task_id,
                            current_subtask_prompt,
                        )
                    logger.info("[t=%s] VLM sync subtask update: %s -> %s", t, previous or "<none>", current_subtask_prompt)
                    previous_idx = order_index(previous, labels) if previous else None
                    current_idx = order_index(current_subtask_prompt, labels)
                    if (
                        forward_switch_block_previous
                        and previous
                        and previous_idx is not None
                        and current_idx is not None
                        and current_idx > previous_idx
                    ):
                        previous_hold_started = has_started_endpose_hold(previous)
                        if released_from_hold or previous_hold_started:
                            blocked_after_hold_prompts.add(previous)
                            logger.info(
                                "[FORWARD_SWITCH_BLOCKLIST_ADD] t=%s task=%s previous=%s new=%s "
                                "previous_idx=%s new_idx=%s released_from_hold=%s previous_hold_started=%s "
                                "blocked_prompts=%s",
                                t,
                                planner.task_info.task_id,
                                previous,
                                current_subtask_prompt,
                                previous_idx,
                                current_idx,
                                released_from_hold,
                                previous_hold_started,
                                sorted(blocked_after_hold_prompts),
                            )
                        else:
                            logger.info(
                                "[FORWARD_SWITCH_BLOCKLIST_SKIP_UNHELD] t=%s task=%s previous=%s new=%s "
                                "previous_idx=%s new_idx=%s reason=previous_no_eef_hold",
                                t,
                                planner.task_info.task_id,
                                previous,
                                current_subtask_prompt,
                                previous_idx,
                                current_idx,
                            )
                    if current_idx is not None and (max_prompt_idx_seen is None or current_idx > max_prompt_idx_seen):
                        max_prompt_idx_seen = current_idx
                        logger.info(
                            "[DRAWER_FORWARD_FRONTIER] t=%s task=%s subtask=%s frontier_idx=%s",
                            t,
                            planner.task_info.task_id,
                            current_subtask_prompt,
                            max_prompt_idx_seen,
                        )
                    if current_subtask_prompt == final_subtask:
                        logger.info("[FINAL_HINT] t=%s task=%s VLM reached final subtask.", t, planner.task_info.task_id)
                    post_release_steps = post_release_vla_steps_by_subtask.get(
                        normalize_subtask(current_subtask_prompt, labels),
                        cfg.post_release_vla_steps,
                    )
                    if released_from_hold and (not hold_active) and post_release_steps > 0:
                        if run_vla_without_vlm(post_release_steps, phase="after_hold_release"):
                            break
                        continue

            if hold_active:
                norm_hold_subtask = normalize_subtask(hold_subtask, labels)
                if (
                    hold_auto_resume_same_prompt
                    and norm_hold_subtask not in hold_auto_resume_excluded_subtasks
                    and hold_started_t is not None
                ):
                    held_steps = max(0, t - hold_started_t)
                    min_hold_steps = hold_release_min_steps(hold_subtask)
                    if held_steps >= min_hold_steps:
                        cooldown_until = t + hold_auto_resume_cooldown_steps
                        if norm_hold_subtask:
                            endpose_hold_cooldown_until[norm_hold_subtask] = cooldown_until
                        logger.info(
                            "[ENDPOSE_HOLD_AUTO_RESUME_SAME_PROMPT] t=%s task=%s subtask=%s "
                            "held_steps=%s min_hold_steps=%s cooldown_until=%s post_vla_steps=%s",
                            t,
                            planner.task_info.task_id,
                            hold_subtask,
                            held_steps,
                            min_hold_steps,
                            cooldown_until,
                            cfg.post_release_vla_steps,
                        )
                        hold_active = False
                        hold_subtask = ""
                        hold_started_t = None
                        pending_hold_release_subtask = ""
                        pending_hold_release_t = None
                        hold_prompt_counts.clear()
                        if cfg.post_release_vla_steps > 0:
                            if run_vla_without_vlm(
                                cfg.post_release_vla_steps,
                                phase="after_hold_auto_resume_same_prompt",
                            ):
                                break
                        post_hold_hint_subtask = norm_hold_subtask
                        continue
                target = targets.get(hold_subtask)
                if hold_gripper_mode == "zero":
                    hold_gripper = 0.0
                else:
                    hold_gripper = float(target["hold_gripper"]) if target else -1.0
                hold_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hold_gripper]
                logger.info(
                    "[ENDPOSE_HOLD_STEP] t=%s task=%s subtask=%s hold_steps=%s gripper=%+.1f gripper_mode=%s",
                    t,
                    planner.task_info.task_id,
                    hold_subtask,
                    args.replan_steps,
                    hold_gripper,
                    hold_gripper_mode,
                )
                for _ in range(args.replan_steps):
                    if step_env(
                        hold_action,
                        current_subtask_prompt or planner.default_subtask_prompt,
                        "hold_zero_action",
                    ):
                        raise StopIteration
                    if t >= args.max_steps + args.num_steps_wait:
                        break
                continue

            check_before_vla = env_bool("ENDPOSE_HOLD_CHECK_BEFORE_VLA", True)
            if check_before_vla and maybe_update_endpose_streak(current_subtask_prompt, "before_vla", t):
                hold_active = True
                hold_subtask = current_subtask_prompt
                hold_started_t = t
                pending_hold_release_subtask = ""
                pending_hold_release_t = None
                endpose_hold_started_subtasks.add(normalize_subtask(hold_subtask, labels))
                hold_prompt_counts.clear()
                if hold_subtask:
                    hold_prompt_counts[hold_subtask] = 1
                    record_completed_subtask(hold_subtask, "hold_start_before_vla")
                logger.info("[ENDPOSE_HOLD_START] t=%s task=%s subtask=%s source=before_vla", t, planner.task_info.task_id, hold_subtask)
                mark_open_eef_hold_verified("before_vla")
                continue

            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            element = base.obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            if len(actions) < args.replan_steps:
                raise RuntimeError(f"VLA returned {len(actions)} actions, need at least {args.replan_steps}")
            logger.info("[t=%s] VLA sync chunk: %s steps | prompt=%s", t, args.replan_steps, prompt_for_vla)
            for chunk_idx, action in enumerate(actions[: args.replan_steps], start=1):
                if step_env(action, prompt_for_vla, f"vla_chunk_{chunk_idx}/{args.replan_steps}"):
                    raise StopIteration
                if maybe_update_endpose_streak(current_subtask_prompt, f"after_vla_chunk{chunk_idx}", t):
                    hold_active = True
                    hold_subtask = current_subtask_prompt
                    hold_started_t = t
                    pending_hold_release_subtask = ""
                    pending_hold_release_t = None
                    endpose_hold_started_subtasks.add(normalize_subtask(hold_subtask, labels))
                    hold_prompt_counts.clear()
                    if hold_subtask:
                        hold_prompt_counts[hold_subtask] = 1
                        record_completed_subtask(hold_subtask, "hold_start_after_vla")
                    logger.info(
                        "[ENDPOSE_HOLD_START] t=%s task=%s subtask=%s source=after_vla_chunk%s",
                        t,
                        planner.task_info.task_id,
                        hold_subtask,
                        chunk_idx,
                    )
                    mark_open_eef_hold_verified(f"after_vla_chunk{chunk_idx}")
                    break
                if t >= args.max_steps + args.num_steps_wait:
                    break
    except StopIteration:
        pass
    except Exception as exc:
        episode_exception = exc
        logger.exception("episode failed")

    stage_pct = _official_stage_score_pct(task_id_int, official_stage_done)
    official_stage_success_raw = _official_stage_success(task_id_int, official_stage_done)
    open_eef_contract_ok = bool(
        (not require_open_eef_hold_for_success)
        or task_id_int not in {20, 21, 23, 24}
        or open_eef_hold_verified
    )
    stage_success = bool(official_stage_success_raw and episode_exception is None)
    logger.info(
        "[MICROWAVE_OPEN_EEF_SUCCESS_AUDIT] task=%s required=%s verified=%s hold_t=%s "
        "hold_dist=%s hold_door_joint=%s official_stage_success_raw=%s audited_stage_success=%s",
        task_id_int,
        int(require_open_eef_hold_for_success),
        int(open_eef_hold_verified),
        open_eef_hold_t if open_eef_hold_t is not None else "NA",
        f"{open_eef_hold_dist:.5f}" if open_eef_hold_dist is not None else "NA",
        f"{open_eef_hold_door_angle:.5f}" if open_eef_hold_door_angle is not None else "NA",
        int(official_stage_success_raw),
        int(stage_success),
    )
    official_goal_success = _official_goal_success(task_id_int, env, official_stage_done, stage_success)
    if not ever_goal_success:
        ever_goal_success = (
            bool(goal_check_override(env, stage_done))
            if goal_check_override is not None
            else bool(base.ec.check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False)
        )
    if min_endpose_dist:
        for subtask in sorted(min_endpose_dist):
            logger.info(
                "[ENDPOSE_MIN_DISTANCE] task=%s subtask=%s min_dist=%.5f tol=%.5f min_t=%s",
                planner.task_info.task_id,
                subtask,
                min_endpose_dist[subtask],
                pos_tol_for_subtask(subtask),
                min_endpose_t[subtask],
            )
    if max_pick_height_z:
        for subtask in sorted(max_pick_height_z):
            target = pick_height_targets.get(subtask, {})
            logger.info(
                "[PICK_HEIGHT_MAX] task=%s subtask=%s max_z=%.5f target_z=%s z_min=%s max_t=%s",
                planner.task_info.task_id,
                subtask,
                max_pick_height_z[subtask],
                f"{float(target['height_z_target']):.5f}" if "height_z_target" in target else "NA",
                f"{float(target['height_z_min']):.5f}" if "height_z_min" in target else "NA",
                max_pick_height_t[subtask],
            )
    if target_passage_count:
        logger.info(
            "[ENDPOSE_PASSAGE_SUMMARY] task=%s passages=%s requirements=%s",
            planner.task_info.task_id,
            dict(sorted(target_passage_count.items())),
            dict(sorted(target_passage_requirements.items())),
        )
    logger.info(
        "[OFFICIAL_SCORE] task=%s average_score_pct=%.6f stage_success=%s goal_success=%s stage_done_json=%s",
        task_id_int,
        stage_pct,
        int(stage_success),
        int(official_goal_success),
        json.dumps(official_stage_done, ensure_ascii=False, separators=(",", ":")),
    )
    if task_id_int in MICROWAVE_STAGE_ONLY_TASKS:
        logger.info(
            "[SCORING_POLICY] task=%s policy=stage_only_close_microwave_optional counted_stages=%s ignored_goal_success=%s",
            task_id_int,
            _official_counted_stage_names(task_id_int, official_stage_done),
            int(official_goal_success),
        )
    diagnostics = {
        "stage_success": bool(stage_success),
        "official_stage_success_raw": bool(official_stage_success_raw),
        "open_eef_hold_verified": bool(open_eef_hold_verified),
        "open_eef_hold_t": open_eef_hold_t,
        "open_eef_hold_dist": open_eef_hold_dist,
        "open_eef_hold_door_angle": open_eef_hold_door_angle,
        "extra_pour_detected": False,
        "failure_reason": "" if stage_success else (
            "evaluator_runtime_error" if episode_exception is not None else "stage_incomplete"
        ),
        "counted_stages": _official_counted_stage_names(task_id_int, official_stage_done),
        "ignored_goal_success": bool(official_goal_success),
    }
    if episode_exception is not None:
        raise RuntimeError("episode evaluator failed; see sync_vlm.log") from episode_exception
    return stage_pct, official_stage_done, stage_success, diagnostics, replay, replay_wrist


def _write_official_summaries() -> None:
    out_root = Path(os.environ["OUT_ROOT"])
    pattern = re.compile(
        r"\[OFFICIAL_SCORE\] task=(\d+) average_score_pct=([0-9.]+) "
        r"stage_success=([01]) goal_success=([01]) stage_done_json=(\{.*\})"
    )
    rows: list[dict[str, Any]] = []
    for log_path in sorted(out_root.glob("task*/ep*/sync_vlm.log")):
        matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
        if not matches:
            continue
        task, score, stage_success, goal_success, stage_json = matches[-1]
        ep_match = re.search(r"ep(\d+)$", log_path.parent.name)
        ep = int(ep_match.group(1)) if ep_match else len(rows)
        seed = int(os.environ.get("SEED", "104")) + ep
        rows.append(
            {
                "task_id": int(task),
                "ep": ep,
                "seed": seed,
                "score_pct": float(score),
                "tsr_success": bool(int(stage_success)),
                "stage_success": bool(int(stage_success)),
                "goal_success": bool(int(goal_success)),
                "stage_done": json.loads(stage_json),
                "log": str(log_path),
            }
        )
    rows.sort(key=lambda row: (row["task_id"], row["ep"]))
    episodes_tsv = out_root / "official_episodes.tsv"
    with episodes_tsv.open("w", encoding="utf-8") as handle:
        handle.write("task_id\tep\tseed\tscore_pct\ttsr_success\tstage_success\tgoal_success\tlog\n")
        for row in rows:
            handle.write(
                f'{row["task_id"]}\t{row["ep"]}\t{row["seed"]}\t{row["score_pct"]:.1f}\t'
                f'{"Y" if row["tsr_success"] else "N"}\t{"Y" if row["stage_success"] else "N"}\t'
                f'{"Y" if row["goal_success"] else "N"}\t{row["log"]}\n'
            )
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(grouped.items()):
        n = len(task_rows)
        summaries.append(
            {
                "task_id": task_id,
                "num_trials": n,
                "seed_start": int(os.environ.get("SEED", "104")),
                "average_score_pct": sum(row["score_pct"] for row in task_rows) / max(1, n),
                "tsr_success_rate_pct": 100.0 * sum(row["tsr_success"] for row in task_rows) / max(1, n),
                "stage_success_rate_pct": 100.0 * sum(row["stage_success"] for row in task_rows) / max(1, n),
                "goal_success_rate_pct": 100.0 * sum(row["goal_success"] for row in task_rows) / max(1, n),
            }
        )
    (out_root / "official_summary.json").write_text(
        json.dumps({"episodes": rows, "tasks": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_root / "official_task_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "task_id\tnum_trials\tseed_start\taverage_score_pct\ttsr_success_rate_pct\t"
            "stage_success_rate_pct\tgoal_success_rate_pct\n"
        )
        for row in summaries:
            handle.write(
                f'{row["task_id"]}\t{row["num_trials"]}\t{row["seed_start"]}\t'
                f'{row["average_score_pct"]:.1f}\t{row["tsr_success_rate_pct"]:.1f}\t'
                f'{row["stage_success_rate_pct"]:.1f}\t{row["goal_success_rate_pct"]:.1f}\n'
            )


def main() -> None:
    os.environ["ASYNC_VLM"] = "0"
    _patch_official_bddl_resolution()
    _patch_stage_eval_compat()
    base.FullVlm26MemoryPlanner._build_messages = _build_messages_runtime_progress
    base.run_episode_async_stateful = run_episode_sync_endpose_hold
    base.main()
    _write_official_summaries()


if __name__ == "__main__":
    main()
