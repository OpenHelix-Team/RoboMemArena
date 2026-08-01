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

from legacy_rollout_compat import run_archived_sync_rollout


BASE_EVAL_PY = Path(
    os.environ.get(
        "TASKS2_26_BASE_EVAL_PY",
        "${ASSET_OFFICIAL_RUNTIME_ROOT}/"
        "evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py",
    )
)

spec = importlib.util.spec_from_file_location("_tasks2_26_base_eval", BASE_EVAL_PY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base eval from {BASE_EVAL_PY}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

_ORIG_FULLVLM26_BUILD_MESSAGES = base.FullVlm26MemoryPlanner._build_messages


_official_scripts_dir = os.environ.get("ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR", "").strip()
if not _official_scripts_dir:
    raise RuntimeError("ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR is required")
OFFICIAL_SCRIPTS_DIR = Path(_official_scripts_dir)
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
    official_stage = _load_official_module(
        "_robomemarena_official_task2_26_stage",
        OFFICIAL_SCRIPTS_DIR / "task2_26_reference_stage.py",
    )
finally:
    if _previous_eval_common is None:
        sys.modules.pop("eval_common", None)
    else:
        sys.modules["eval_common"] = _previous_eval_common


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
    return messages


DEFAULT_TARGET_JSON = (
    "${ASSET_OPENPI_INFERENCE_ROOT}/tmp/tasks2_26_holdstatic_general/"
    "tasks2_26_endpose_targets_seed100_199.json"
)
LEGACY_TARGET_PASSAGE_COUNTS_JSON = (
    "${ASSET_OPENPI_INFERENCE_ROOT}/tmp/tasks2_26_holdstatic_general/"
    "tasks2_26_target_passage_counts_seed100_199.json"
)
DEFAULT_TARGET_PASSAGE_COUNTS_JSON = (
    "${ASSET_OPENPI_INFERENCE_ROOT}/tmp/tasks2_26_holdstatic_general/"
    "tasks2_26_target_passage_counts_seed100_199_alltasks_tol045_20260624_074452.json"
)
DEFAULT_H5DUMP_BIN = os.environ.get("H5DUMP_BIN", "/share/anaconda3/bin/h5dump")


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
            anchor_hdf5 = str(rule.get("anchor_hdf5", "")).strip()
            frame_idx = int(rule.get("frame_idx", 0))
            if not released or not nxt or not anchor_hdf5:
                raise ValueError(f"Task {task_key} rule #{idx} must contain released/next/anchor_hdf5")
            parsed_rules.append(
                {
                    "released": released,
                    "next": nxt,
                    "anchor_hdf5": anchor_hdf5,
                    "frame_idx": max(0, frame_idx),
                    "tag": str(rule.get("tag", f"{released}->{nxt}")).strip() or f"{released}->{nxt}",
                }
            )
        out[task_id] = parsed_rules
    return out


def active_release_anchor_rules(task_id: int) -> list[dict[str, Any]]:
    path_str = os.environ.get("SUBTASK_RELEASE_ANCHORS_JSON", "").strip()
    if not path_str:
        return []
    return _load_release_anchor_rules_from_json(path_str).get(task_id, [])


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


def run_episode_sync_endpose_hold(
    *,
    env: Any,
    client: Any,
    planner: Any,
    args: Any,
    stage_specs: list[Any],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    vlm_camera_pose: dict | None,
    logger: logging.Logger,
) -> tuple[float, dict[str, bool], bool, list[np.ndarray], list[np.ndarray]]:
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
    official_stage_specs = official_stage._task_specs(task_id_int)
    release_anchor_rules = active_release_anchor_rules(task_id_int)
    use_direction_hold = bool(direction_signatures)
    disable_output_normalize = env_bool("DISABLE_OUTPUT_NORMALIZE", False)
    drawer_forward_advance_guard = env_bool("DRAWER_FORWARD_ADVANCE_GUARD", False)
    forward_switch_block_previous = env_bool("FORWARD_SWITCH_BLOCK_PREVIOUS", False)
    hold_release_block_past_subtasks = env_bool("HOLD_RELEASE_BLOCK_PAST_SUBTASKS", False)
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
    min_endpose_dist: dict[str, float] = {}
    min_endpose_t: dict[str, int] = {}
    max_pick_height_z: dict[str, float] = {}
    max_pick_height_t: dict[str, int] = {}
    pick_object_key_cache: dict[str, str] = {}
    pick_object_key_source: dict[str, str] = {}
    pick_object_baseline_z: dict[str, float] = {}
    pick_object_baseline_t: dict[str, int] = {}
    target_inside_region: dict[str, bool] = {}
    target_passage_count: dict[str, int] = {}
    regression_guard_active = not cfg.regression_guard_after_hold_release
    blocked_after_hold_prompts: set[str] = set()
    hold_prompt_counts: dict[str, int] = {}
    runtime_completed_subtasks: list[str] = []
    setattr(planner, "_runtime_completed_subtasks", runtime_completed_subtasks)
    ever_goal_success = False
    last_gripper_action: float | None = None
    pick_gate_open_seen = False
    pick_gate_closed_after_open = False
    pick_gate_open_t: int | None = None
    pick_gate_close_t: int | None = None
    max_prompt_idx_seen: int | None = None
    close_hold_stage_gate_logged: set[str] = set()
    t = 0

    def is_pick_subtask(subtask: str) -> bool:
        return " ".join(str(subtask).strip().lower().split()).startswith("pick ")

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

    def most_common_hold_prompt() -> str:
        if not hold_prompt_counts:
            return hold_subtask
        return max(
            hold_prompt_counts.items(),
            key=lambda item: (item[1], 1 if item[0] == hold_subtask else 0, item[0]),
        )[0]

    def record_completed_subtask(subtask: str, source: str) -> None:
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
        if signature is None:
            return False, None, None, None, "missing_signature"
        window = max(1, int(signature.get("window", cfg.direction_window) or cfg.direction_window))
        if len(eef_pos_history) <= window:
            return False, None, None, None, "short_history"
        start_pos = eef_pos_history[-1 - window]
        end_pos = eef_pos_history[-1]
        motion = end_pos - start_pos
        displacement = float(np.linalg.norm(motion))
        if displacement < cfg.direction_min_displacement:
            return False, None, displacement, None, "low_displacement"
        motion_dir = motion / displacement
        target_dir = np.asarray(signature["direction_mean"], dtype=np.float64)
        cos_sim = float(np.dot(motion_dir, target_dir))
        prev_dist = float(np.linalg.norm(start_pos - target["target_ee_pos"]))
        trend_ok = dist <= prev_dist + cfg.direction_trend_eps
        if cos_sim < cfg.direction_cos_min:
            return False, cos_sim, displacement, prev_dist, "cos_low"
        if not trend_ok:
            return False, cos_sim, displacement, prev_dist, "moving_away"
        return True, cos_sim, displacement, prev_dist, "ok"

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
                    official_stage_idx += 1
                    official_current_stage_start = official_state["step_idx"]
            if official_stage_idx >= len(official_stage_specs) and not official_all_stages_logged:
                logger.info("[t=%s] all official stages done", t)
                official_all_stages_logged = True
        return check_goal(done)

    def maybe_update_endpose_streak(subtask: str, phase: str, t_now: int) -> bool:
        nonlocal endpose_streak
        if subtask not in targets:
            return False
        dist = distance_to_target(obs, targets[subtask])
        pos_tol = pos_tol_for_subtask(subtask)
        explicit_pick_height_target = pick_height_targets.get(subtask)
        pick_object_key, pick_object_key_source = resolved_pick_object_key(subtask)
        pick_height_applies = bool(
            is_pick_subtask(subtask)
            and pick_object_key is not None
            and (cfg.pick_object_lift_gate or explicit_pick_height_target is not None)
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
            near_target = current_z >= height_z_min
        else:
            height_z_min = None
            height_z_target = None
            baseline_z = None
            near_target = dist <= pos_tol
        required_passages = max(1, int(target_passage_requirements.get(subtask, 1)))
        was_inside = target_inside_region.get(subtask, False)
        if pick_height_applies:
            if near_target or was_inside:
                logger.info(
                    "[PICK_HEIGHT_GATE] t=%s task=%s subtask=%s z=%.5f target_z=%.5f z_min=%.5f "
                    "height_ok=%s active_steps=%s object_key=%s object_key_source=%s baseline_z=%s phase=%s",
                    t_now,
                    planner.task_info.task_id,
                    subtask,
                    current_z,
                    height_z_target,
                    height_z_min,
                    near_target,
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
            passage_ok = True if use_direction_hold else target_passage_count.get(subtask, 0) >= required_passages
            direction_ok, direction_cos, direction_disp, prev_dist, direction_reason = (
                direction_gate(subtask, targets[subtask], dist) if near_target else (False, None, None, None, "not_near")
            )
            gripper_gate_ok = True
        should_count = (
            can_hold(subtask)
            and active_steps >= cfg.min_active_steps
            and near_target
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
                "height_z_min=%s height_z_target=%s "
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
                endpose_streak,
                cfg.consecutive,
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
        return can_hold(subtask) and endpose_streak >= cfg.consecutive

    def build_video_overlay_lines(prompt_for_vla: str, control_mode: str) -> list[str]:
        raw_prompt = " ".join(str(prompt_for_vla).strip().split()) if prompt_for_vla else "<none>"
        subtask = normalize_subtask(prompt_for_vla, labels) if prompt_for_vla else ""
        lines = [
            f"t={t} control={control_mode}",
            f"vla_prompt={raw_prompt}",
        ]
        if hold_active or hold_subtask:
            lines.append(
                f"hold_active={int(bool(hold_active))} hold_subtask={hold_subtask or '<none>'} "
                f"endpose_streak={endpose_streak}/{cfg.consecutive}"
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
            logger.info(
                "[SUBTASK_RELEASE_ANCHOR] t=%s task=%s released_hold_subtask=%s next_subtask=%s "
                "rule=%s anchor_file=%s frame_idx=%s anchor_ee=%s joint=%s gripper=%s",
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
        "consecutive=%s post_hold_release_vla_steps=%s strict_hold_release_next=%s prevent_regression=%s "
        "guard_after_hold=%s regression_guard_mode=hold_majority_prompt disable_output_normalize=%s "
        "forward_switch_block_previous=%s hold_release_block_past_subtasks=%s "
        "vlm_task_text_mode=%s hold_gripper_mode=%s tol_by_subtask=%s targets=%s "
        "target_passage_requirements=%s direction_hold=%s direction_signatures=%s direction_cos_min=%.3f "
        "direction_window=%s direction_min_displacement=%.5f direction_trend_eps=%.5f "
        "pick_gripper_gate=%s pick_gripper_open_max=%.3f pick_gripper_close_min=%.3f "
        "pick_height_gate=%s pick_height_targets=%s pick_height_tol=%.5f "
        "pick_object_lift_gate=%s pick_object_lift_delta=%.5f "
        "drawer_forward_advance_guard=%s drawer_task_mode=%s "
        "drawer_close_hold_require_stage=%s release_anchor_json=%s release_anchor_rules=%s",
        planner.task_info.task_id,
        args.replan_steps,
        cfg.enabled,
        cfg.pos_tol,
        cfg.eef_default_tol,
        cfg.eef_p95_extra_tol,
        cfg.eef_tol_cap,
        cfg.min_active_steps,
        cfg.consecutive,
        cfg.post_release_vla_steps,
        cfg.strict_hold_release_next,
        cfg.prevent_regression,
        cfg.regression_guard_after_hold_release,
        disable_output_normalize,
        forward_switch_block_previous,
        hold_release_block_past_subtasks,
        os.environ.get("VLM_TASK_TEXT_MODE", "default"),
        hold_gripper_mode,
        dict(sorted(tol_by_subtask.items())),
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
        cfg.pick_height_gate,
        dict(sorted(pick_height_targets.items())),
        cfg.pick_height_tol,
        cfg.pick_object_lift_gate,
        cfg.pick_object_lift_delta,
        drawer_forward_advance_guard,
        drawer_task_mode,
        cfg.drawer_close_hold_require_stage,
        os.environ.get("SUBTASK_RELEASE_ANCHORS_JSON", "").strip(),
        release_anchor_rules,
    )

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
            latest_subtask = planner.infer_sync(effective_t, clone_recent_frames())
            if disable_output_normalize:
                latest_subtask = " ".join(str(latest_subtask).strip().lower().replace("_", " ").split())
            else:
                latest_subtask = normalize_subtask(latest_subtask, labels)

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

            if cfg.prevent_regression and regression_guard_active and current_subtask_prompt and latest_subtask:
                current_idx = order_index(current_subtask_prompt, labels)
                latest_idx = order_index(latest_subtask, labels)
                if (
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
                if latest_subtask != current_subtask_prompt and latest_subtask in blocked_after_hold_prompts:
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

            if hold_active and latest_subtask:
                hold_prompt_counts[latest_subtask] = hold_prompt_counts.get(latest_subtask, 0) + 1

            if latest_subtask and latest_subtask != current_subtask_prompt:
                previous = current_subtask_prompt
                released_from_hold = hold_active
                released_hold_subtask = hold_subtask
                current_subtask_prompt = latest_subtask
                current_subtask_start_t = t
                endpose_streak = 0
                reset_pick_completion_gate(current_subtask_prompt)
                if released_from_hold:
                    record_completed_subtask(released_hold_subtask, "hold_release")
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
                    maybe_apply_release_anchor(released_hold_subtask, current_subtask_prompt)
                    if hold_release_block_past_subtasks:
                        logger.info(
                            "[HOLD_RELEASE_PAST_BLOCKLIST_ADD] t=%s task=%s old_subtask=%s "
                            "blocked_prompts=%s",
                            t,
                            planner.task_info.task_id,
                            released_hold_subtask,
                            sorted(blocked_after_hold_prompts),
                        )
                hold_active = False
                hold_subtask = ""
                hold_prompt_counts.clear()
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
                    blocked_after_hold_prompts.add(previous)
                    logger.info(
                        "[FORWARD_SWITCH_BLOCKLIST_ADD] t=%s task=%s previous=%s new=%s "
                        "previous_idx=%s new_idx=%s blocked_prompts=%s",
                        t,
                        planner.task_info.task_id,
                        previous,
                        current_subtask_prompt,
                        previous_idx,
                        current_idx,
                        sorted(blocked_after_hold_prompts),
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
                if released_from_hold and cfg.post_release_vla_steps > 0:
                    if run_vla_without_vlm(cfg.post_release_vla_steps, phase="after_hold_release"):
                        break
                    continue

            if hold_active:
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
                hold_prompt_counts.clear()
                if hold_subtask:
                    hold_prompt_counts[hold_subtask] = 1
                    record_completed_subtask(hold_subtask, "hold_start_before_vla")
                logger.info("[ENDPOSE_HOLD_START] t=%s task=%s subtask=%s source=before_vla", t, planner.task_info.task_id, hold_subtask)
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
                    break
                if t >= args.max_steps + args.num_steps_wait:
                    break
    except StopIteration:
        pass
    except Exception:
        logger.exception("episode failed")

    stage_pct = official_stage._stage_score_pct(task_id_int, official_stage_done)
    stage_success = official_stage._stage_success_from_stage_done(task_id_int, official_stage_done)
    official_goal_override = official_stage._goal_override_check(task_id_int)
    if official_goal_override is not None:
        official_goal_success = bool(official_goal_override(env, official_stage_done))
    else:
        official_goal_success = bool(stage_success)
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
    return stage_pct, official_stage_done, official_goal_success, replay, replay_wrist


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


def run_episode_sync_endpose_hold_current_base(**kwargs: Any):
    """Expose the archived rollout through the current official base API."""
    return run_archived_sync_rollout(
        run_episode_sync_endpose_hold,
        official_stage._stage_success_from_stage_done,
        **kwargs,
    )


def main() -> None:
    os.environ["ASYNC_VLM"] = "0"
    base.FullVlm26MemoryPlanner._build_messages = _build_messages_runtime_progress
    base.run_episode_async_stateful = run_episode_sync_endpose_hold_current_base
    base.main()
    _write_official_summaries()


if __name__ == "__main__":
    main()
