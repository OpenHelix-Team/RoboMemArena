# examples/robocerebra/robocerebra_adapter.py
# -*- coding: utf-8 -*-
"""
Adapter utilities to bridge RoboCerebra env observations to pi-0.5 policy inputs.
Matched exactly with DataRecorder logic:
  1. Only np.flipud (No horizontal flip/mirroring)
  2. Force resize (Squish to square, no padding)
"""

from __future__ import annotations
from collections import deque
from typing import Any, Dict
import numpy as np
import cv2


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to axis-angle (robosuite-compatible)."""
    quat = quat.astype(np.float64).copy()
    # Normalize quaternion first to avoid numerical errors
    norm = np.linalg.norm(quat)
    if norm > 1e-12:
        quat = quat / norm
    
    # Robosuite convention: w is last [x, y, z, w]
    # Check if we need to flip sign to ensure w is positive (canonical representation)
    if quat[3] < 0:
        quat = -quat

    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    
    if np.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    
    # Calculate angle and axis
    angle = 2.0 * np.arccos(quat[3])
    out = (quat[:3] * angle) / den
    return out.astype(np.float32)


def _process_image_match_training(x: np.ndarray, size: int) -> np.ndarray:
    """
    Process image EXACTLY as done in DataRecorder:
    1. np.flipud (Fix Robosuite upside-down)
    2. cv2.resize (Force resize/squish to target size, ignoring aspect ratio)
    """
    # 1.  HWC 
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))
    
    # 2.  uint8
    if x.dtype != np.uint8:
        if x.max() <= 1.0:
            x = (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)

    # 3. 🔥  A:  (flipud)，
    #  DataRecorder: img = np.flipud(img)
    x = np.flipud(x)

    # 4. 🔥  B:  (Squish)，
    #  DataRecorder: cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    if x.shape[0] != size or x.shape[1] != size:
        x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA)
        
    return x


def _extract_state(obs: Dict[str, Any]) -> np.ndarray:
    eef_pos = obs.get("robot0_eef_pos")
    eef_quat = obs.get("robot0_eef_quat")
    gripper = obs.get("robot0_gripper_qpos")

    if eef_pos is None:
        eef_pos = obs.get("eef_pos") or np.zeros(3, dtype=np.float32)
    if eef_quat is None:
        eef_quat = obs.get("eef_quat") or np.array([0, 0, 0, 1], dtype=np.float32)
    if gripper is None:
        gripper = obs.get("gripper_qpos") or np.zeros(1, dtype=np.float32)

    return np.concatenate(
        [
            np.asarray(eef_pos, dtype=np.float32),
            _quat2axisangle(np.asarray(eef_quat, dtype=np.float32)),
            np.asarray(gripper, dtype=np.float32),
        ]
    )


def _extract_images(obs: Dict[str, Any], resize_size: int) -> tuple[np.ndarray, np.ndarray]:
    img_main = obs.get("agentview_image", None)
    if img_main is None:
        img_main = obs.get("agentview_rgb", None)
    if img_main is None:
        raise KeyError("Neither 'agentview_image' nor 'agentview_rgb' found in obs")

    img_wrist = obs.get("robot0_eye_in_hand_image", None)
    if img_wrist is None:
        img_wrist = obs.get("wrist_image", None)
    if img_wrist is None:
        raise KeyError("Neither 'robot0_eye_in_hand_image' nor 'wrist_image' found in obs")

    processed_main = _process_image_match_training(np.asarray(img_main), resize_size)
    processed_wrist = _process_image_match_training(np.asarray(img_wrist), resize_size)
    return processed_main, processed_wrist


def create_history(mem_obs_steps: int) -> dict[str, deque]:
    return {
        "state": deque(maxlen=mem_obs_steps),
        "cam_high": deque(maxlen=mem_obs_steps),
        "cam_left_wrist": deque(maxlen=mem_obs_steps),
    }


def _pad_sequence(valid: list[np.ndarray], target_len: int, pad_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad_count = max(0, target_len - len(valid))
    seq = [pad_value.copy() for _ in range(pad_count)] + [np.asarray(x) for x in valid]
    mask = np.array([True] * pad_count + [False] * len(valid), dtype=np.bool_)
    return np.stack(seq, axis=0), mask


def obs_to_pi_element(
    obs: Dict[str, Any],
    resize_size: int,
    prompt: str | None = None,
) -> Dict[str, Any]:
    """
    Build the single policy input element for pi-0.5 server.
    """
    state = _extract_state(obs)
    processed_main, processed_wrist = _extract_images(obs, resize_size)

    # --- 4.  OpenPI  ---
    #  pi0  3  image slot，
    element = {
        #  Config  ( base -> agentview)
        "observation/image": processed_main,
        "observation/wrist_image": processed_wrist,
        
        #  key ， server 
        "base_0_rgb": processed_main,
        "left_wrist_0_rgb": np.zeros_like(processed_wrist), #  ()
        "right_wrist_0_rgb": processed_wrist,               #  eye_in_hand 
        
        "observation/state": state,
        "prompt": "" if prompt is None else str(prompt),
    }
    
    return element


def obs_to_pi_mem_element(
    obs: Dict[str, Any],
    history: dict[str, deque],
    resize_size: int,
    mem_obs_steps: int,
    prompt: str | None = None,
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    state = _extract_state(obs)
    processed_main, processed_wrist = _extract_images(obs, resize_size)

    history["state"].append(state)
    history["cam_high"].append(processed_main)
    history["cam_left_wrist"].append(processed_wrist)

    pad_state = np.zeros_like(state)
    pad_image = np.zeros_like(processed_main)

    state_history, state_is_pad = _pad_sequence(list(history["state"]), mem_obs_steps, pad_state)
    cam_high_history, cam_high_is_pad = _pad_sequence(list(history["cam_high"]), mem_obs_steps, pad_image)
    wrist_history, wrist_is_pad = _pad_sequence(list(history["cam_left_wrist"]), mem_obs_steps, pad_image)

    element = {
        "state_history": state_history,
        "state_history_is_pad": state_is_pad,
        "images_history": {
            "cam_high": cam_high_history,
            "cam_left_wrist": wrist_history,
        },
        "image_history_is_pad": {
            "cam_high": cam_high_is_pad,
            "cam_left_wrist": wrist_is_pad,
        },
        "prompt": "" if prompt is None else str(prompt),
    }
    return element, processed_main, processed_wrist
