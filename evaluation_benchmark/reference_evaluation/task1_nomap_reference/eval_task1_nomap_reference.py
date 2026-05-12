"""
Task1 （HKUST）

- VLM  VLA  RoboMemArena LIBERO  agentview
- ：VLM  -> VLA  10  ->  VLM
"""
from __future__ import annotations
import dataclasses
import io
import logging
import pathlib
import threading
import queue
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import imageio
import numpy as np
import tqdm
import tyro
import os
import sys
import json
import torch
from PIL import Image
import re
import websockets.sync.client
import websockets.exceptions

# Resolve OpenPI runtime root:
# 1) OPENPI_ROOT (if explicitly provided)
# 2) repo-bundled minimal runtime under third_party/openpi_minimal
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OPENPI_ROOT = REPO_ROOT / "third_party" / "openpi_minimal"
OPENPI_ROOT = pathlib.Path(
    os.environ.get("OPENPI_ROOT", str(DEFAULT_OPENPI_ROOT))
).resolve()
DRAWER_DIR = os.path.join(str(OPENPI_ROOT), "examples", "robocerebra_drawer")
HIGH_QWEN_DIR = os.path.join(DRAWER_DIR, "high_qwen")
OPENPI_CLIENT_SRC = os.path.join(str(OPENPI_ROOT), "packages", "openpi-client", "src")
OPENPI_SRC = os.path.join(str(OPENPI_ROOT), "packages", "openpi", "src")

for path in [DRAWER_DIR, HIGH_QWEN_DIR, OPENPI_CLIENT_SRC, OPENPI_SRC]:
    if path not in sys.path:
        sys.path.insert(0, path)

# VLM 
from transformers import AutoProcessor, AutoModelForCausalLM
from keyframe_selection import build_visual_memory, get_frames_from_indices

# VLA client
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocerebra_adapter import obs_to_pi_element, _process_image_match_training

# Libero ：
# 1) TARGET_LIBERO_PATH（）
# 2) uv/.venv 
# 3)  openpi/third_party/libero（）
_libero_path = os.environ.get("TARGET_LIBERO_PATH", "").strip()
if not _libero_path:
    _fallback_libero = REPO_ROOT / "evaluation_benchmark" / "libero_fork"
    if _fallback_libero.exists():
        _libero_path = str(_fallback_libero)
if _libero_path and os.path.exists(_libero_path):
    _libero_path_parent = os.path.dirname(_libero_path.rstrip("/"))
    for _p in (_libero_path, _libero_path_parent):
        if _p and _p not in sys.path:
            sys.path.insert(0, _p)

from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# ========================
# System prompt profiles
# ========================
SYSTEM_PROMPT_TASK1_KF5 = """You are a robotic planning assistant specialized in memory-based task understanding.

Your task is to infer the robot's current primitive action from visual evidence:
1. Historical keyframes from earlier in the episode. These keyframes are listed in chronological order from earliest to latest.
2. The recent short-term visual context, which always contains 5 consecutive frames ending at the current frame.

Important rules:
- Historical keyframes come from earlier times in the same long task.
- The recent 5-frame context is the primary evidence for what is happening now.
- Some samples have no keyframe inside the recent 5-frame context. In that case, keyframe_positions should be an empty list.
- Do not choose from a fixed subtask vocabulary. Write current_primitive freely based on the images.
- Return only a JSON object with exactly these fields:
  - current_primitive: the current primitive action in your own words
  - keyframe_positions: a 1-indexed list of keyframe positions inside the recent 5-frame context
- Do not output any explanation outside the JSON object."""

SYSTEM_PROMPT_TASK1_PLACEIT_NOKF = """You are a visual reasoning assistant for robotic manipulation tasks.
Given the current visual observation, determine the current subtask.

AVAILABLE SUBTASKS (you MUST choose EXACTLY one from this list):
- pick the cookies
- place it into the basket
- pick the tomato sauce

Respond in JSON: {"current_primitive": "<subtask>"}"""

# （5  + ）
SYSTEM_PROMPT = SYSTEM_PROMPT_TASK1_KF5

# ========================
# Global prompt ( task_info.json task1 global_desc )
# ========================
GLOBAL_PROMPT = """Global Task: Execute a sequential dual-object storage task that involves organizing two items into a shared container. Begin by locating and picking up the cookies from the table surface, carefully place it to the basket positioned on the table or countertop, and place the cookies inside the basket to ensure it rest stably within the container interior. Next, return to the workspace to locate and grasp the tomato sauce container, transport it to the same container, and place the tomato sauce into the basket adjacent to the cookies without causing displacement or instability. This systematic workflow ensures both cookies and tomato sauce are stored together in the basket through sequential pick-and-place operations, maintaining proper item organization within the shared storage container."""

GLOBAL_PROMPT_TASK1_PLACEIT_NOKF = """Global Task: Execute a sequential dual-object storage task that involves organizing two items into a shared container.
Begin by locating and picking up the cookies from the table surface, then place it into the basket positioned
on the table or countertop so the item rests stably within the container interior. Next, return to the workspace
to locate and grasp the tomato sauce container, transport it to the same basket, and place it into the basket
adjacent to the cookies without causing displacement or instability. This workflow ensures both cookies and
tomato sauce are stored together in the basket through sequential pick-and-place operations while maintaining
proper item organization within the shared storage container."""

SCENE_DESCRIPTION = """Scene description:
- The scene shows a tabletop workspace.
- On the right side of the table there is a basket.
- Near the middle of the table, the square object is the cookies item.
- Near the middle of the table, the cylindrical object is the tomato sauce container."""


@dataclasses.dataclass
class Args:
    # =====  =====
    bddl_file: str = "bddl/1_cookies_tomato_basket.bddl"
    #  640x480 
    # VLA  obs_to_pi_element -> flipud + resize_size
    env_img_height: int = 480
    env_img_width: int = 640
    # VLM  q3vl_fv_r1 ：
    #  fullvlm  256x256 ， resize  768x432
    vlm_render_height: int = 720
    vlm_render_width: int = 1280
    vlm_use_openpi_camera_pose: bool = False
    # VLM  profile:
    # - fullvlm_256:   flipud -> 256x256(square) -> 256x256
    # - task1_768:     flipud -> 768x432
    # - task1_1080:    flipud -> 1080x1080
    # - custom:        
    vlm_input_profile: str = "fullvlm_256"
    #  True， VLM  VLA ：
    # agentview_image -> flipud + resize_size VLA 
    vlm_match_vla_preprocess: bool = False
    #  True， VLM  flip/resize ，
    #  task1/breakfast_like  JPEG 
    vlm_match_training_jpeg_roundtrip: bool = False
    vlm_training_jpeg_quality: int = 30
    vlm_match_fullvlm_source_square: bool = True
    vlm_source_square_size: int = 256
    vlm_resize_for_training: bool = True
    vlm_train_width: int = 256
    vlm_train_height: int = 256
    # VLM prompt profile:
    # - task1_kf5:           5-frame recent context +  + keyframe_positions
    # - task1_placeit_nokf:   +  + {"current_primitive": "..."}
    vlm_prompt_profile: str = "task1_kf5"
    vlm_use_keyframe_memory: bool = True
    #  wristunlimited ：VLM  + 
    vlm_use_wrist: bool = True
    #  True， obs （）
    vlm_wrist_required: bool = False

    # ===== VLA  =====
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 10   #  VLM ，VLA  10 

    # ===== VLM  =====
    base_model_dir: str = "Qwen/Qwen3-VL-8B-Instruct"
    lora_path: str = "none"
    vlm_device: str = "cuda:1"
    max_new_tokens: int = 512
    vlm_model_type: str = "qwen3_vl"  #  Qwen3-VL
    enable_thinking: bool = False
    crop_right_half: bool = False

    # ===== VLM / VLA  =====
    # VLM ： vlm_interval 
    vlm_interval: int = 5
    n_recent: int = 5         # 
    # ；<=0 （unlimited）
    k_max: int = 0
    d_merge: int = 6          # 
    # True: VLM  VLA ，VLM->VLA  subtask buffer 
    async_vlm: bool = True
    vlm_queue_size: int = 1

    # =====  =====
    num_steps_wait: int = 5
    num_trials_per_task: int = 3
    max_steps: int = 2000
    seed: int = 42
    websocket_ping_interval: float | None = None
    websocket_ping_timeout: float | None = None
    websocket_close_timeout: float = 30.0

    # =====  =====
    log_base: str = "output/task1_sync"
    run_id: str = ""  # 
    video_out_path: str = ""
    task_prompt: str = GLOBAL_PROMPT


def _seed_everywhere(seed: int) -> None:
    np.random.seed(seed)


def _extract_problem_name_from_bddl(bddl_file: str) -> str:
    try:
        with open(bddl_file, "r", encoding="utf-8") as f:
            head = f.read(512)
    except OSError:
        return ""
    m = re.search(r"\(define\s+\(problem\s+([^)]+)\)", head)
    return m.group(1).strip() if m else ""


def _resolve_openpi_agentview_pose(bddl_file: str):
    _ = _extract_problem_name_from_bddl(bddl_file)
    return None


def _get_camera_pose(env, camera_name: str = "agentview"):
    cam_id = env.sim.model.camera_name2id(camera_name)
    return (
        env.sim.model.cam_pos[cam_id].copy(),
        env.sim.model.cam_quat[cam_id].copy(),
    )


def _render_agentview_with_pose(
    env,
    *,
    height: int,
    width: int,
    camera_pose: Optional[dict],
):
    if camera_pose is None:
        raw_img = env.sim.render(height=height, width=width, camera_name="agentview")
        return np.asarray(raw_img)

    cam_id = env.sim.model.camera_name2id("agentview")
    orig_pos = env.sim.model.cam_pos[cam_id].copy()
    orig_quat = env.sim.model.cam_quat[cam_id].copy()
    try:
        env.sim.model.cam_pos[cam_id] = np.asarray(camera_pose["pos"], dtype=np.float64)
        env.sim.model.cam_quat[cam_id] = np.asarray(camera_pose["quat"], dtype=np.float64)
        env.sim.forward()
        raw_img = env.sim.render(height=height, width=width, camera_name="agentview")
        return np.asarray(raw_img)
    finally:
        env.sim.model.cam_pos[cam_id] = orig_pos
        env.sim.model.cam_quat[cam_id] = orig_quat
        env.sim.forward()


def _normalize_rgb_uint8(raw_img) -> np.ndarray:
    raw_arr = np.asarray(raw_img)
    if raw_arr.ndim == 3 and raw_arr.shape[0] in (1, 3) and raw_arr.shape[-1] != 3:
        raw_arr = np.transpose(raw_arr, (1, 2, 0))
    if raw_arr.dtype != np.uint8:
        raw_arr = (
            np.clip(raw_arr, 0, 255)
            if raw_arr.max() > 1.0
            else np.clip(raw_arr, 0.0, 1.0) * 255
        ).astype(np.uint8)
    return raw_arr


def _extract_vlm_frame(
    env,
    obs,
    args: Args,
    vlm_camera_pose: Optional[dict],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if args.vlm_match_vla_preprocess:
        raw_img = obs.get("agentview_image", obs.get("agentview_rgb"))
        if raw_img is None:
            raise KeyError("Neither 'agentview_image' nor 'agentview_rgb' found in obs")
        rgb_main_vlm = _process_image_match_training(np.asarray(raw_img), args.resize_size)
    else:
        if args.vlm_use_openpi_camera_pose and vlm_camera_pose is not None:
            raw_img = _render_agentview_with_pose(
                env,
                height=args.vlm_render_height,
                width=args.vlm_render_width,
                camera_pose=vlm_camera_pose,
            )
        else:
            raw_img = obs.get("agentview_image", obs.get("agentview_rgb"))
        raw_arr = _normalize_rgb_uint8(raw_img)

        # LIBERO ；fullvlm 
        #  VLM  flipud， fullvlm  256x256 ，
        #  resize 
        rgb_main_vlm = np.flipud(raw_arr)
        if args.vlm_match_fullvlm_source_square:
            rgb_main_vlm = _resize_vlm_like_training(
                rgb_main_vlm,
                args.vlm_source_square_size,
                args.vlm_source_square_size,
            )
        if args.vlm_resize_for_training:
            rgb_main_vlm = _resize_vlm_like_training(
                rgb_main_vlm, args.vlm_train_width, args.vlm_train_height
            )

    if args.vlm_match_training_jpeg_roundtrip:
        rgb_main_vlm = _jpeg_roundtrip_like_training(
            rgb_main_vlm,
            args.vlm_training_jpeg_quality,
        )
    rgb_wrist_vlm: Optional[np.ndarray] = None
    if args.vlm_use_wrist:
        raw_wrist = obs.get("robot0_eye_in_hand_image", obs.get("wrist_image"))
        if raw_wrist is None:
            if args.vlm_wrist_required:
                raise KeyError(
                    "vlm_wrist_required=True, but neither 'robot0_eye_in_hand_image' nor "
                    "'wrist_image' found in obs"
                )
        else:
            if args.vlm_match_vla_preprocess:
                rgb_wrist_vlm = _process_image_match_training(
                    np.asarray(raw_wrist), args.resize_size
                )
            else:
                wrist_arr = _normalize_rgb_uint8(raw_wrist)
                rgb_wrist_vlm = np.flipud(wrist_arr)
                if args.vlm_match_fullvlm_source_square:
                    rgb_wrist_vlm = _resize_vlm_like_training(
                        rgb_wrist_vlm,
                        args.vlm_source_square_size,
                        args.vlm_source_square_size,
                    )
                if args.vlm_resize_for_training:
                    rgb_wrist_vlm = _resize_vlm_like_training(
                        rgb_wrist_vlm, args.vlm_train_width, args.vlm_train_height
                    )
            if args.vlm_match_training_jpeg_roundtrip and rgb_wrist_vlm is not None:
                rgb_wrist_vlm = _jpeg_roundtrip_like_training(
                    rgb_wrist_vlm,
                    args.vlm_training_jpeg_quality,
                )
    return rgb_main_vlm, rgb_wrist_vlm


def _apply_vlm_input_profile(args: Args) -> None:
    profile = (args.vlm_input_profile or "").strip().lower()
    if profile == "custom":
        return
    if profile == "fullvlm_256":
        args.vlm_match_fullvlm_source_square = True
        args.vlm_source_square_size = 256
        args.vlm_resize_for_training = True
        args.vlm_train_width = 256
        args.vlm_train_height = 256
        return
    if profile == "task1_768":
        args.vlm_match_fullvlm_source_square = False
        args.vlm_source_square_size = 256
        args.vlm_resize_for_training = True
        args.vlm_train_width = 768
        args.vlm_train_height = 432
        return
    if profile == "task1_1080":
        args.vlm_match_fullvlm_source_square = False
        args.vlm_source_square_size = 256
        args.vlm_resize_for_training = True
        args.vlm_train_width = 1080
        args.vlm_train_height = 1080
        return
    raise ValueError(
        f"Unknown vlm_input_profile={args.vlm_input_profile!r}. "
        f"Choose from: fullvlm_256 | task1_768 | task1_1080 | custom"
    )


def _apply_vlm_prompt_profile(args: Args) -> None:
    profile = (args.vlm_prompt_profile or "").strip().lower()
    if profile == "task1_kf5":
        return
    if profile == "task1_placeit_nokf":
        #  no-kf ： + 
        args.n_recent = 1
        args.k_max = 0
        args.vlm_use_keyframe_memory = False
        args.task_prompt = GLOBAL_PROMPT_TASK1_PLACEIT_NOKF
        return
    raise ValueError(
        f"Unknown vlm_prompt_profile={args.vlm_prompt_profile!r}. "
        f"Choose from: task1_kf5 | task1_placeit_nokf"
    )


# （task1 kf5）
KNOWN_SUBTASKS_TASK1_KF5 = [
    "pick cookies",
    "place cookies into basket",
    "pick tomato sauce",
    "place tomato into basket",
]

# no-kf placeit 
KNOWN_SUBTASKS_TASK1_PLACEIT_NOKF = [
    "pick the cookies",
    "place it into the basket",
    "pick the tomato sauce",
]

# （）
SUBTASK_SEQUENCE_TASK1_KF5 = [
    "pick cookies",
    "place cookies into basket",
    "pick tomato sauce",
    "place tomato into basket",
]

# no-kf placeit 
SUBTASK_SEQUENCE_TASK1_PLACEIT_NOKF = [
    "pick the cookies",
    "place it into the basket",
    "pick the tomato sauce",
]

# 
KNOWN_SUBTASKS = KNOWN_SUBTASKS_TASK1_KF5
SUBTASK_SEQUENCE = SUBTASK_SEQUENCE_TASK1_KF5


class StableWebsocketClientPolicy(_websocket_client_policy.WebsocketClientPolicy):
    """Eval-local websocket client wrapper.

    Keep openpi read-only. We only relax keepalive behavior here because
    the first VLA request may block for a long time during JAX/XLA warmup.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        ping_interval: float | None = None,
        ping_timeout: float | None = None,
        close_timeout: float = 30.0,
    ) -> None:
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._close_timeout = close_timeout
        super().__init__(host=host, port=port, api_key=api_key)

    def _wait_for_server(self):
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=None,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    close_timeout=self._close_timeout,
                )
                metadata = _websocket_client_policy.msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (
                ConnectionRefusedError,
                OSError,
                EOFError,
                websockets.exceptions.InvalidMessage,
            ) as e:
                logging.info("Still waiting for server... (%s: %s)", type(e).__name__, e)
                import time
                time.sleep(5)


def _write_video(path: pathlib.Path, frames: list[np.ndarray], fps: int = 10) -> None:
    """Write mp4 robustly.

    `imageio` may fail with pyav codec detection on some nodes. Fall back to
    OpenCV VideoWriter instead of aborting the whole eval.
    """

    if not frames:
        return

    norm_frames = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        norm_frames.append(arr)

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        imageio.mimwrite(
            path,
            norm_frames,
            fps=fps,
            codec="libx264",
        )
        return
    except Exception as e:
        logging.warning("imageio ， cv2: %s", e)

    h, w = norm_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f": {path}")
    try:
        for frame in norm_frames:
            cur = frame
            if cur.shape[:2] != (h, w):
                cur = cv2.resize(cur, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(cur, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _normalize_primitive(primitive: str, allowed_subtasks: Optional[list[str]] = None) -> str:
    """Free-form primitive parser: no whitelist, no mapping, no fallback vocabulary."""
    import re as _re
    p = str(primitive or "").strip()
    p = p.replace("_", " ")
    p = _re.sub(r"\s+", " ", p).strip()
    p = _re.sub(r"\s+\d+$", "", p).strip()
    return p


def _parse_output(
    output_text: str,
    max_pos: int,
    allowed_subtasks: Optional[list[str]] = None,
):
    """:
    {"current_primitive": "...", "keyframe_positions": [...]}
    """
    s = output_text.strip()

    # thinking ： </think> 
    if "</think>" in s:
        idx = s.rfind("</think>")
        s = s[idx + len("</think>"):].strip()

    # 
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # 1.  JSON（）
    primitive = ""
    keyframe_positions = []
    try:
        j = json.loads(s)
        primitive = str(j.get("current_primitive", j.get("current_subtask", ""))).strip()
        keyframe_positions = [
            int(p)
            for p in j.get("keyframe_positions", [])
            if isinstance(p, (int, float))
        ]
    except Exception:
        primitive = s

    keyframe_positions = [p for p in keyframe_positions if 1 <= p <= max_pos]
    return _normalize_primitive(primitive, allowed_subtasks=allowed_subtasks), keyframe_positions


def _crop_right_half(img) -> Image.Image:
    """"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    w, h = img.size
    return img.crop((w // 2, 0, w, h))


def _resize_for_training(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """ VLM （W,H）， LIBERO """
    if img.shape[:2] == (height, width):
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _resize_vlm_like_training(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """ q3vl_fv_r1  fullvlm ：PIL + LANCZOS resize"""
    if img.shape[:2] == (height, width):
        return img
    pil = Image.fromarray(img.astype(np.uint8))
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.LANCZOS
    return np.asarray(pil.resize((width, height), resample))


def _jpeg_roundtrip_like_training(img: np.ndarray, quality: int) -> np.ndarray:
    """ task1/breakfast_like  JPEG encode/decode """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


class SyncLoRAPlanner:
    """
     Qwen3-VL LoRA VLM Planner（Task1 ）
     LoRA （PeftModel）， thinking
    """

    def __init__(
        self,
        base_model_dir: str,
        lora_path: str,
        instruction: str = GLOBAL_PROMPT,
        system_prompt: str = SYSTEM_PROMPT,
        prompt_profile: str = "task1_kf5",
        n_recent: int = 8,
        d_merge: int = 6,
        k_max: int = 0,
        use_keyframe_memory: bool = True,
        max_new_tokens: int = 128,
        device: str = "cuda:0",
        logger: Optional[logging.Logger] = None,
        vlm_model_type: str = "qwen3_vl",
        enable_thinking: bool = False,
        crop_right_half: bool = False,
        use_wrist: bool = True,
    ):
        self.device = device
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.prompt_profile = (prompt_profile or "task1_kf5").strip().lower()
        self.n_recent = n_recent
        self.d_merge = d_merge
        self.k_max = k_max
        self.use_keyframe_memory = use_keyframe_memory
        self.max_new_tokens = max_new_tokens
        self.logger = logger
        self.vlm_model_type = vlm_model_type
        self.enable_thinking = enable_thinking
        self.crop_right_half = crop_right_half
        self.use_wrist = bool(use_wrist)
        self.vlm_training_jpeg_roundtrip = False
        self.vlm_training_jpeg_quality: int | None = None

        if self.prompt_profile == "task1_placeit_nokf":
            self.allowed_subtasks = list(SUBTASK_SEQUENCE_TASK1_PLACEIT_NOKF)
        else:
            self.allowed_subtasks = list(SUBTASK_SEQUENCE_TASK1_KF5)

        self.processor = AutoProcessor.from_pretrained(
            base_model_dir, trust_remote_code=True
        )
        if vlm_model_type == "qwen3_vl":
            logging.info(f" Qwen3-VL: {base_model_dir}")
            from transformers import Qwen3VLForConditionalGeneration

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_dir,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            )
        elif vlm_model_type == "qwen2_5_vl":
            logging.info(f" Qwen2.5-VL: {base_model_dir}")
            from transformers import Qwen2_5_VLForConditionalGeneration

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                base_model_dir,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            )
        else:
            logging.info(f" Qwen3.5: {base_model_dir}")
            from transformers import Qwen3_5ForConditionalGeneration

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                base_model_dir,
                torch_dtype=torch.bfloat16,
                device_map=device,
                trust_remote_code=True,
                local_files_only=True,
            )

        #  LoRA 
        if lora_path and lora_path.lower() != "none":
            from peft import PeftModel
            logging.info(f" LoRA : {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            logging.info("LoRA ！")

        self.model.eval()
        logging.info(f"！(thinking={enable_thinking}, crop_right_half={crop_right_half})")

        # 
        self._subtask_idx: int = 0
        self._current_subtask = ""
        self._advance_votes: int = 0
        self._advance_required: int = 2   #  2 
        self._completed_subtasks: set[str] = set()
        self.R_main: list[Image.Image] = []
        self.R_wrist: list[Optional[Image.Image]] = []
        self.K_main_frames: list[Image.Image] = []
        self.K_wrist_frames: list[Optional[Image.Image]] = []
        self.K_indices_abs: list[int] = []
        self.J_hist: list[list[int]] = []
        self.step: int = 0
        self.frame_store_main: dict[int, Image.Image] = {}
        self.frame_store_wrist: dict[int, Optional[Image.Image]] = {}
        self._saved_k_indices: set = set()

        # 
        self._trace_fh = None
        self.run_dir: Optional[pathlib.Path] = None
        self.kf_dir: Optional[pathlib.Path] = None

    def reset_episode(
        self,
        instruction: Optional[str] = None,
        run_dir=None,
        logger=None,
    ):
        """ episode """
        if self._trace_fh is not None:
            self._trace_fh.close()
            self._trace_fh = None

        if instruction is not None:
            self.instruction = instruction
        if logger is not None:
            self.logger = logger

        self.R_main = []
        self.R_wrist = []
        self.frame_store_main = {}
        self.frame_store_wrist = {}
        self.K_indices_abs = []
        self.K_main_frames = []
        self.K_wrist_frames = []
        self.J_hist = []
        self.step = 0
        self._saved_k_indices = set()
        self._current_subtask = ""
        self._advance_votes = 0
        self._completed_subtasks = set()

        if run_dir is not None:
            self.run_dir = pathlib.Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.kf_dir = self.run_dir / "keyframes"
            self.kf_dir.mkdir(parents=True, exist_ok=True)
            self._trace_fh = open(self.run_dir / "sync_vlm_trace.jsonl", "w")
        else:
            self.run_dir = None
            self.kf_dir = None
            self._trace_fh = None

        if self.logger:
            self.logger.info("SyncLoRAPlanner reset: ")

    def push_frame(self, main_rgb: np.ndarray, wrist_rgb: Optional[np.ndarray] = None):
        """； 5 """
        m_img = Image.fromarray(main_rgb.astype(np.uint8))
        w_img = (
            Image.fromarray(wrist_rgb.astype(np.uint8))
            if wrist_rgb is not None and self.use_wrist
            else None
        )
        self.frame_store_main[self.step] = m_img
        self.frame_store_wrist[self.step] = w_img
        self.R_main.append(m_img)
        self.R_wrist.append(w_img)
        self.step += 1

    def infer_sync(
        self,
        step_idx: int,
        context_frames_np: list[tuple[np.ndarray, Optional[np.ndarray]]],
    ) -> str:
        """
         VLM  — 
         prompt_profile ：
        - task1_kf5: 5  + 
        - task1_placeit_nokf:  + 
        """
        if not context_frames_np:
            return self._current_subtask

        recent_start = step_idx - len(context_frames_np) + 1
        context_main_frames: list[Image.Image] = []
        context_wrist_frames: list[Optional[Image.Image]] = []
        for offset, frame_pack in enumerate(context_frames_np):
            abs_idx = recent_start + offset
            main_frame_np = frame_pack[0] if isinstance(frame_pack, tuple) else frame_pack
            wrist_frame_np = frame_pack[1] if isinstance(frame_pack, tuple) else None
            main_frame_img: Image.Image = (
                Image.fromarray(main_frame_np.astype(np.uint8))
                if isinstance(main_frame_np, np.ndarray)
                else main_frame_np
            )
            if self.crop_right_half:
                main_frame_img = _crop_right_half(main_frame_img)
            wrist_frame_img: Optional[Image.Image] = None
            if self.use_wrist and wrist_frame_np is not None:
                wrist_frame_img = (
                    Image.fromarray(wrist_frame_np.astype(np.uint8))
                    if isinstance(wrist_frame_np, np.ndarray)
                    else wrist_frame_np
                )
                if self.crop_right_half:
                    wrist_frame_img = _crop_right_half(wrist_frame_img)
            self.frame_store_main[abs_idx] = main_frame_img
            self.frame_store_wrist[abs_idx] = wrist_frame_img
            context_main_frames.append(main_frame_img)
            context_wrist_frames.append(wrist_frame_img)
        self.step = max(self.step, step_idx + 1)

        if self.use_keyframe_memory:
            memory_main_frames = list(self.K_main_frames)
            memory_wrist_frames = list(self.K_wrist_frames)
            memory_indices = list(self.K_indices_abs)
        else:
            memory_main_frames = []
            memory_wrist_frames = []
            memory_indices = []
        messages = self._build_messages(
            memory_main_frames,
            memory_wrist_frames,
            context_main_frames,
            context_wrist_frames,
        )

        images = []
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if isinstance(c, dict) and c.get("type") == "image" and "image" in c:
                    images.append(c["image"])

        if self.vlm_model_type == "qwen3_vl":
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if isinstance(text, list):
                text = text[0]
            inputs = self.processor(
                text=[text], images=images if images else None,
                return_tensors="pt", padding=False,
            )
        else:
            tokenizer = self.processor.tokenizer
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            if isinstance(prompt, list):
                prompt = prompt[0]
            inputs = self.processor(
                text=[prompt], images=images if images else None,
                return_tensors="pt", padding=False,
            )

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        vlm_subtask, j_rel = _parse_output(
            out_text,
            max_pos=len(context_main_frames),
            allowed_subtasks=self.allowed_subtasks,
        )
        j_abs = [recent_start + (p - 1) for p in j_rel]

        if self.use_keyframe_memory:
            self.J_hist.append(j_abs)
            raw_k_indices = build_visual_memory(
                self.J_hist, t=self.step, N=len(context_main_frames), d=self.d_merge
            )
            self.K_indices_abs = [idx for idx in raw_k_indices if idx < recent_start]
            self.K_main_frames = get_frames_from_indices(
                self.K_indices_abs, self.frame_store_main
            )
            self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in self.K_indices_abs]
            if self.k_max > 0 and len(self.K_indices_abs) > self.k_max:
                self.K_indices_abs = self.K_indices_abs[-self.k_max:]
                self.K_main_frames = self.K_main_frames[-self.k_max:]
                self.K_wrist_frames = self.K_wrist_frames[-self.k_max:]
        else:
            j_rel = []
            j_abs = []
            self.J_hist = []
            self.K_indices_abs = []
            self.K_main_frames = []
            self.K_wrist_frames = []

        self._dump_new_keyframes()

        # No-map mode: use the VLM primitive text directly as the VLA prompt.
        if vlm_subtask:
            self._current_subtask = vlm_subtask

        subtask = self._current_subtask

        #  VLM  trace
        if self.run_dir is not None:
            image_rel = self._save_vlm_input_bundle(
                step_idx=step_idx,
                memory_main_frames=memory_main_frames,
                memory_wrist_frames=memory_wrist_frames,
                memory_indices=memory_indices,
                context_main_frames=context_main_frames,
                context_wrist_frames=context_wrist_frames,
                subtask=subtask,
            )
        else:
            image_rel = None

        self._append_trace({
            "t": int(step_idx),
            "subtask": subtask,
            "keyframe_positions": j_rel,
            "J_abs": j_abs,
            "K_indices_abs": list(self.K_indices_abs),
            "vlm_training_jpeg_roundtrip": bool(self.vlm_training_jpeg_roundtrip),
            "vlm_training_jpeg_quality": self.vlm_training_jpeg_quality,
            "out_text": out_text.strip()[:600],
            "image": image_rel,
        })

        if self.logger:
            self.logger.info(f"VLM @t={step_idx}: subtask='{subtask}'")
            self.logger.info(f"  keyframe_positions={j_rel}, J_abs={j_abs}, K={self.K_indices_abs}")
            self.logger.info(f"  VLM raw: {out_text[:200]}")

        return self._current_subtask

    def get_current_subtask(self) -> str:
        return self._current_subtask

    def _build_messages(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Optional[Image.Image]],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Optional[Image.Image]],
    ):
        use_wrist_images = self.use_wrist and any(
            frame is not None for frame in (memory_wrist_frames + context_wrist_frames)
        )
        camera_order_text = (
            "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb."
            if use_wrist_images
            else "Camera: agentview_rgb."
        )

        def _append_timestep_images(
            target: list[dict],
            main_frames: list[Image.Image],
            wrist_frames: list[Optional[Image.Image]],
        ) -> int:
            image_count = 0
            for idx, main_img in enumerate(main_frames):
                target.append({"type": "image", "image": main_img})
                image_count += 1
                if use_wrist_images:
                    wrist_img = wrist_frames[idx] if idx < len(wrist_frames) else None
                    if wrist_img is not None:
                        target.append({"type": "image", "image": wrist_img})
                        image_count += 1
            return image_count

        if self.prompt_profile == "task1_placeit_nokf":
            current_main = context_main_frames[-1]
            current_wrist = context_wrist_frames[-1] if context_wrist_frames else None
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"{self.instruction}\n"
                        f"{camera_order_text}\n"
                        "Current observation:"
                    ),
                },
                {"type": "image", "image": current_main},
            ]
            if use_wrist_images and current_wrist is not None:
                user_content.append({"type": "image", "image": current_wrist})
            user_content.append(
                {
                    "type": "text",
                    "text": "Based on the visual information, determine the current subtask.",
                }
            )
            return [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {"role": "user", "content": user_content},
            ]

        user_content = [
            {
                "type": "text",
                "text": (
                    f"{self.instruction}\n"
                    f"{SCENE_DESCRIPTION}\n"
                    f"{camera_order_text}\n"
                    "Current observation:"
                ),
            }
        ]
        if memory_main_frames:
            num_history_keyframes = len(memory_main_frames)
            num_history_images = num_history_keyframes * (2 if use_wrist_images else 1)
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Historical keyframes from earlier in the same demonstration "
                        f"({num_history_keyframes} timesteps, {num_history_images} images):"
                    ),
                }
            )
            _append_timestep_images(
                user_content,
                memory_main_frames,
                memory_wrist_frames,
            )
        num_context_frames = len(context_main_frames)
        num_context_images = num_context_frames * (2 if use_wrist_images else 1)
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Recent visual context: "
                    f"{num_context_frames} consecutive frames ending at the current frame "
                    f"({num_context_images} images):"
                ),
            }
        )
        _append_timestep_images(
            user_content,
            context_main_frames,
            context_wrist_frames,
        )
        user_content.append(
            {
                "type": "text",
                "text": "Predict the current subtask and the keyframe positions inside the recent 5-frame context.",
            }
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _dump_new_keyframes(self):
        if self.kf_dir is None:
            return
        for idx in self.K_indices_abs:
            if idx in self._saved_k_indices:
                continue
            frame = self.frame_store_main.get(idx)
            if frame is None:
                continue
            out = self.kf_dir / f"kf_abs{idx:04d}.png"
            frame.save(out)
            self._saved_k_indices.add(idx)

    def _save_vlm_input_bundle(
        self,
        step_idx: int,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Optional[Image.Image]],
        memory_indices: list[int],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Optional[Image.Image]],
        subtask: str,
    ) -> dict[str, list[str]]:
        if self.run_dir is None:
            return {}
        bundle_dir = self.run_dir / "vlm_inputs" / f"t{step_idx:04d}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        safe_subtask = subtask.replace(" ", "_") if subtask else "unknown"
        memory_paths = []
        for order, (idx, main_frame) in enumerate(zip(memory_indices, memory_main_frames), start=1):
            out_main = bundle_dir / f"memory_{order:02d}_abs{idx:04d}_agentview_{safe_subtask}.png"
            main_frame.save(out_main)
            memory_paths.append(str(out_main.relative_to(self.run_dir)))
            wrist_frame = memory_wrist_frames[order - 1] if order - 1 < len(memory_wrist_frames) else None
            if self.use_wrist and wrist_frame is not None:
                out_wrist = bundle_dir / f"memory_{order:02d}_abs{idx:04d}_wrist_{safe_subtask}.png"
                wrist_frame.save(out_wrist)
                memory_paths.append(str(out_wrist.relative_to(self.run_dir)))

        context_paths = []
        for order, main_frame in enumerate(context_main_frames, start=1):
            out_main = bundle_dir / f"recent_{order:02d}_agentview_{safe_subtask}.png"
            main_frame.save(out_main)
            context_paths.append(str(out_main.relative_to(self.run_dir)))
            wrist_frame = context_wrist_frames[order - 1] if order - 1 < len(context_wrist_frames) else None
            if self.use_wrist and wrist_frame is not None:
                out_wrist = bundle_dir / f"recent_{order:02d}_wrist_{safe_subtask}.png"
                wrist_frame.save(out_wrist)
                context_paths.append(str(out_wrist.relative_to(self.run_dir)))

        return {
            "memory": memory_paths,
            "recent": context_paths,
        }

    def _append_trace(self, record):
        if self._trace_fh is not None:
            self._trace_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._trace_fh.flush()

    def close(self):
        if self._trace_fh is not None:
            self._trace_fh.close()


def make_episode_logger(run_dir: pathlib.Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"task1_sync_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir / "sync_vlm.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def run_episode_async(
    env,
    client: _websocket_client_policy.WebsocketClientPolicy,
    prompt: str,
    planner: SyncLoRAPlanner,
    args: Args,
    vlm_camera_pose: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[bool, list[np.ndarray]]:
    """
     rollout（）:
    - VLM ， subtask buffer
    - VLA ， replan  buffer  subtask
    - VLM  buffer，VLA  buffer
    """
    obs = env.reset()
    current_subtask_prompt = ""
    replay = []
    recent_vlm_frames: deque[tuple[np.ndarray, Optional[np.ndarray]]] = deque(maxlen=args.n_recent)
    worker_error: list[str] = []
    worker_stop = threading.Event()
    vlm_job_queue: Optional[queue.Queue] = None
    vlm_thread: Optional[threading.Thread] = None
    subtask_lock = threading.Lock()
    subtask_buffer = {"value": "", "step_idx": -1}
    last_drop_log_step = -1

    def _write_subtask(step_idx: int, subtask: str) -> None:
        with subtask_lock:
            subtask_buffer["value"] = subtask
            subtask_buffer["step_idx"] = step_idx

    def _read_subtask() -> tuple[str, int]:
        with subtask_lock:
            return str(subtask_buffer["value"]), int(subtask_buffer["step_idx"])

    def _clone_recent_frames() -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
        return [
            (main.copy(), wrist.copy() if wrist is not None else None)
            for main, wrist in recent_vlm_frames
        ]

    def _submit_vlm_job(step_idx: int) -> None:
        nonlocal last_drop_log_step
        if not args.async_vlm or vlm_job_queue is None:
            return
        if step_idx < 0:
            return
        if len(recent_vlm_frames) < args.n_recent:
            return
        if args.vlm_interval > 1 and (step_idx % args.vlm_interval != 0):
            return

        payload = (step_idx, _clone_recent_frames())
        try:
            vlm_job_queue.put_nowait(payload)
            return
        except queue.Full:
            pass

        try:
            _ = vlm_job_queue.get_nowait()
        except queue.Empty:
            return

        try:
            vlm_job_queue.put_nowait(payload)
        except queue.Full:
            return

        if logger and step_idx != last_drop_log_step:
            logger.info(
                "[t=%s] VLM job queue ，，",
                step_idx + args.num_steps_wait,
            )
            last_drop_log_step = step_idx

    def _vlm_worker() -> None:
        assert vlm_job_queue is not None
        while not worker_stop.is_set():
            try:
                payload = vlm_job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            step_idx, context_frames = payload
            try:
                subtask = planner.infer_sync(step_idx=step_idx, context_frames_np=context_frames)
                if subtask:
                    _write_subtask(step_idx, subtask)
            except Exception as e:
                worker_error.append(f"{type(e).__name__}: {e}")
                if logger:
                    logger.error("VLM worker : %s", e, exc_info=True)
                break

    try:
        if args.async_vlm:
            queue_size = max(1, int(args.vlm_queue_size))
            vlm_job_queue = queue.Queue(maxsize=queue_size)
            vlm_thread = threading.Thread(
                target=_vlm_worker,
                name="task1-vlm-worker",
                daemon=True,
            )
            vlm_thread.start()
            if logger:
                logger.info(
                    "：single-slot subtask buffer + vlm_job_queue(maxsize=%s)",
                    queue_size,
                )

        t = 0

        while t < args.max_steps + args.num_steps_wait:
            if worker_error:
                raise RuntimeError(f"VLM worker failed: {worker_error[-1]}")

            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(
                    _extract_vlm_frame(env, obs, args, vlm_camera_pose)
                )
                t += 1
                _submit_vlm_job(t - args.num_steps_wait)
                continue

            effective_t = t - args.num_steps_wait
            if len(recent_vlm_frames) < args.n_recent:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(
                    _extract_vlm_frame(env, obs, args, vlm_camera_pose)
                )
                t += 1
                _submit_vlm_job(t - args.num_steps_wait)
                continue

            if args.async_vlm:
                _submit_vlm_job(effective_t)
                latest_subtask, latest_step = _read_subtask()
            else:
                latest_subtask = planner.infer_sync(
                    step_idx=effective_t,
                    context_frames_np=_clone_recent_frames(),
                )
                latest_step = effective_t

            if latest_subtask and latest_subtask != current_subtask_prompt:
                current_subtask_prompt = latest_subtask
                if logger:
                    if args.async_vlm:
                        logger.info(
                            "[t=%s]  subtask buffer  VLM (step=%s): %s",
                            t,
                            latest_step,
                            current_subtask_prompt,
                        )
                    else:
                        logger.info(f"[t={t}] VLM : {current_subtask_prompt}")
                if current_subtask_prompt == "place tomato into basket" and logger:
                    logger.info(f"[FINAL_HINT] t={t}, VLM ")

            element = obs_to_pi_element(
                obs,
                resize_size=args.resize_size,
                prompt=current_subtask_prompt or prompt,
            )
            out = client.infer(element)
            actions = out["actions"]
            assert len(actions) >= args.replan_steps

            if logger:
                logger.info(
                    f"[t={t}] VLA  chunk: {args.replan_steps} steps | prompt={current_subtask_prompt or prompt}"
                )

            for chunk_idx, action in enumerate(actions[:args.replan_steps], start=1):
                element_step = obs_to_pi_element(
                    obs,
                    resize_size=args.resize_size,
                    prompt=current_subtask_prompt or prompt,
                )
                replay.append(element_step["observation/image"])

                obs, _, done, info = env.step(action.tolist())
                recent_vlm_frames.append(
                    _extract_vlm_frame(env, obs, args, vlm_camera_pose)
                )
                t += 1
                _submit_vlm_job(t - args.num_steps_wait)

                # ， done 
                try:
                    if env.check_success():
                        if logger:
                            logger.info(f"[SUCCESS] t={t}, env.check_success()=True")
                        return True, replay
                except Exception:
                    #  check_success， done 
                    pass

                if done:
                    if logger:
                        logger.info(f"[DONE] t={t}, ！")
                    return True, replay

                if t >= args.max_steps + args.num_steps_wait:
                    break

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        return False, replay
    except Exception as e:
        logging.error(f"Episode : {e}", exc_info=True)
        return False, replay
    finally:
        if args.async_vlm and vlm_job_queue is not None:
            worker_stop.set()
            try:
                vlm_job_queue.put_nowait(None)
            except queue.Full:
                try:
                    _ = vlm_job_queue.get_nowait()
                    vlm_job_queue.put_nowait(None)
                except queue.Empty:
                    pass
            if vlm_thread is not None and vlm_thread.is_alive():
                vlm_thread.join(timeout=3.0)

    return False, replay


def eval_task1(args: Args):
    """Task1 （cookies + tomato sauce  basket）"""
    rid = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = pathlib.Path(args.log_base) / "task1_sync" / rid
    run_root.mkdir(parents=True, exist_ok=True)
    logging.info(f" run_id: {rid}，: {run_root}")
    logging.info(f"BDDL: {args.bddl_file}")
    logging.info(
        "VLA env render size: %sx%s | VLM render size: %sx%s",
        args.env_img_width,
        args.env_img_height,
        args.vlm_render_width,
        args.vlm_render_height,
    )
    logging.info(
        "VLA preprocess: obs_to_pi_element -> flipud + resize_size=%s",
        args.resize_size,
    )
    if args.vlm_match_vla_preprocess:
        logging.info(
            "VLM preprocess: match VLA exactly -> agentview + wrist -> flipud + resize_size=%s",
            args.resize_size,
        )
    else:
        logging.info(
            "VLM preprocess: raw agentview(+wrist) -> flipud -> fullvlm_square=%sx%s -> train_size=%sx%s",
            args.vlm_source_square_size,
            args.vlm_source_square_size,
            args.vlm_train_width,
            args.vlm_train_height,
        )
    if args.vlm_match_training_jpeg_roundtrip:
        logging.info(
            "VLM postprocess: JPEG roundtrip aligned to training -> quality=%s",
            args.vlm_training_jpeg_quality,
        )
    if _libero_path:
        logging.info(f"Libero path from TARGET_LIBERO_PATH: {_libero_path}")
    else:
        logging.info("Libero path: using uv/python default import")
    if not args.video_out_path:
        args.video_out_path = str(run_root / "videos")
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    _seed_everywhere(args.seed)
    _apply_vlm_prompt_profile(args)
    _apply_vlm_input_profile(args)
    logging.info(
        "VLM prompt profile: %s | use_keyframe_memory=%s | n_recent=%s | use_wrist=%s",
        args.vlm_prompt_profile,
        args.vlm_use_keyframe_memory,
        args.n_recent,
        args.vlm_use_wrist,
    )
    logging.info(
        "VLM input profile: %s",
        args.vlm_input_profile,
    )
    if args.vlm_match_fullvlm_source_square and args.vlm_resize_for_training:
        if (
            args.vlm_source_square_size == args.vlm_train_width
            and args.vlm_source_square_size == args.vlm_train_height
        ):
            logging.info(
                "VLM second resize is effectively no-op: source_square=%sx%s == train_size=%sx%s",
                args.vlm_source_square_size,
                args.vlm_source_square_size,
                args.vlm_train_width,
                args.vlm_train_height,
            )

    #  VLA server
    logging.info(" VLA server: ws://%s:%d ...", args.host, args.port)
    client = StableWebsocketClientPolicy(
        args.host,
        args.port,
        ping_interval=args.websocket_ping_interval,
        ping_timeout=args.websocket_ping_timeout,
        close_timeout=args.websocket_close_timeout,
    )

    system_prompt = (
        SYSTEM_PROMPT_TASK1_PLACEIT_NOKF
        if args.vlm_prompt_profile.strip().lower() == "task1_placeit_nokf"
        else SYSTEM_PROMPT_TASK1_KF5
    )

    #  VLM planner（ LoRA）
    planner = SyncLoRAPlanner(
        base_model_dir=args.base_model_dir,
        lora_path=args.lora_path,
        instruction=args.task_prompt,
        system_prompt=system_prompt,
        prompt_profile=args.vlm_prompt_profile,
        n_recent=args.n_recent,
        d_merge=args.d_merge,
        k_max=args.k_max,
        use_keyframe_memory=args.vlm_use_keyframe_memory,
        max_new_tokens=args.max_new_tokens,
        device=args.vlm_device,
        vlm_model_type=args.vlm_model_type,
        enable_thinking=args.enable_thinking,
        crop_right_half=args.crop_right_half,
        use_wrist=args.vlm_use_wrist,
    )
    planner.vlm_training_jpeg_roundtrip = args.vlm_match_training_jpeg_roundtrip
    planner.vlm_training_jpeg_quality = args.vlm_training_jpeg_quality

    # 
    logging.info(f" BDDL: {args.bddl_file}")
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=args.bddl_file,
            camera_heights=args.env_img_height,
            camera_widths=args.env_img_width,
            ignore_done=True,
            reward_shaping=True,
            control_freq=20,
            initialization_noise=None,
        )
        _ = env.reset()
    except Exception as e:
        logging.error(f": {e}")
        return

    vlm_camera_pose = _resolve_openpi_agentview_pose(args.bddl_file)
    try:
        vla_cam_pos, vla_cam_quat = _get_camera_pose(env, "agentview")
        logging.info(
            "VLA agentview (RoboMemArena default): pos=%s quat=%s",
            np.asarray(vla_cam_pos).tolist(),
            np.asarray(vla_cam_quat).tolist(),
        )
    except Exception as e:
        logging.warning(f" RoboMemArena agentview : {e}")

    if args.vlm_use_openpi_camera_pose and vlm_camera_pose is not None:
        logging.info(
            "VLM agentview override (eval fixed pose): pos=%s quat=%s source=%s",
            vlm_camera_pose["pos"],
            vlm_camera_pose["quat"],
            vlm_camera_pose["source"],
        )
    elif args.vlm_use_openpi_camera_pose:
        logging.warning(
            " BDDL  openpi ，VLM  RoboMemArena  agentview"
        )
    elif args.vlm_match_vla_preprocess:
        logging.info("VLM agentview: exact same live obs path as VLA.")
    else:
        logging.info("VLM agentview: use the same RoboMemArena default pose as VLA.")

    prompt = args.task_prompt
    total_success = 0

    for ep in tqdm.tqdm(range(args.num_trials_per_task), desc="task1"):
        run_dir = run_root / f"ep{ep}"
        episode_logger = make_episode_logger(run_dir)

        planner.reset_episode(
            instruction=prompt,
            run_dir=run_dir,
            logger=episode_logger,
        )

        try:
            success, replay = run_episode_async(
                env, client, prompt, planner, args,
                vlm_camera_pose=vlm_camera_pose,
                logger=episode_logger,
            )
        except Exception as e:
            logging.exception(f"Episode {ep} : {e}")
            success, replay = False, []

        # 
        suffix = "success" if success else "failure"
        out_name = f"task1_{suffix}_ep{ep}.mp4"
        if replay:
            try:
                _write_video(pathlib.Path(args.video_out_path) / out_name, replay, fps=10)
            except Exception:
                logging.exception(": %s", pathlib.Path(args.video_out_path) / out_name)

        total_success += int(success)
        episode_logger.info(f"Episode {ep}: {'SUCCESS' if success else 'FAILURE'}")

    try:
        env.close()
    except Exception:
        pass

    planner.close()

    success_rate = total_success / max(1, args.num_trials_per_task)
    logging.warning(
        f"[Task1 ] ={total_success}/{args.num_trials_per_task} ({success_rate:.2%})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_task1(args)
