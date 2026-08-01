#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import pathlib
import sys
from collections import deque
from typing import Optional
import io
import re

import numpy as np
from PIL import Image

from official_stage_tracker import OrderedOfficialStageTracker


BASE_EVAL_PY = pathlib.Path(
    os.environ.get(
        "TASK1_BASE_EVAL_PY",
        "${ASSET_OPENPI_ROOT}/examples/robocerebra_drawer_hkust/eval_task1_qwen3_async_openpi_inference_vla_cam.py",
    )
)

spec = importlib.util.spec_from_file_location("_task1_base_eval", BASE_EVAL_PY)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base eval from {BASE_EVAL_PY}")
_base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _base
spec.loader.exec_module(_base)

_official_scripts_dir = os.environ.get("ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR", "").strip()
if not _official_scripts_dir:
    raise RuntimeError("ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR is required")
OFFICIAL_SCRIPTS_DIR = pathlib.Path(_official_scripts_dir)
if str(OFFICIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_SCRIPTS_DIR))
official_spec = importlib.util.spec_from_file_location(
    "_robomemarena_official_eval_common_task1", OFFICIAL_SCRIPTS_DIR / "eval_common.py"
)
if official_spec is None or official_spec.loader is None:
    raise RuntimeError(f"Cannot load official eval_common from {OFFICIAL_SCRIPTS_DIR}")
official_ec = importlib.util.module_from_spec(official_spec)
sys.modules[official_spec.name] = official_ec
official_spec.loader.exec_module(official_ec)

_previous_eval_common = sys.modules.get("eval_common")
sys.modules["eval_common"] = official_ec
try:
    official_stage_spec = importlib.util.spec_from_file_location(
        "_robomemarena_official_task1_stage",
        OFFICIAL_SCRIPTS_DIR / "task2_26_reference_stage.py",
    )
    if official_stage_spec is None or official_stage_spec.loader is None:
        raise RuntimeError(f"Cannot load official task stage scorer from {OFFICIAL_SCRIPTS_DIR}")
    official_stage = importlib.util.module_from_spec(official_stage_spec)
    sys.modules[official_stage_spec.name] = official_stage
    official_stage_spec.loader.exec_module(official_stage)
finally:
    if _previous_eval_common is None:
        sys.modules.pop("eval_common", None)
    else:
        sys.modules["eval_common"] = _previous_eval_common


TASK1_ORDER = [
    "pick cookies",
    "place cookies into basket",
    "pick tomato sauce",
    "place tomato into basket",
]
FINAL_SUBTASK = "place tomato into basket"
DEFAULT_TARGET_JSON = (
    "${ASSET_OPENPI_INFERENCE_ROOT}/task1_eval/"
    "task1_subtask_end_poses_successindex_seed100_199.json"
)

SYSTEM_PROMPT_NOWRIST = (
    "You are a robotic planning assistant specialized in memory-based task understanding.\n\n"
    "Your task is to infer the next low-level policy primitive prompt from multi-image observation:\n"
    "1. Historical keyframes from earlier in the same full demonstration.\n"
    "2. A recent 5-frame main-camera visual window ending at the current frame.\n\n"
    "Important rules:\n"
    "- Historical keyframes are earlier than the current window.\n"
    "- The recent 5-frame window is the primary evidence for what the robot policy should execute next.\n"
    "- If there is no keyframe inside the current window, keyframe_positions must be an empty list.\n"
    "- Return strict JSON only. Do not output extra text.\n"
    "- Return exactly two fields:\n"
    "  - current_primitive: the next executable primitive prompt for the low-level policy, chosen from the predefined task1 primitive set.\n"
    "  - keyframe_positions: 1-indexed keyframe positions inside the recent 5-frame window.\n"
    "- At primitive boundaries, output the next executable primitive prompt rather than the primitive that merely describes the last completed motion.\n"
    "- Only agentview_rgb images are provided; eye_in_hand_rgb is intentionally omitted."
)

SYSTEM_PROMPT_CURRENT_REPEAT5 = (
    "You are a robotic planning assistant specialized in memory-based task understanding.\n\n"
    "Your task is to infer the next low-level policy primitive prompt from the current main-camera observation.\n\n"
    "Important rules:\n"
    "- The same current agentview_rgb frame is repeated across 5 image slots to emphasize the current state.\n"
    "- No historical keyframes and no temporal 5-frame motion context are provided in this variant.\n"
    "- keyframe_positions must be an empty list.\n"
    "- Return strict JSON only. Do not output extra text.\n"
    "- Return exactly two fields:\n"
    "  - current_primitive: the next executable primitive prompt for the low-level policy, chosen from the predefined task1 primitive set.\n"
    "  - keyframe_positions: always an empty list for this current-frame-only variant.\n"
    "- At primitive boundaries, output the next executable primitive prompt rather than the primitive that merely describes the last completed motion."
)

SYSTEM_PROMPT_KEYFRAMES_CURRENT_REPEAT5 = (
    "You are a robotic planning assistant specialized in memory-based task understanding.\n\n"
    "Your task is to infer the next low-level policy primitive prompt from multi-image observation:\n"
    "1. Historical keyframes from earlier in the same full demonstration.\n"
    "2. The current visual observation repeated across 5 timestep slots.\n\n"
    "Important rules:\n"
    "- Historical keyframes are earlier than the current repeated observation window.\n"
    "- The repeated current observation is the primary evidence for what the robot policy should execute next.\n"
    "- Each repeated timestep contains agentview_rgb followed by eye_in_hand_rgb.\n"
    "- keyframe_positions refer to 1-indexed positions inside the repeated 5-slot current-observation window.\n"
    "- If the current observation itself should be saved as a keyframe, use keyframe_positions [5]. Otherwise use an empty list.\n"
    "- Return strict JSON only. Do not output extra text.\n"
    "- Return exactly two fields:\n"
    "  - current_primitive: the next executable primitive prompt for the low-level policy, chosen from the predefined task1 primitive set.\n"
    "  - keyframe_positions: either [5] for the current observation or an empty list.\n"
    "- At primitive boundaries, output the next executable primitive prompt rather than the primitive that merely describes the last completed motion."
)

SYSTEM_PROMPT_NO_LABEL_NO_ORDER = (
    "You are a robotic visual planning assistant.\n\n"
    "Infer the next executable low-level robot action from the provided images only.\n"
    "Do not rely on a provided primitive list or a fixed task order. Use historical keyframes and the "
    "recent visual context to decide what action should be run now.\n\n"
    "Important rules:\n"
    "- Historical keyframes are earlier than the current window.\n"
    "- The recent 5-frame context is the primary evidence for the current state.\n"
    "- Return strict JSON only. Do not output extra text.\n"
    "- Return exactly two fields:\n"
    "  - current_primitive: a short natural-language low-level action prompt inferred from the images.\n"
    "  - keyframe_positions: a 1-indexed list of keyframe positions inside the recent 5-frame context, or an empty list."
)


_ORIG_APPLY_VLM_INPUT_PROFILE = _base._apply_vlm_input_profile
_ORIG_APPLY_VLM_PROMPT_PROFILE = _base._apply_vlm_prompt_profile


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _parse_task1_output_no_mapping(output_text: str, max_pos: int) -> tuple[str, list[int]]:
    s = output_text.strip()
    if "</think>" in s:
        s = s[s.rfind("</think>") + len("</think>"):].strip()
    if s.startswith("```"):
        lines = s.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    primitive = ""
    keyframe_positions: list[int] = []
    try:
        parsed = json.loads(s)
        primitive = str(parsed.get("current_primitive", parsed.get("current_subtask", ""))).strip()
        raw_positions = parsed.get("keyframe_positions", [])
        if isinstance(raw_positions, list):
            for p in raw_positions:
                try:
                    pi = int(p)
                except Exception:
                    continue
                if 1 <= pi <= max_pos:
                    keyframe_positions.append(pi)
    except Exception:
        primitive = s

    primitive = " ".join(primitive.lower().replace("_", " ").split())
    return primitive, keyframe_positions


def _set_fullvlm_256_preprocess(args: "_base.Args") -> None:
    args.vlm_match_fullvlm_source_square = True
    args.vlm_source_square_size = 256
    args.vlm_resize_for_training = True
    args.vlm_train_width = 256
    args.vlm_train_height = 256


def _apply_task1_context_prompt_profile(args: "_base.Args") -> None:
    profile = (args.vlm_prompt_profile or "").strip().lower()
    if profile == "task1_no_label_no_order":
        args.n_recent = 5
        args.vlm_use_keyframe_memory = True
        args.vlm_use_wrist = True
        return
    if profile == "task1_kf5_nowrist":
        args.vlm_use_wrist = False
        return
    if profile == "task1_current_repeat5":
        args.n_recent = 5
        args.k_max = 0
        args.vlm_use_keyframe_memory = False
        args.vlm_use_wrist = False
        return
    if profile == "task1_keyframes_current_repeat5":
        args.n_recent = 5
        args.vlm_use_keyframe_memory = True
        args.vlm_use_wrist = True
        return
    _ORIG_APPLY_VLM_PROMPT_PROFILE(args)


def _apply_task1_context_input_profile(args: "_base.Args") -> None:
    profile = (args.vlm_input_profile or "").strip().lower()
    if profile in {"main_nowrist", "nowrist", "fullvlm_256_nowrist"}:
        _set_fullvlm_256_preprocess(args)
        args.vlm_use_wrist = False
        args.vlm_wrist_required = False
        args.vlm_prompt_profile = "task1_kf5_nowrist"
        return
    if profile in {
        "current_agentview_repeat5",
        "current_repeat5",
        "agentview_repeat5",
    }:
        _set_fullvlm_256_preprocess(args)
        args.vlm_use_wrist = False
        args.vlm_wrist_required = False
        args.vlm_use_keyframe_memory = False
        args.k_max = 0
        args.n_recent = 5
        args.vlm_prompt_profile = "task1_current_repeat5"
        return
    if profile in {
        "keyframes_current_repeat5",
        "fullmem_current_repeat5",
        "fullvlm_256_keyframes_current_repeat5",
    }:
        _set_fullvlm_256_preprocess(args)
        args.vlm_use_wrist = True
        args.vlm_wrist_required = True
        args.vlm_use_keyframe_memory = True
        args.n_recent = 5
        args.vlm_prompt_profile = "task1_keyframes_current_repeat5"
        return
    _ORIG_APPLY_VLM_INPUT_PROFILE(args)


class Task1ContextVariantPlanner(_base.SyncLoRAPlanner):
    def infer_sync(
        self,
        step_idx: int,
        context_frames_np: list[tuple[np.ndarray, Optional[np.ndarray]]],
    ) -> str:
        if self.prompt_profile == "task1_current_repeat5" and context_frames_np:
            main_frame = context_frames_np[-1][0]
            context_frames_np = [(main_frame, None) for _ in range(5)]
        elif self.prompt_profile == "task1_keyframes_current_repeat5" and context_frames_np:
            main_frame, wrist_frame = context_frames_np[-1]
            context_frames_np = [
                (main_frame, wrist_frame.copy() if wrist_frame is not None else None)
                for _ in range(5)
            ]
        elif self.prompt_profile == "task1_kf5_nowrist":
            context_frames_np = [(frame_pack[0], None) for frame_pack in context_frames_np]
        if _env_bool("TASK1_ACCEPT_RAW_VLM_OUTPUT", False):
            return self._infer_sync_accept_raw(step_idx, context_frames_np)
        return super().infer_sync(step_idx, context_frames_np)

    def _infer_sync_accept_raw(
        self,
        step_idx: int,
        context_frames_np: list[tuple[np.ndarray, Optional[np.ndarray]]],
    ) -> str:
        if not context_frames_np:
            return self._current_subtask

        recent_start = step_idx - len(context_frames_np) + 1
        context_main_frames: list[Image.Image] = []
        context_wrist_frames: list[Optional[Image.Image]] = []
        for offset, frame_pack in enumerate(context_frames_np):
            abs_idx = recent_start + offset
            main_frame_np = frame_pack[0] if isinstance(frame_pack, tuple) else frame_pack
            wrist_frame_np = frame_pack[1] if isinstance(frame_pack, tuple) else None
            main_frame_img = (
                Image.fromarray(main_frame_np.astype(np.uint8))
                if isinstance(main_frame_np, np.ndarray)
                else main_frame_np
            )
            if self.crop_right_half:
                main_frame_img = _base._crop_right_half(main_frame_img)
            wrist_frame_img: Optional[Image.Image] = None
            if self.use_wrist and wrist_frame_np is not None:
                wrist_frame_img = (
                    Image.fromarray(wrist_frame_np.astype(np.uint8))
                    if isinstance(wrist_frame_np, np.ndarray)
                    else wrist_frame_np
                )
                if self.crop_right_half:
                    wrist_frame_img = _base._crop_right_half(wrist_frame_img)
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
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            images.extend(c["image"] for c in content if isinstance(c, dict) and c.get("type") == "image")

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        inputs = self.processor(text=[text], images=images if images else None, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with _base.torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
        out_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        vlm_subtask, j_rel = _parse_task1_output_no_mapping(out_text, max_pos=len(context_main_frames))
        j_abs = [recent_start + (p - 1) for p in j_rel]

        if self.use_keyframe_memory:
            self.J_hist.append(j_abs)
            raw_k_indices = _base.build_visual_memory(
                self.J_hist,
                t=self.step,
                N=len(context_main_frames),
                d=self.d_merge,
            )
            self.K_indices_abs = [idx for idx in raw_k_indices if idx < recent_start]
            self.K_main_frames = _base.get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
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
        if vlm_subtask:
            self._current_subtask = vlm_subtask
        subtask = self._current_subtask

        image_rel = None
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
        self._append_trace(
            {
                "t": int(step_idx),
                "subtask": subtask,
                "raw_output_mode": True,
                "keyframe_positions": j_rel,
                "J_abs": j_abs,
                "K_indices_abs": list(self.K_indices_abs),
                "out_text": out_text.strip()[:600],
                "image": image_rel,
            }
        )
        if self.logger:
            self.logger.info("VLM @t=%s raw_subtask='%s'", step_idx, subtask)
            self.logger.info("  raw=%s", out_text.strip()[:220])
        return self._current_subtask

    def _build_messages(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Optional[Image.Image]],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Optional[Image.Image]],
    ):
        if self.prompt_profile == "task1_current_repeat5":
            current_main = context_main_frames[-1]
            repeated_frames = [current_main for _ in range(5)]
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"{self.instruction}\n"
                        f"{_base.SCENE_DESCRIPTION}\n"
                        "Camera order for every timestep: agentview_rgb.\n"
                        "Current observation:\n"
                        "Current agentview_rgb frame repeated 5 times (5 images):"
                    ),
                }
            ]
            for frame in repeated_frames:
                user_content.append({"type": "image", "image": frame})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Return strict JSON with fields current_primitive and keyframe_positions. "
                        "Here current_primitive means the next low-level policy prompt; "
                        "keyframe_positions must be an empty list."
                    ),
                }
            )
            return [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_CURRENT_REPEAT5}]},
                {"role": "user", "content": user_content},
            ]

        if self.prompt_profile == "task1_keyframes_current_repeat5":
            current_main = context_main_frames[-1]
            current_wrist = context_wrist_frames[-1] if context_wrist_frames else None
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"{self.instruction}\n"
                        f"{_base.SCENE_DESCRIPTION}\n"
                        "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb.\n"
                        "Current observation:"
                    ),
                }
            ]
            memory_pairs = list(zip(memory_main_frames, memory_wrist_frames))
            if memory_pairs:
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            "Historical keyframes from earlier in the same demonstration "
                            f"({len(memory_pairs)} timesteps, {len(memory_pairs) * 2} images):"
                        ),
                    }
                )
                for main_frame, wrist_frame in memory_pairs:
                    user_content.append({"type": "image", "image": main_frame})
                    if wrist_frame is not None:
                        user_content.append({"type": "image", "image": wrist_frame})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Current observation repeated across 5 timestep slots "
                        "(10 images; each slot is agentview_rgb then eye_in_hand_rgb):"
                    ),
                }
            )
            for _ in range(5):
                user_content.append({"type": "image", "image": current_main})
                if current_wrist is not None:
                    user_content.append({"type": "image", "image": current_wrist})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Return strict JSON with fields current_primitive and keyframe_positions. "
                        "Here current_primitive means the next low-level policy prompt. "
                        "Use keyframe_positions [5] only if the current repeated observation should be saved as a keyframe; otherwise use []."
                    ),
                }
            )
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_KEYFRAMES_CURRENT_REPEAT5}],
                },
                {"role": "user", "content": user_content},
            ]

        if self.prompt_profile == "task1_no_label_no_order":
            use_wrist_images = self.use_wrist and any(
                frame is not None for frame in (memory_wrist_frames + context_wrist_frames)
            )
            camera_order = (
                "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb."
                if use_wrist_images
                else "Camera order for every timestep: agentview_rgb."
            )
            user_content = [
                {
                    "type": "text",
                    "text": (
                        "High-level objective: infer the next executable robot action from visual evidence only. "
                        "No candidate primitive list and no fixed execution order are provided.\n"
                        f"{_base.SCENE_DESCRIPTION}\n"
                        f"{camera_order}\n"
                        "Current observation:"
                    ),
                }
            ]
            if memory_main_frames:
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            "Historical keyframes from earlier in the same episode "
                            f"({len(memory_main_frames)} timesteps):"
                        ),
                    }
                )
                for idx, frame in enumerate(memory_main_frames):
                    user_content.append({"type": "image", "image": frame})
                    if use_wrist_images and idx < len(memory_wrist_frames) and memory_wrist_frames[idx] is not None:
                        user_content.append({"type": "image", "image": memory_wrist_frames[idx]})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Recent visual context: 5 consecutive frames ending at the current frame:"
                    ),
                }
            )
            for idx, frame in enumerate(context_main_frames):
                user_content.append({"type": "image", "image": frame})
                if use_wrist_images and idx < len(context_wrist_frames) and context_wrist_frames[idx] is not None:
                    user_content.append({"type": "image", "image": context_wrist_frames[idx]})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Return strict JSON with fields current_primitive and keyframe_positions. "
                        "The current_primitive should be the next executable low-level action inferred from the images."
                    ),
                }
            )
            return [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_NO_LABEL_NO_ORDER}]},
                {"role": "user", "content": user_content},
            ]

        if self.prompt_profile == "task1_kf5_nowrist":
            user_content = [
                {
                    "type": "text",
                    "text": (
                        f"{self.instruction}\n"
                        f"{_base.SCENE_DESCRIPTION}\n"
                        "Camera order for every timestep: agentview_rgb.\n"
                        "Current observation:"
                    ),
                }
            ]
            if memory_main_frames:
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            "Historical keyframes from earlier in the same demonstration "
                            f"({len(memory_main_frames)} timesteps, {len(memory_main_frames)} main-camera images):"
                        ),
                    }
                )
                for frame in memory_main_frames:
                    user_content.append({"type": "image", "image": frame})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Recent visual context: 5 consecutive frames ending at the current frame "
                        f"({len(context_main_frames)} main-camera images):"
                    ),
                }
            )
            for frame in context_main_frames:
                user_content.append({"type": "image", "image": frame})
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Return strict JSON with fields current_primitive and keyframe_positions. "
                        "Here current_primitive means the next low-level policy prompt "
                        "(1-indexed positions inside the recent 5-frame context)."
                    ),
                }
            )
            return [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_NOWRIST}]},
                {"role": "user", "content": user_content},
            ]

        return super()._build_messages(
            memory_main_frames,
            memory_wrist_frames,
            context_main_frames,
            context_wrist_frames,
        )


@dataclasses.dataclass
class Args(_base.Args):
    bddl_file: str = "${ASSET_OFFICIAL_RUNTIME_ROOT}/bddl/1_cookies_tomato_basket.bddl"
    resize_size: int = 256
    replan_steps: int = 5
    async_vlm: bool = False
    k_max: int = 0
    seed: int = 104
    num_trials_per_task: int = 1
    base_model_dir: str = (
        "${ASSET_OPENPI_INFERENCE_ROOT}/output/"
        "task1_qwen3_vl_fullvlm_v2_simpngpath_wristunlimited_breakfastprompt_lfp_fullft_freezevit_bs4_len8192_20260412_011556"
    )
    lora_path: str = "none"
    vlm_input_profile: str = "fullvlm_256"
    vlm_prompt_profile: str = "task1_kf5"
    vlm_match_vla_preprocess: bool = True
    vlm_match_training_jpeg_roundtrip: bool = False
    enable_endpose_hold: bool = True
    endpose_hold_targets_json: str = DEFAULT_TARGET_JSON
    endpose_hold_pos_tol: float = 0.04
    endpose_hold_min_active_steps: int = 20
    endpose_hold_consecutive: int = 2
    endpose_hold_disable_final: bool = True
    post_hold_release_vla_steps: int = 0
    prevent_subtask_regression: bool = False
    prevent_subtask_regression_after_hold_release: bool = False


def _normalize_subtask(subtask: str) -> str:
    try:
        return _base._normalize_primitive(subtask, allowed_subtasks=TASK1_ORDER)
    except Exception:
        return " ".join(str(subtask).strip().lower().split())


def _load_endpose_targets(args: Args) -> dict[str, dict]:
    if not args.enable_endpose_hold:
        return {}
    path = pathlib.Path(args.endpose_hold_targets_json).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"End-pose target JSON does not exist: {path}. "
            "Run compute_task1_subtask_end_poses.py first."
        )

    raw = json.loads(path.read_text())
    raw_subtasks = raw.get("subtasks", raw)
    targets: dict[str, dict] = {}
    for name, payload in raw_subtasks.items():
        subtask = _normalize_subtask(name)
        pos = payload.get("target_ee_pos") or payload.get("ee_pos") or payload.get("median_ee_pos")
        if pos is None:
            raise ValueError(f"{path}: missing target_ee_pos for {name}")
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos_arr.size < 3:
            raise ValueError(f"{path}: invalid target_ee_pos for {name}: {pos}")
        hold_gripper = float(payload.get("hold_gripper", -1.0))
        targets[subtask] = {
            "target_ee_pos": pos_arr[:3],
            "hold_gripper": 1.0 if hold_gripper >= 0.0 else -1.0,
        }
    return targets


def _get_eef_pos(obs: dict) -> np.ndarray:
    for key in ("robot0_eef_pos", "ee_pos"):
        if key in obs:
            value = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if value.size >= 3:
                return value[:3]
    raise KeyError(f"Cannot find EEF position in obs keys={sorted(obs.keys())}")


def _task1_order_index(subtask: str) -> Optional[int]:
    try:
        return TASK1_ORDER.index(_normalize_subtask(subtask))
    except ValueError:
        return None


def _distance_to_target(obs: dict, target: dict) -> float:
    return float(np.linalg.norm(_get_eef_pos(obs) - target["target_ee_pos"]))


def _check_success_or_done(env, done: bool, t: int, logger: Optional[logging.Logger]) -> bool:
    try:
        if env.check_success():
            if logger:
                logger.info("[SUCCESS] t=%s, env.check_success()=True", t)
            return True
    except Exception:
        pass
    if done:
        if logger:
            logger.info("[DONE] t=%s, task done", t)
        return True
    return False


def run_episode_sync_endpose_hold(
    env,
    client,
    prompt: str,
    planner,
    args: Args,
    vlm_camera_pose: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[bool, list[np.ndarray]]:
    obs = env.reset()
    official_stage_tracker = OrderedOfficialStageTracker(official_stage, task_id=1, env=env)
    official_goal_monitor = official_ec._build_goal_monitor_dict(pathlib.Path(args.bddl_file))
    official_ever_goal_success = False
    current_subtask_prompt = ""
    current_subtask_start_t = 0
    endpose_streak = 0
    hold_active = False
    hold_subtask = ""
    regression_guard_active = not args.prevent_subtask_regression_after_hold_release
    blocked_after_hold_prompts: set[str] = set()
    hold_prompt_counts: dict[str, int] = {}
    disable_output_normalize = _env_bool("TASK1_DISABLE_OUTPUT_NORMALIZE", False)
    replay: list[np.ndarray] = []
    recent_vlm_frames: deque[tuple[np.ndarray, Optional[np.ndarray]]] = deque(
        maxlen=args.n_recent
    )
    targets = _load_endpose_targets(args)

    def step_and_score(action):
        nonlocal obs, official_ever_goal_success
        obs, reward, done, info = env.step(action)
        for name in official_stage_tracker.update(env=env, obs=obs):
            if logger:
                logger.info("[OFFICIAL_STAGE_DONE] stage=%s", name)
        official_ever_goal_success = official_ever_goal_success or official_ec.check_goal_success(
            env, official_goal_monitor
        )
        return obs, reward, done, info

    def finish(success: bool) -> tuple[bool, list[np.ndarray]]:
        official_stage_done = official_stage_tracker.done
        score_pct = official_stage._stage_score_pct(1, official_stage_done)
        stage_success = official_stage._stage_success_from_stage_done(1, official_stage_done)
        if logger:
            logger.info(
                "[OFFICIAL_SCORE] task=1 average_score_pct=%.6f stage_success=%s goal_success=%s stage_done_json=%s",
                score_pct,
                int(stage_success),
                int(official_ever_goal_success),
                json.dumps(official_stage_done, ensure_ascii=False, separators=(",", ":")),
            )
        return success, replay

    def most_common_hold_prompt() -> str:
        if not hold_prompt_counts:
            return hold_subtask
        return max(
            hold_prompt_counts.items(),
            key=lambda item: (item[1], 1 if item[0] == hold_subtask else 0, item[0]),
        )[0]

    def clone_recent_frames() -> list[tuple[np.ndarray, Optional[np.ndarray]]]:
        return [
            (main.copy(), wrist.copy() if wrist is not None else None)
            for main, wrist in recent_vlm_frames
        ]

    def append_vlm_frame() -> None:
        recent_vlm_frames.append(
            _base._extract_vlm_frame(env, obs, args, vlm_camera_pose)
        )

    def can_hold(subtask: str) -> bool:
        if not args.enable_endpose_hold:
            return False
        if not subtask or subtask not in targets:
            return False
        if args.endpose_hold_disable_final and subtask == FINAL_SUBTASK:
            return False
        return True

    def maybe_update_endpose_streak(subtask: str, phase: str, t_now: int) -> bool:
        nonlocal endpose_streak
        if subtask not in targets:
            return False
        dist = _distance_to_target(obs, targets[subtask])
        active_steps = max(0, t_now - current_subtask_start_t)
        final_no_hold = args.endpose_hold_disable_final and subtask == FINAL_SUBTASK
        should_count = (
            can_hold(subtask)
            and active_steps >= args.endpose_hold_min_active_steps
            and dist <= args.endpose_hold_pos_tol
        )
        endpose_streak = endpose_streak + 1 if should_count else 0
        if logger:
            if final_no_hold:
                logger.info(
                    "[ENDPOSE_FINAL_LOG] t=%s subtask=%s dist=%.5f tol=%.5f phase=%s",
                    t_now,
                    subtask,
                    dist,
                    args.endpose_hold_pos_tol,
                    phase,
                )
            elif should_count or dist <= args.endpose_hold_pos_tol:
                logger.info(
                    "[ENDPOSE_NEAR] t=%s subtask=%s dist=%.5f tol=%.5f "
                    "active_steps=%s streak=%s/%s phase=%s",
                    t_now,
                    subtask,
                    dist,
                    args.endpose_hold_pos_tol,
                    active_steps,
                    endpose_streak,
                    args.endpose_hold_consecutive,
                    phase,
                )
        return can_hold(subtask) and endpose_streak >= args.endpose_hold_consecutive

    def run_vla_without_vlm(step_budget: int, phase: str) -> Optional[bool]:
        nonlocal obs, t
        remaining = max(0, int(step_budget))
        if remaining <= 0:
            return None
        if logger:
            logger.info(
                "[POST_HOLD_RELEASE_VLA_START] t=%s subtask=%s steps=%s phase=%s",
                t,
                current_subtask_prompt or prompt,
                remaining,
                phase,
            )
        while remaining > 0 and t < args.max_steps + args.num_steps_wait:
            element = _base.obs_to_pi_element(
                obs,
                resize_size=args.resize_size,
                prompt=current_subtask_prompt or prompt,
            )
            out = client.infer(element)
            actions = out["actions"]
            if len(actions) <= 0:
                raise RuntimeError("VLA returned an empty action chunk")
            chunk_len = min(len(actions), remaining, args.max_steps + args.num_steps_wait - t)
            if logger:
                logger.info(
                    "[POST_HOLD_RELEASE_VLA_CHUNK] t=%s subtask=%s chunk_steps=%s remaining_before=%s",
                    t,
                    current_subtask_prompt or prompt,
                    chunk_len,
                    remaining,
                )
            for action in actions[:chunk_len]:
                element_step = _base.obs_to_pi_element(
                    obs,
                    resize_size=args.resize_size,
                    prompt=current_subtask_prompt or prompt,
                )
                replay.append(element_step["observation/image"])
                obs, _, done, _ = step_and_score(action.tolist())
                append_vlm_frame()
                t += 1
                remaining -= 1
                if _check_success_or_done(env, done, t, logger):
                    return True
        if logger:
            logger.info(
                "[POST_HOLD_RELEASE_VLA_END] t=%s subtask=%s phase=%s",
                t,
                current_subtask_prompt or prompt,
                phase,
            )
        return None

    if logger:
        logger.info(
            "sync endpose-hold rollout: replan_steps=%s hold=%s tol=%.5f "
            "min_active_steps=%s consecutive=%s post_hold_release_vla_steps=%s "
            "prevent_subtask_regression=%s "
            "prevent_subtask_regression_after_hold_release=%s "
            "regression_guard_mode=hold_majority_prompt "
            "disable_output_normalize=%s raw_output_mode=%s prompt_profile=%s targets=%s",
            args.replan_steps,
            args.enable_endpose_hold,
            args.endpose_hold_pos_tol,
            args.endpose_hold_min_active_steps,
            args.endpose_hold_consecutive,
            args.post_hold_release_vla_steps,
            args.prevent_subtask_regression,
            args.prevent_subtask_regression_after_hold_release,
            disable_output_normalize,
            _env_bool("TASK1_ACCEPT_RAW_VLM_OUTPUT", False),
            args.vlm_prompt_profile,
            sorted(targets.keys()),
        )

    try:
        t = 0
        while t < args.max_steps + args.num_steps_wait:
            if t < args.num_steps_wait or len(recent_vlm_frames) < args.n_recent:
                obs, _, done, _ = step_and_score(_base.LIBERO_DUMMY_ACTION)
                append_vlm_frame()
                t += 1
                if _check_success_or_done(env, done, t, logger):
                    return finish(True)
                continue

            effective_t = t - args.num_steps_wait
            latest_subtask = planner.infer_sync(
                step_idx=effective_t,
                context_frames_np=clone_recent_frames(),
            )
            if disable_output_normalize:
                latest_subtask = " ".join(str(latest_subtask).strip().lower().replace("_", " ").split())
            else:
                latest_subtask = _normalize_subtask(latest_subtask)

            if (
                args.prevent_subtask_regression
                and regression_guard_active
                and current_subtask_prompt
                and latest_subtask
            ):
                if latest_subtask != current_subtask_prompt and latest_subtask in blocked_after_hold_prompts:
                    if logger:
                        logger.info(
                            "[SUBTASK_REGRESSION_BLOCKED] t=%s raw_subtask=%s "
                            "current_subtask=%s guard_mode=hold_majority_prompt blocked_prompts=%s",
                            t,
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
                if released_from_hold:
                    block_prompt = most_common_hold_prompt()
                    if block_prompt:
                        blocked_after_hold_prompts.add(block_prompt)
                    if logger:
                        logger.info(
                            "[ENDPOSE_HOLD_RELEASE] t=%s old_subtask=%s new_subtask=%s "
                            "blocked_after_release=%s hold_prompt_counts=%s",
                            t,
                            released_hold_subtask,
                            current_subtask_prompt,
                            block_prompt,
                            dict(sorted(hold_prompt_counts.items())),
                        )
                hold_active = False
                hold_subtask = ""
                hold_prompt_counts.clear()
                if released_from_hold and args.prevent_subtask_regression_after_hold_release:
                    regression_guard_active = True
                    if logger:
                        logger.info(
                            "[SUBTASK_REGRESSION_GUARD_ON] t=%s trigger=hold_release subtask=%s",
                            t,
                            current_subtask_prompt,
                        )
                if logger:
                    logger.info(
                        "[t=%s] VLM sync subtask update: %s -> %s",
                        t,
                        previous or "<none>",
                        current_subtask_prompt,
                    )
                if current_subtask_prompt == FINAL_SUBTASK and logger:
                    logger.info("[FINAL_HINT] t=%s, VLM reached final subtask.", t)
                if released_from_hold and args.post_hold_release_vla_steps > 0:
                    success = run_vla_without_vlm(
                        args.post_hold_release_vla_steps,
                        phase="after_hold_release",
                    )
                    if success is True:
                        return finish(True)
                    continue

            if hold_active:
                target = targets.get(hold_subtask)
                hold_gripper = float(target["hold_gripper"]) if target else -1.0
                hold_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, hold_gripper]
                if logger:
                    logger.info(
                        "[ENDPOSE_HOLD_STEP] t=%s subtask=%s hold_steps=%s gripper=%+.0f",
                        t,
                        hold_subtask,
                        args.replan_steps,
                        hold_gripper,
                    )
                for _ in range(args.replan_steps):
                    element_step = _base.obs_to_pi_element(
                        obs,
                        resize_size=args.resize_size,
                        prompt=current_subtask_prompt or prompt,
                    )
                    replay.append(element_step["observation/image"])
                    obs, _, done, _ = step_and_score(hold_action)
                    append_vlm_frame()
                    t += 1
                    if _check_success_or_done(env, done, t, logger):
                        return finish(True)
                    if t >= args.max_steps + args.num_steps_wait:
                        break
                continue

            if maybe_update_endpose_streak(current_subtask_prompt, "before_vla", t):
                hold_active = True
                hold_subtask = current_subtask_prompt
                hold_prompt_counts.clear()
                if hold_subtask:
                    hold_prompt_counts[hold_subtask] = 1
                if logger:
                    logger.info(
                        "[ENDPOSE_HOLD_START] t=%s subtask=%s source=before_vla",
                        t,
                        hold_subtask,
                    )
                continue

            element = _base.obs_to_pi_element(
                obs,
                resize_size=args.resize_size,
                prompt=current_subtask_prompt or prompt,
            )
            out = client.infer(element)
            actions = out["actions"]
            assert len(actions) >= args.replan_steps

            if logger:
                logger.info(
                    "[t=%s] VLA sync chunk: %s steps | prompt=%s",
                    t,
                    args.replan_steps,
                    current_subtask_prompt or prompt,
                )

            for chunk_idx, action in enumerate(actions[: args.replan_steps], start=1):
                element_step = _base.obs_to_pi_element(
                    obs,
                    resize_size=args.resize_size,
                    prompt=current_subtask_prompt or prompt,
                )
                replay.append(element_step["observation/image"])
                obs, _, done, _ = step_and_score(action.tolist())
                append_vlm_frame()
                t += 1

                if _check_success_or_done(env, done, t, logger):
                    return finish(True)

                if maybe_update_endpose_streak(
                    current_subtask_prompt,
                    f"after_vla_chunk{chunk_idx}",
                    t,
                ):
                    hold_active = True
                    hold_subtask = current_subtask_prompt
                    hold_prompt_counts.clear()
                    if hold_subtask:
                        hold_prompt_counts[hold_subtask] = 1
                    if logger:
                        logger.info(
                            "[ENDPOSE_HOLD_START] t=%s subtask=%s source=after_vla_chunk%s",
                            t,
                            hold_subtask,
                            chunk_idx,
                        )
                    break

                if t >= args.max_steps + args.num_steps_wait:
                    break

    except KeyError as exc:
        logging.error("KeyError during sync endpose-hold episode: %s", exc)
        return finish(False)
    except Exception as exc:
        logging.error("Episode failed during sync endpose-hold rollout: %s", exc, exc_info=True)
        return finish(False)

    return finish(False)


def _write_official_task1_summary(args: Args) -> None:
    run_root = pathlib.Path(args.log_base) / "task1_sync" / args.run_id
    pattern = re.compile(
        r"\[OFFICIAL_SCORE\] task=1 average_score_pct=([0-9.]+) "
        r"stage_success=([01]) goal_success=([01]) stage_done_json=(\{.*\})"
    )
    rows = []
    for log_path in sorted(run_root.glob("ep*/sync_vlm.log")):
        matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="ignore"))
        if not matches:
            continue
        score, stage_success, goal_success, stage_json = matches[-1]
        ep_match = re.search(r"ep(\d+)$", log_path.parent.name)
        ep = int(ep_match.group(1)) if ep_match else len(rows)
        rows.append(
            {
                "task_id": 1,
                "ep": ep,
                "seed": int(args.seed) + ep,
                "score_pct": float(score),
                "tsr_success": bool(int(stage_success)),
                "stage_success": bool(int(stage_success)),
                "goal_success": bool(int(goal_success)),
                "stage_done": json.loads(stage_json),
                "log": str(log_path),
            }
        )
    rows.sort(key=lambda row: row["ep"])
    with (run_root / "official_episodes.tsv").open("w", encoding="utf-8") as handle:
        handle.write("task_id\tep\tseed\tscore_pct\ttsr_success\tstage_success\tgoal_success\tlog\n")
        for row in rows:
            handle.write(
                f'1\t{row["ep"]}\t{row["seed"]}\t{row["score_pct"]:.1f}\t'
                f'{"Y" if row["tsr_success"] else "N"}\t{"Y" if row["stage_success"] else "N"}\t'
                f'{"Y" if row["goal_success"] else "N"}\t{row["log"]}\n'
            )
    n = len(rows)
    summary = {
        "task_id": 1,
        "num_trials": n,
        "seed_start": int(args.seed),
        "average_score_pct": sum(row["score_pct"] for row in rows) / max(1, n),
        "tsr_success_rate_pct": 100.0 * sum(row["tsr_success"] for row in rows) / max(1, n),
        "stage_success_rate_pct": 100.0 * sum(row["stage_success"] for row in rows) / max(1, n),
        "goal_success_rate_pct": 100.0 * sum(row["goal_success"] for row in rows) / max(1, n),
    }
    (run_root / "official_summary.json").write_text(
        json.dumps({"episodes": rows, "tasks": [summary]}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (run_root / "official_task_summary.tsv").open("w", encoding="utf-8") as handle:
        handle.write(
            "task_id\tnum_trials\tseed_start\taverage_score_pct\ttsr_success_rate_pct\t"
            "stage_success_rate_pct\tgoal_success_rate_pct\n"
        )
        handle.write(
            f'1\t{n}\t{args.seed}\t{summary["average_score_pct"]:.1f}\t'
            f'{summary["tsr_success_rate_pct"]:.1f}\t{summary["stage_success_rate_pct"]:.1f}\t'
            f'{summary["goal_success_rate_pct"]:.1f}\n'
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    _base.run_episode_async = run_episode_sync_endpose_hold
    _base._apply_vlm_prompt_profile = _apply_task1_context_prompt_profile
    _base._apply_vlm_input_profile = _apply_task1_context_input_profile
    _base.SyncLoRAPlanner = Task1ContextVariantPlanner
    args = _base.tyro.cli(Args)
    args.async_vlm = False
    _base.eval_task1(args)
    _write_official_task1_summary(args)


if __name__ == "__main__":
    main()
