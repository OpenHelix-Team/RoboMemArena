from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import tqdm
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: F401


PACK_DIR = Path(__file__).resolve().parent.parent
ROBOMEMARENA_SOURCE_ROOT = Path(
    os.environ.get(
        "ROBOMEMARENA_SOURCE_ROOT",
        os.environ["SOURCE_ROOT"],
    )
)
REFERENCE_DIR = ROBOMEMARENA_SOURCE_ROOT / "evaluation_benchmark" / "async_vlm26_reference"
EVAL_BENCHMARK_DIR = ROBOMEMARENA_SOURCE_ROOT / "evaluation_benchmark"
RUNTIME_DIR = EVAL_BENCHMARK_DIR / "openpi_minimal_runtime"
SCRIPTS_DIR = EVAL_BENCHMARK_DIR / "scripts"
ROOT = Path(os.environ["OPENPI_ROOT"])
INFERENCE_ROOT = Path(os.environ.get("OPENPI_INFERENCE_ROOT", str(ROOT.parent / "openpi_inference")))
OPENPI_CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
OPENPI_SRC = ROOT / "packages" / "openpi" / "src"
LIBERO_PATH_ENV = os.environ.get("TARGET_LIBERO_PATH", "").strip()
if not LIBERO_PATH_ENV:
    _fallback_libero = ROOT / "third_party" / "libero"
    if _fallback_libero.exists():
        LIBERO_PATH_ENV = str(_fallback_libero)
LIBERO_PATHS: list[Path] = []
if LIBERO_PATH_ENV:
    _libero_path = Path(LIBERO_PATH_ENV)
    LIBERO_PATHS.extend([_libero_path, _libero_path.parent])

module_paths = [
    str(RUNTIME_DIR),
    str(SCRIPTS_DIR),
    str(OPENPI_CLIENT_SRC),
    str(OPENPI_SRC),
]
for _lib_path in LIBERO_PATHS:
    module_paths.append(str(_lib_path))

for p in module_paths:
    if p and p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_GL", "egl")

import eval_common as ec
import task2_26_reference_stage as stage_eval
from eval_task1_qwen3_async_openpi_inference_vla_cam import (
    Args as BaseArgs,
    StableWebsocketClientPolicy,
    SyncLoRAPlanner,
    _apply_vlm_input_profile,
    _extract_vlm_frame,
    _seed_everywhere,
    _write_video,
    make_episode_logger,
)
from keyframe_selection import build_visual_memory, get_frames_from_indices
from robocerebra_adapter import obs_to_pi_element


SYSTEM_PROMPT_MEMORY_DEMO = """You are an embodied-memory robot VLM planner.

You will observe two kinds of visual evidence from the same long-horizon execution:
1. Historical keyframes: moments before the current step, used to remember important past states.
2. A recent 5-frame dual-camera window ending at the current frame, used to infer the current primitive.
Temporal order: historical keyframes are ordered from earliest to latest; the recent 5-frame window is also ordered from earliest to latest, and the last timestep in that window is the current frame.

Your goal is not to narrate the full execution. Your goal is to infer the primitive the robot is currently executing, or should execute now, from these images.

Important rules:
- Historical keyframes are always earlier than the recent visual window.
- If there is no keyframe in the recent window, keyframe_positions must be an empty list.
- keyframe_positions are 1-indexed positions within the recent 5-frame window.
- Output strict JSON only, with no extra text.
- The JSON must contain exactly two fields: current_primitive and keyframe_positions."""

SYSTEM_PROMPT_MEMORY_LONGTASK = SYSTEM_PROMPT_MEMORY_DEMO

SYSTEM_PROMPT_MEMORY = (
    SYSTEM_PROMPT_MEMORY_LONGTASK
    if os.environ.get("VLM_LONGTASK_PROMPT", "0") == "1"
    else SYSTEM_PROMPT_MEMORY_DEMO
)


@dataclass(frozen=True)
class TaskInfo:
    task_id: int
    suite: str
    task_name: str
    memory_type: str
    challenge: str
    brief_description: str
    task_block: str
    scene_description: str
    primitive_labels: list[str]


def load_task_infos(path: Path) -> dict[int, TaskInfo]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, TaskInfo] = {}
    for task in raw["tasks"]:
        out[int(task["task_id"])] = TaskInfo(
            task_id=int(task["task_id"]),
            suite=str(task["suite"]),
            task_name=str(task["task_name"]),
            memory_type=str(task["memory_type"]),
            challenge=str(task["challenge"]),
            brief_description=str(task["brief_description"]),
            task_block=str(task["task_block"]),
            scene_description=str(task.get("scene_description", "")),
            primitive_labels=[str(p["label"]) for p in task["primitive_order"]],
        )
    return out


def _parse_task_set_env(name: str) -> set[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, int):
            return {int(parsed)}
        if isinstance(parsed, list):
            return {int(x) for x in parsed}
    except Exception:
        pass
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def _stage_prompt_for(task_info: TaskInfo, stage_idx: int) -> str:
    labels = [label.strip() for label in task_info.primitive_labels if label.strip()]
    if not labels:
        return task_info.brief_description.strip()
    idx = min(max(0, int(stage_idx)), len(labels) - 1)
    return labels[idx]


class StagePromptPlanner:
    """Diagnostic planner that maps the official stage index to a VLA prompt."""

    def __init__(self, task_info: TaskInfo) -> None:
        self.logger = None
        self.set_task_info(task_info)

    def set_task_info(self, task_info: TaskInfo) -> None:
        self.task_info = task_info
        self.default_subtask_prompt = _stage_prompt_for(task_info, 0)

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None):
        del instruction, run_dir
        self.logger = logger

    def infer_sync(self, step_idx: int, context_frames_np: list[Any]) -> str:
        del step_idx, context_frames_np
        return self.default_subtask_prompt

    def close(self) -> None:
        return None


def _camera_order_text(use_wrist_images: bool) -> str:
    if use_wrist_images:
        return (
            "Camera order for every timestep: agentview_rgb, eye_in_hand_rgb. "
            "agentview_rgb is the external main-view camera, and eye_in_hand_rgb is the wrist/end-effector camera."
        )
    return "Camera: agentview_rgb. agentview_rgb is the external main-view camera."


def _parse_output_no_mapping(output_text: str, max_pos: int) -> tuple[str, list[int]]:
    """Parse VLM JSON output without any task-specific vocabulary mapping."""
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
        # Keep raw decoded text as-is when JSON parsing fails.
        primitive = s

    return primitive, keyframe_positions


class FullVlm26MemoryPlanner(SyncLoRAPlanner):
    def __init__(self, *args: Any, task_info: TaskInfo, **kwargs: Any) -> None:
        processor_model_dir = kwargs.pop("processor_model_dir", None)
        if processor_model_dir is None:
            processor_model_dir = os.environ.get("VLM_PROCESSOR_DIR", kwargs.get("base_model_dir", args[0] if args else ""))
        model_dir = Path(kwargs.get("base_model_dir", args[0] if args else ""))
        processor_dir = Path(processor_model_dir)
        if model_dir.is_dir() and processor_dir.is_dir():
            for name in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json"):
                src = processor_dir / name
                dst = model_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        super().__init__(*args, **kwargs)
        # Some DeepSpeed/Trainer checkpoints save model weights but not the image processor files.
        # Keep the trained weights from base_model_dir, but use the canonical Qwen3-VL processor.
        self.processor = AutoProcessor.from_pretrained(
            processor_model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.set_task_info(task_info)

    def set_task_info(self, task_info: TaskInfo) -> None:
        self.task_info = task_info
        self.default_subtask_prompt = task_info.brief_description.strip()
        self._current_subtask = self.default_subtask_prompt

    def reset_episode(self, instruction: str | None = None, run_dir=None, logger=None):
        super().reset_episode(instruction=instruction, run_dir=run_dir, logger=logger)
        self._current_subtask = self.default_subtask_prompt

    def _build_messages(
        self,
        memory_main_frames: list[Image.Image],
        memory_wrist_frames: list[Image.Image | None],
        context_main_frames: list[Image.Image],
        context_wrist_frames: list[Image.Image | None],
    ):
        use_wrist_images = self.use_wrist and any(
            frame is not None for frame in (memory_wrist_frames + context_wrist_frames)
        )
        num_history_keyframes = len(memory_main_frames)
        num_history_images = num_history_keyframes * (2 if use_wrist_images else 1)
        num_context_frames = len(context_main_frames)
        num_context_images = num_context_frames * (2 if use_wrist_images else 1)

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Global objective: infer the robot's current primitive action from historical keyframes before the current step and recent visual history within the same execution.\n\n"
                    "Task objective:\n"
                    f"{self.task_info.task_block}\n\n"
                    "Scene description:\n"
                    f"{self.task_info.scene_description or self.task_info.brief_description}\n\n"
                    f"{_camera_order_text(use_wrist_images)}\n"
                    "Current observation:"
                ),
            }
        ]

        def append_timestep_images(main_frames, wrist_frames) -> None:
            for idx, main_img in enumerate(main_frames):
                user_content.append({"type": "image", "image": main_img})
                if use_wrist_images:
                    wrist_img = wrist_frames[idx] if idx < len(wrist_frames) else None
                    if wrist_img is not None:
                        user_content.append({"type": "image", "image": wrist_img})

        if memory_main_frames:
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        "Historical keyframes from moments before the current step in the same execution "
                        f"({num_history_keyframes} timesteps, {num_history_images} images):"
                    ),
                }
            )
            append_timestep_images(memory_main_frames, memory_wrist_frames)

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
        append_timestep_images(context_main_frames, context_wrist_frames)
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Output strict JSON with exactly two fields: current_primitive and keyframe_positions. "
                    "keyframe_positions are 1-indexed keyframe positions inside the recent visual window."
                ),
            }
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def infer_sync(self, step_idx: int, context_frames_np: list[tuple[np.ndarray, np.ndarray | None]]) -> str:
        if not context_frames_np:
            return self._current_subtask

        recent_start = step_idx - len(context_frames_np) + 1
        context_main_frames: list[Image.Image] = []
        context_wrist_frames: list[Image.Image | None] = []
        for offset, frame_pack in enumerate(context_frames_np):
            abs_idx = recent_start + offset
            main_np, wrist_np = frame_pack
            main_img = Image.fromarray(main_np.astype(np.uint8))
            wrist_img = Image.fromarray(wrist_np.astype(np.uint8)) if self.use_wrist and wrist_np is not None else None
            self.frame_store_main[abs_idx] = main_img
            self.frame_store_wrist[abs_idx] = wrist_img
            context_main_frames.append(main_img)
            context_wrist_frames.append(wrist_img)
        self.step = max(self.step, step_idx + 1)

        memory_main_frames = list(self.K_main_frames) if self.use_keyframe_memory else []
        memory_wrist_frames = list(self.K_wrist_frames) if self.use_keyframe_memory else []
        memory_indices = list(self.K_indices_abs) if self.use_keyframe_memory else []
        messages = self._build_messages(memory_main_frames, memory_wrist_frames, context_main_frames, context_wrist_frames)

        images = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                images.extend(c["image"] for c in content if isinstance(c, dict) and c.get("type") == "image")

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if isinstance(text, list):
            text = text[0]
        inputs = self.processor(text=[text], images=images if images else None, return_tensors="pt", padding=False)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with __import__("torch").inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], gen)]
        out_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        vlm_subtask, j_rel = _parse_output_no_mapping(
            out_text,
            max_pos=len(context_main_frames),
        )
        j_abs = [recent_start + (p - 1) for p in j_rel]

        if self.use_keyframe_memory:
            self.J_hist.append(j_abs)
            raw_k_indices = build_visual_memory(self.J_hist, t=self.step, N=len(context_main_frames), d=self.d_merge)
            self.K_indices_abs = [idx for idx in raw_k_indices if idx < recent_start]
            self.K_main_frames = get_frames_from_indices(self.K_indices_abs, self.frame_store_main)
            self.K_wrist_frames = [self.frame_store_wrist.get(idx) for idx in self.K_indices_abs]
            if self.k_max > 0 and len(self.K_indices_abs) > self.k_max:
                self.K_indices_abs = self.K_indices_abs[-self.k_max:]
                self.K_main_frames = self.K_main_frames[-self.k_max:]
                self.K_wrist_frames = self.K_wrist_frames[-self.k_max:]

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
                "task_id": int(self.task_info.task_id),
                "subtask": subtask,
                "keyframe_positions": j_rel,
                "J_abs": j_abs,
                "K_indices_abs": list(self.K_indices_abs),
                "out_text": out_text.strip()[:600],
                "image": image_rel,
            }
        )
        if self.logger:
            self.logger.info("VLM @t=%s task=%s subtask=%s keyframes=%s", step_idx, self.task_info.task_id, subtask, j_rel)
            self.logger.info("  raw=%s", out_text.strip()[:220])
        return subtask


def _task_specs(task_id: int) -> list[stage_eval.StageSpec]:
    return stage_eval._task_specs(task_id)


def _goal_override_check(task_id: int):
    return stage_eval._goal_override_check(task_id)


def run_episode_async_stateful(
    *,
    task_id: int,
    env: Any,
    client: Any,
    planner: Any,
    args: BaseArgs,
    stage_specs: list[stage_eval.StageSpec],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    goal_check_override,
    vlm_camera_pose: dict | None,
    logger: logging.Logger,
    fail_on_extra_pour: bool,
    extra_pour_monitor_steps: int,
    post_goal_steps: int,
) -> tuple[float, dict[str, bool], bool | None, dict[str, Any], list[np.ndarray], list[np.ndarray]]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    recent_vlm_frames: deque[tuple[np.ndarray, np.ndarray | None]] = deque(maxlen=args.n_recent)
    worker_error: list[str] = []
    worker_stop = threading.Event()
    vlm_job_queue: queue.Queue | None = queue.Queue(maxsize=max(1, args.vlm_queue_size)) if args.async_vlm else None
    subtask_lock = threading.Lock()
    subtask_buffer = {"value": "", "step_idx": -1}
    stage_done = {spec.name: False for spec in stage_specs}
    stage_idx = 0
    all_stages_logged = False
    state: dict[str, Any] | None = None
    current_stage_start = 0
    current_subtask_prompt = ""
    counting_pour_task = stage_eval._is_counting_pour_task(task_id)
    drawer_task = stage_eval._is_drawer_task(task_id)
    goal_success: bool | None = None if counting_pour_task else False
    goal_reached_t: int | None = None
    extra_pour_check = stage_eval._extra_pour_check(task_id)
    extra_monitor_start_state_idx: int | None = None
    extra_monitor_deadline_t: int | None = None
    extra_pour_detected = False
    pour_1_step: int | None = None
    pour_2_step: int | None = None
    stage_prompt_override = task_id in getattr(args, "stage_prompt_override_tasks", set())
    stage_prompt_override_only = bool(getattr(args, "stage_prompt_override_only", False))

    def write_subtask(step_idx: int, subtask: str) -> None:
        with subtask_lock:
            subtask_buffer["value"] = subtask
            subtask_buffer["step_idx"] = step_idx

    def read_subtask() -> tuple[str, int]:
        with subtask_lock:
            return str(subtask_buffer["value"]), int(subtask_buffer["step_idx"])

    def clone_recent_frames() -> list[tuple[np.ndarray, np.ndarray | None]]:
        return [(m.copy(), w.copy() if w is not None else None) for m, w in recent_vlm_frames]

    def submit_vlm_job(step_idx: int) -> None:
        if not args.async_vlm or vlm_job_queue is None:
            return
        if step_idx < 0 or len(recent_vlm_frames) < args.n_recent:
            return
        if args.vlm_interval > 1 and step_idx % args.vlm_interval != 0:
            return
        payload = (step_idx, clone_recent_frames())
        try:
            vlm_job_queue.put_nowait(payload)
            return
        except queue.Full:
            try:
                vlm_job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                vlm_job_queue.put_nowait(payload)
            except queue.Full:
                return

    def vlm_worker() -> None:
        assert vlm_job_queue is not None
        while not worker_stop.is_set():
            try:
                payload = vlm_job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if payload is None:
                break
            step_idx, frames = payload
            try:
                subtask = planner.infer_sync(step_idx=step_idx, context_frames_np=frames)
                if subtask:
                    write_subtask(step_idx, subtask)
            except Exception as exc:
                worker_error.append(f"{type(exc).__name__}: {exc}")
                logger.error("VLM worker failed", exc_info=True)
                break

    vlm_thread = None
    if args.async_vlm and not (stage_prompt_override and stage_prompt_override_only):
        vlm_thread = threading.Thread(target=vlm_worker, name=f"vlm-task{planner.task_info.task_id}", daemon=True)
        vlm_thread.start()
        logger.info("async VLM enabled: single-slot subtask buffer")
    elif stage_prompt_override:
        logger.info(
            "stage prompt override enabled: task=%s only=%s labels=%s",
            task_id,
            stage_prompt_override_only,
            planner.task_info.primitive_labels,
        )

    try:
        t = 0
        while t < args.max_steps + args.num_steps_wait:
            if worker_error:
                raise RuntimeError(worker_error[-1])

            if t < args.num_steps_wait:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                t += 1
                submit_vlm_job(t - args.num_steps_wait)
                continue

            if state is None:
                state = stage_eval._build_initial_state(env)
                current_stage_start = state["step_idx"]

            effective_t = t - args.num_steps_wait
            if len(recent_vlm_frames) < args.n_recent:
                obs, _, _, _ = env.step(ec.LIBERO_DUMMY_ACTION)
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                t += 1
                submit_vlm_job(t - args.num_steps_wait)
                continue

            if stage_prompt_override and stage_prompt_override_only:
                latest_subtask = ""
                latest_step = -1
            elif args.async_vlm:
                submit_vlm_job(effective_t)
                latest_subtask, latest_step = read_subtask()
            else:
                latest_subtask = planner.infer_sync(effective_t, clone_recent_frames())
                latest_step = effective_t

            if latest_subtask and latest_subtask != current_subtask_prompt:
                current_subtask_prompt = latest_subtask
                logger.info("[t=%s] VLM prompt update from step=%s: %s", t, latest_step, current_subtask_prompt)

            if stage_prompt_override:
                stage_prompt = _stage_prompt_for(planner.task_info, stage_idx)
                if stage_prompt != current_subtask_prompt:
                    current_subtask_prompt = stage_prompt
                    logger.info(
                        "[t=%s] stage prompt override stage_idx=%s: %s",
                        t,
                        stage_idx,
                        current_subtask_prompt,
                    )

            prompt_for_vla = current_subtask_prompt or planner.default_subtask_prompt
            element = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
            out = client.infer(element)
            actions = np.asarray(out["actions"])
            logger.info("[t=%s] VLA chunk prompt=%s", t, prompt_for_vla)

            for action in actions[: args.replan_steps]:
                element_step = obs_to_pi_element(obs, resize_size=args.resize_size, prompt=prompt_for_vla)
                replay.append(element_step["observation/image"])
                wrist = element_step.get("observation/wrist_image")
                if wrist is not None:
                    replay_wrist.append(wrist)

                obs, _, done, _ = env.step(action.tolist())
                recent_vlm_frames.append(_extract_vlm_frame(env, obs, args, vlm_camera_pose))
                if state is not None:
                    stage_eval._update_state(obs, state)
                t += 1
                submit_vlm_job(t - args.num_steps_wait)

                if state is not None and stage_idx < len(stage_specs):
                    spec = stage_specs[stage_idx]
                    if spec.check_fn(env, state, current_stage_start):
                        stage_done[spec.name] = True
                        logger.info("[t=%s] stage done: %s", t, spec.name)
                        if spec.name.endswith("_Pour_One"):
                            pour_1_step = t
                        elif spec.name.endswith("_Pour_Two"):
                            pour_2_step = t
                            if counting_pour_task and fail_on_extra_pour:
                                extra_monitor_start_state_idx = int(state["step_idx"])
                                extra_monitor_deadline_t = t + extra_pour_monitor_steps
                                logger.info(
                                    "[t=%s] extra-pour monitor started; deadline=%s",
                                    t,
                                    extra_monitor_deadline_t,
                                )
                        stage_idx += 1
                        current_stage_start = state["step_idx"]

                if stage_idx >= len(stage_specs) and not all_stages_logged:
                    logger.info("[t=%s] all stages done", t)
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
                    logger.info("[t=%s] third pour detected; episode failed", t)
                    raise StopIteration

                if not counting_pour_task and stage_eval._stage_success_from_stage_done(task_id, stage_done):
                    goal_success = True
                    logger.info("[t=%s] required stages done", t)
                    raise StopIteration

                all_stages_complete = bool(stage_done) and all(stage_done.values())
                extra_monitor_complete = (
                    not fail_on_extra_pour
                    or (
                        extra_monitor_deadline_t is not None
                        and t >= extra_monitor_deadline_t
                    )
                )
                if counting_pour_task and all_stages_complete and extra_monitor_complete:
                    raise StopIteration
                if done or t >= args.max_steps + args.num_steps_wait:
                    raise StopIteration
    except StopIteration:
        pass
    except Exception:
        logger.exception("episode failed")
    finally:
        if args.async_vlm and vlm_job_queue is not None:
            worker_stop.set()
            try:
                vlm_job_queue.put_nowait(None)
            except queue.Full:
                pass
            if vlm_thread is not None and vlm_thread.is_alive():
                vlm_thread.join(timeout=3.0)

    stage_pct = stage_eval._stage_score_pct(task_id, stage_done)
    all_stages_complete = bool(stage_done) and all(stage_done.values())
    extra_monitor_complete = (
        not fail_on_extra_pour
        or (
            extra_monitor_deadline_t is not None
            and t >= extra_monitor_deadline_t
        )
    )
    required_stages_complete = stage_eval._stage_success_from_stage_done(task_id, stage_done)
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
    return stage_pct, stage_done, goal_success, diagnostics, replay, replay_wrist


def patch_env_resolution() -> None:
    base_env = ec._get_env_class()
    orig_init = base_env.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["camera_heights"] = 480
        kwargs["camera_widths"] = 640
        return orig_init(self, *args, **kwargs)

    base_env.__init__ = patched_init
    ec._get_env_class = lambda: base_env


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    patch_env_resolution()

    out_root = Path(os.environ["OUT_ROOT"])
    video_root = Path(os.environ["VIDEO_DIR"])
    summary_json = Path(os.environ["SUMMARY_JSON"])
    summary_tsv = Path(os.environ["SUMMARY_TSV"])
    prompt_trace_tsv = Path(os.environ.get("PROMPT_TRACE_TSV", str(out_root / "prompt_trace.tsv")))
    task_config = Path(os.environ.get("TASK_CONFIG", str(REFERENCE_DIR / "fullvlm_v2_26_memory_tasks.json")))
    task_infos = load_task_infos(task_config)
    tasks = [int(x) for x in json.loads(os.environ.get("TASKS_JSON", json.dumps(list(range(1, 27)))))]

    args = BaseArgs()
    args.host = os.environ.get("HOST", "127.0.0.1")
    args.port = int(os.environ.get("PORT", "8026"))
    args.base_model_dir = os.environ["VLM_CKPT"]
    args.lora_path = os.environ.get("VLM_LORA_PATH", "none")
    args.vlm_device = os.environ.get("VLM_DEVICE", "cuda:1")
    args.resize_size = int(os.environ.get("RESIZE_SIZE", "256"))
    args.replan_steps = int(os.environ.get("REPLAN_STEPS", "10"))
    args.num_steps_wait = int(os.environ.get("NUM_STEPS_WAIT", "10"))
    args.max_steps = int(os.environ.get("MAX_STEPS", "2500"))
    args.seed = int(os.environ.get("SEED", "100"))
    args.num_trials_per_task = int(os.environ.get("NUM_TRIALS", "1"))
    args.vlm_input_profile = os.environ.get("VLM_INPUT_PROFILE", "fullvlm_256")
    args.vlm_match_training_jpeg_roundtrip = os.environ.get("VLM_MATCH_TRAINING_JPEG_ROUNDTRIP", "0") in {"1", "true", "yes"}
    args.vlm_training_jpeg_quality = int(os.environ.get("VLM_TRAINING_JPEG_QUALITY", "30"))
    args.async_vlm = os.environ.get("ASYNC_VLM", "1") in {"1", "true", "yes"}
    args.vlm_interval = int(os.environ.get("VLM_INTERVAL", "5"))
    args.vlm_queue_size = int(os.environ.get("VLM_QUEUE_SIZE", "1"))
    args.n_recent = int(os.environ.get("N_RECENT", "5"))
    args.k_max = int(os.environ.get("K_MAX", "0"))
    args.d_merge = int(os.environ.get("D_MERGE", "6"))
    args.vlm_use_wrist = os.environ.get("VLM_USE_WRIST", "1") in {"1", "true", "yes"}
    args.vlm_use_keyframe_memory = os.environ.get("VLM_USE_KEYFRAME_MEMORY", "1") in {"1", "true", "yes"}
    args.stage_prompt_override_tasks = _parse_task_set_env("STAGE_PROMPT_OVERRIDE_TASKS")
    args.stage_prompt_override_only = os.environ.get("STAGE_PROMPT_OVERRIDE_ONLY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    fail_on_extra_pour = os.environ.get("FAIL_ON_EXTRA_POUR", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    extra_pour_monitor_steps = int(os.environ.get("POST_STAGE_STEPS", os.environ.get("EXTRA_POUR_MONITOR_STEPS", "30")))
    post_goal_steps = int(os.environ.get("POST_GOAL_STEPS", "200"))
    _apply_vlm_input_profile(args)

    out_root.mkdir(parents=True, exist_ok=True)
    video_root.mkdir(parents=True, exist_ok=True)
    prompt_trace_tsv.write_text(
        "task_id\ttrial\tseed\tvlm_ckpt\tvla_prompt_last\tstage_success\tgoal_success\t"
        "stage_score_pct\textra_pour_detected\tfailure_reason\n",
        encoding="utf-8",
    )
    summary_tsv.write_text(
        "task_id\tstatus\terror\tstage_score_pct\tstage_success_rate\tgoal_success_rate\t"
        "video_dir\tduration_sec\n",
        encoding="utf-8",
    )

    _seed_everywhere(args.seed)
    client = StableWebsocketClientPolicy(args.host, args.port, ping_interval=None, ping_timeout=None, close_timeout=30.0)
    first_task = task_infos[tasks[0]]
    if first_task.task_id in args.stage_prompt_override_tasks and args.stage_prompt_override_only:
        planner = StagePromptPlanner(first_task)
        logging.info("using StagePromptPlanner; VLM weights are not loaded")
    else:
        planner = FullVlm26MemoryPlanner(
            base_model_dir=args.base_model_dir,
            lora_path=args.lora_path,
            instruction="",
            system_prompt=SYSTEM_PROMPT_MEMORY,
            prompt_profile="task1_kf5",
            n_recent=args.n_recent,
            d_merge=args.d_merge,
            k_max=args.k_max,
            use_keyframe_memory=args.vlm_use_keyframe_memory,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "256")),
            device=args.vlm_device,
            logger=None,
            vlm_model_type=args.vlm_model_type,
            enable_thinking=False,
            crop_right_half=False,
            use_wrist=args.vlm_use_wrist,
            task_info=first_task,
        )

    results = []
    for task_id in tasks:
        task_info = task_infos[task_id]
        planner.set_task_info(task_info)
        bddl_path = ec._resolve_bddl_path(task_id)
        stage_specs = _task_specs(task_id)
        counting_pour_task = stage_eval._is_counting_pour_task(task_id)
        goal_monitor_dict = {} if counting_pour_task else ec._build_goal_monitor_dict(bddl_path)
        goal_check_override = _goal_override_check(task_id)
        task_video = video_root / f"task{task_id}"
        task_video.mkdir(parents=True, exist_ok=True)
        task_root = out_root / f"task{task_id}"
        task_root.mkdir(parents=True, exist_ok=True)
        status = "completed"
        err = ""
        st = time.time()
        stage_sum = 0.0
        goal_cnt = 0
        stage_success_cnt = 0

        try:
            env_cls = ec._get_env_class()
            env = env_cls(
                bddl_file_name=str(bddl_path),
                camera_heights=480,
                camera_widths=640,
                ignore_done=True,
                reward_shaping=True,
                control_freq=20,
                initialization_noise=None,
            )
            for ep in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task{task_id}"):
                seed = args.seed + ep
                _seed_everywhere(seed)
                try:
                    env.seed(seed)
                except AttributeError:
                    pass
                run_dir = task_root / f"ep{ep}"
                ep_logger = make_episode_logger(run_dir)
                ep_logger.info("task_id=%s bddl=%s vlm_ckpt=%s", task_id, bddl_path, args.base_model_dir)
                planner.reset_episode(instruction="", run_dir=run_dir, logger=ep_logger)
                stage_pct, stage_done, goal_success, diagnostics, replay, replay_wrist = run_episode_async_stateful(
                    task_id=task_id,
                    env=env,
                    client=client,
                    planner=planner,
                    args=args,
                    stage_specs=stage_specs,
                    goal_monitor_dict=goal_monitor_dict,
                    goal_check_override=goal_check_override,
                    vlm_camera_pose=None,
                    logger=ep_logger,
                    fail_on_extra_pour=fail_on_extra_pour,
                    extra_pour_monitor_steps=extra_pour_monitor_steps,
                    post_goal_steps=post_goal_steps,
                )
                stage_sum += stage_pct
                stage_success_cnt += int(diagnostics["stage_success"])
                goal_cnt += stage_pct / 100.0
                base_name = ec.get_video_basename(
                    task_id,
                    ep,
                    seed,
                    diagnostics["stage_success"],
                )
                stages_str = " | ".join(f"{k}={'Y' if v else 'N'}" for k, v in stage_done.items())
                ep_logger.info(
                    "Episode %s seed=%s stage_score=%.1f stage_success=%s goal=%s failure_reason=%s | %s",
                    ep,
                    seed,
                    stage_pct,
                    int(diagnostics["stage_success"]),
                    f"{stage_pct / 100.0:.3f}",
                    diagnostics["failure_reason"],
                    stages_str,
                )
                with prompt_trace_tsv.open("a", encoding="utf-8") as f:
                    goal_text = f"{stage_pct / 100.0:.4f}"
                    f.write(
                        f"{task_id}\t{ep}\t{seed}\t{args.base_model_dir}\t\t"
                        f"{int(diagnostics['stage_success'])}\t{goal_text}\t{stage_pct:.1f}\t"
                        f"{int(diagnostics['extra_pour_detected'])}\t{diagnostics['failure_reason']}\n"
                    )
                if replay:
                    try:
                        _write_video(task_video / f"{base_name}.mp4", replay, fps=10)
                    except Exception:
                        ep_logger.exception("Failed to write main video for %s", base_name)
                if replay_wrist:
                    try:
                        _write_video(task_video / f"{base_name}_wrist.mp4", replay_wrist, fps=10)
                    except Exception:
                        ep_logger.exception("Failed to write wrist video for %s", base_name)
            env.close()
        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        n = max(1, args.num_trials_per_task)
        stage_score = stage_sum / n
        stage_success_rate = stage_success_cnt / n
        goal_success_rate = goal_cnt / n
        dur = round(time.time() - st, 2)
        row = {
            "task_id": task_id,
            "status": status,
            "error": err,
            "stage_score_pct": stage_score,
            "stage_success_rate": stage_success_rate,
            "goal_success_rate": goal_success_rate,
            "video_dir": str(task_video),
            "duration_sec": dur,
        }
        results.append(row)
        summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        with summary_tsv.open("a", encoding="utf-8") as f:
            goal_text = f"{goal_success_rate:.4f}"
            f.write(
                f"{task_id}\t{status}\t{err.replace(chr(9), ' ')}\t{stage_score:.1f}\t"
                f"{stage_success_rate:.4f}\t{goal_text}\t{task_video}\t{dur}\n"
            )
        logging.info(
            "task=%s status=%s stage_score=%.1f stage_success=%.3f goal=%s",
            task_id,
            status,
            stage_score,
            stage_success_rate,
            goal_text,
        )

    planner.close()
    completed = [r for r in results if r["status"] == "completed"]
    aggregate = {
        "macro_stage_score_pct": sum(r["stage_score_pct"] for r in completed) / max(1, len(completed)),
        "macro_stage_success_rate": sum(r["stage_success_rate"] for r in completed) / max(1, len(completed)),
        "macro_goal_success_rate": sum(r["goal_success_rate"] for r in completed) / max(1, len(completed)),
        "num_tasks": len(results),
        "num_goal_scored_tasks": len(completed),
    }
    (out_root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("done aggregate=%s summary=%s", aggregate, summary_tsv)


if __name__ == "__main__":
    main()
