from __future__ import annotations

from collections import deque
from pathlib import Path
import logging
import os
import random
import re
from typing import Any, Callable

import imageio
import numpy as np
import tqdm

from openpi_client import websocket_client_policy as _websocket_client_policy
from robocerebra_adapter import create_history, obs_to_pi_element, obs_to_pi_mem_element
from task_prompts import get_prompt


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_BDDL_BASE = Path(__file__).resolve().parents[2] / "bddl"
TASK2_4PROMPTMIX_PROMPTS = (
    "pick butter",
    "place butter into basket",
    "pick popcorn",
    "place popcorn into basket",
)
PROMPT_POOLS = {
    "task2_4promptmix": TASK2_4PROMPTMIX_PROMPTS,
}


def build_policy_input_builder(
    *,
    resize_size: int,
    prompt: str,
    mem_policy: bool = False,
    mem_obs_steps: int = 4,
) -> Callable[[dict[str, Any]], tuple[dict[str, Any], np.ndarray, np.ndarray | None]]:
    if not mem_policy:
        def _single_frame_builder(obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
            element = obs_to_pi_element(obs, resize_size=resize_size, prompt=prompt)
            return element, element["observation/image"], element.get("observation/wrist_image")

        return _single_frame_builder

    history = create_history(mem_obs_steps)

    def _mem_builder(obs: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
        element, rgb, wrist_rgb = obs_to_pi_mem_element(
            obs=obs,
            history=history,
            resize_size=resize_size,
            mem_obs_steps=mem_obs_steps,
            prompt=prompt,
        )
        return element, rgb, wrist_rgb

    return _mem_builder


def _resolve_task_id(task_id: int | str) -> tuple[int | None, str]:
    if isinstance(task_id, int):
        return task_id, f"task{task_id}"
    s = str(task_id).strip()
    if s.isdigit():
        tid = int(s)
        return tid, f"task{tid}"
    m = re.fullmatch(r"task(\d+)", s)
    if m:
        tid = int(m.group(1))
        return tid, f"task{tid}"
    return None, s


def _resolve_bddl_path(task_id: int | str) -> Path:
    tid, key = _resolve_task_id(task_id)
    candidates: list[Path] = []

    if tid is not None:
        candidates.extend(sorted(DEFAULT_BDDL_BASE.glob(f"{tid}_*.bddl")))
        candidates.append(DEFAULT_BDDL_BASE / f"task{tid}.bddl")

    if isinstance(task_id, str):
        p = Path(task_id)
        if p.suffix == ".bddl":
            candidates.insert(0, p)
            candidates.insert(1, DEFAULT_BDDL_BASE / p.name)

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"Cannot resolve BDDL for task_id={task_id}. Checked: {candidates}")


def get_prompt_pool(pool_name: str) -> tuple[str, ...]:
    try:
        return PROMPT_POOLS[pool_name]
    except KeyError as exc:
        valid = ", ".join(sorted(PROMPT_POOLS))
        raise ValueError(f"Unknown prompt pool: {pool_name}. Valid pools: {valid}") from exc


def resolve_prompt(
    task_id: int | str,
    fallback_task_name: str = "",
    *,
    prompt_mode: str = "fixed",
    prompt_pool_name: str | None = None,
    rng_seed: int | None = None,
) -> str:
    if prompt_mode == "fixed":
        _, task_key = _resolve_task_id(task_id)
        return get_prompt(task_key, fallback_task_name)

    if prompt_mode == "task2_4prompt_random_episode":
        pool = get_prompt_pool(prompt_pool_name or "task2_4promptmix")
        if rng_seed is None:
            raise ValueError("rng_seed is required when prompt_mode=task2_4prompt_random_episode")
        return random.Random(rng_seed).choice(pool)

    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def _get_env_class():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def _resolve_body_id(env: Any, name: str) -> int | None:
    variants = [name]
    if not name.endswith("_main"):
        variants.append(f"{name}_main")
    if name.endswith("_main"):
        variants.append(name[:-5])
    # common fallback for object names with trailing index, e.g. basket_1
    if "_main" not in name and name.count("_") >= 1:
        variants.append(name.replace("_", "_", 1) + "_main")
    for v in variants:
        try:
            return env.sim.model.body_name2id(v)
        except Exception:
            continue
    return None


def _body_pos(env: Any, name: str) -> np.ndarray | None:
    bid = _resolve_body_id(env, name)
    if bid is None:
        return None
    return np.asarray(env.sim.data.body_xpos[bid], dtype=np.float32)


def _resolve_site_id(env: Any, name: str) -> int | None:
    variants = [name]
    if not name.endswith("_main"):
        variants.append(f"{name}_main")
    if name.endswith("_main"):
        variants.append(name[:-5])
    for v in variants:
        try:
            return env.sim.model.site_name2id(v)
        except Exception:
            continue
    return None


def _site_pos(env: Any, name: str) -> np.ndarray | None:
    sid = _resolve_site_id(env, name)
    if sid is None:
        return None
    return np.asarray(env.sim.data.site_xpos[sid], dtype=np.float32)


def _is_obj_in_container(
    env: Any,
    obj_name: str,
    container_name: str,
    xy_thresh: float = 0.12,
    z_low: float = -0.15,
    z_high: float = 0.20,
) -> bool:
    obj_pos = _body_pos(env, obj_name)
    ctr_pos = _body_pos(env, container_name)
    if obj_pos is None or ctr_pos is None:
        return False
    xy_dist = float(np.linalg.norm(obj_pos[:2] - ctr_pos[:2]))
    z_delta = float(obj_pos[2] - ctr_pos[2])
    return xy_dist <= xy_thresh and z_low <= z_delta <= z_high


def _is_obj_in_site_region(
    env: Any,
    obj_name: str,
    site_name: str,
    x_thresh: float = 0.20,
    y_thresh: float = 0.20,
    z_low: float = -1.0,
    z_high: float = 1.0,
) -> bool:
    obj_pos = _body_pos(env, obj_name)
    site_pos = _site_pos(env, site_name)
    if obj_pos is None or site_pos is None:
        return False
    x_diff = abs(float(obj_pos[0] - site_pos[0]))
    y_diff = abs(float(obj_pos[1] - site_pos[1]))
    z_delta = float(obj_pos[2] - site_pos[2])
    return x_diff <= x_thresh and y_diff <= y_thresh and z_low <= z_delta <= z_high


def make_obj_in_basket_check(obj_body_name: str) -> Callable[[Any], bool]:
    def check(env: Any) -> bool:
        for basket_name in ("basket_1", "basket", "basket_1_main", "basket_main"):
            if _resolve_body_id(env, basket_name) is not None:
                return _is_obj_in_container(env, obj_body_name, basket_name)
        return False

    return check


def _flatten_goal_states(goal_state: Any) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []

    if goal_state is None:
        return out
    if isinstance(goal_state, (list, tuple)):
        if len(goal_state) >= 3 and isinstance(goal_state[0], str):
            rel = goal_state[0]
            if rel in {"In", "On"}:
                out.append((rel, str(goal_state[1]), str(goal_state[2])))
                return out
        for x in goal_state:
            out.extend(_flatten_goal_states(x))
        return out
    if isinstance(goal_state, str):
        for rel, obj, reg in re.findall(r"\((In|On)\s+([A-Za-z0-9_]+)\s+([A-Za-z0-9_]+)\)", goal_state):
            out.append((rel, obj, reg))
    return out


def goal_state_to_monitor_dict(goal_state: Any) -> dict[str, list[tuple[str, str]]]:
    # Return: {obj_name: [(relation, target_container_name), ...]}
    monitor: dict[str, list[tuple[str, str]]] = {}
    flat = _flatten_goal_states(goal_state)
    for rel, obj, region_or_target in flat:
        target = region_or_target
        if region_or_target.endswith("_contain_region"):
            target = region_or_target[: -len("_contain_region")]
        monitor.setdefault(obj, []).append((rel, target))
    return monitor


def _parse_goal_state_from_bddl(bddl_path: Path) -> str:
    text = bddl_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\(:goal\s*(.*?)\)\s*\)\s*$", text, flags=re.S)
    if not m:
        return ""
    return m.group(1)


def _build_goal_monitor_dict(bddl_path: Path) -> dict[str, list[tuple[str, str]]]:
    goal_str = _parse_goal_state_from_bddl(bddl_path)
    return goal_state_to_monitor_dict(goal_str)


def check_goal_success(env: Any, monitor_dict: dict[str, list[tuple[str, str]]]) -> bool:
    if not monitor_dict:
        return False
    for obj, conditions in monitor_dict.items():
        obj_ok = True
        for rel, target in conditions:
            if rel == "In":
                in_body = _is_obj_in_container(env, obj, target)
                in_site = _is_obj_in_site_region(env, obj, target)
                if not (in_body or in_site):
                    obj_ok = False
                    break
            elif rel == "On":
                on_body = _is_obj_in_container(env, obj, target, xy_thresh=0.10, z_low=-0.05, z_high=0.15)
                on_site = _is_obj_in_site_region(env, obj, target, x_thresh=0.10, y_thresh=0.10, z_low=-0.05, z_high=0.15)
                if not (on_body or on_site):
                    obj_ok = False
                    break
        if not obj_ok:
            return False
    return True


def run_episode_with_stages(
    env: Any,
    client: _websocket_client_policy.WebsocketClientPolicy,
    prompt: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    stage_checks: list[tuple[str, Callable[[Any], bool]]],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    stage_checks_sequential: bool,
    mem_policy: bool = False,
    mem_obs_steps: int = 4,
) -> tuple[float, dict[str, bool], bool, list[np.ndarray], list[np.ndarray]]:
    """ episode， (score, stage_done, goal_success, replay, replay_wrist)"""
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    stage_done = {name: False for name, _ in stage_checks}
    t = 0
    build_element = build_policy_input_builder(
        resize_size=resize_size,
        prompt=prompt,
        mem_policy=mem_policy,
        mem_obs_steps=mem_obs_steps,
    )

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            element, rgb, wrist_rgb = build_element(obs)
            replay.append(rgb)
            if wrist_rgb is not None:
                replay_wrist.append(wrist_rgb)

            if not action_plan:
                out = client.infer(element)
                actions = np.asarray(out["actions"])
                action_plan.extend(actions[:replan_steps])

            action = action_plan.popleft()
            obs, _, done, info = env.step(action.tolist())

            if stage_checks_sequential:
                for i, (name, check_fn) in enumerate(stage_checks):
                    if stage_done[name]:
                        continue
                    prev_all_done = all(stage_done[n] for n, _ in stage_checks[:i])
                    if prev_all_done and check_fn(env):
                        stage_done[name] = True
                        logging.info(f"  [t={t}] : {name}")
            else:
                for name, check_fn in stage_checks:
                    if not stage_done[name] and check_fn(env):
                        stage_done[name] = True
                        logging.info(f"  [t={t}] : {name}")

            if all(stage_done.values()):
                logging.info(f"  [t={t}] !")
                break

            if done:
                break
            t += 1
    except Exception as e:
        logging.exception(f"Episode failed: {e}")

    num_done = sum(1 for name, _ in stage_checks if stage_done[name])
    score = 100.0 * num_done / len(stage_checks)
    goal_success = check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False
    return score, stage_done, goal_success, replay, replay_wrist


def run_episode_simple(
    env: Any,
    client: _websocket_client_policy.WebsocketClientPolicy,
    prompt: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    mem_policy: bool = False,
    mem_obs_steps: int = 4,
) -> tuple[bool, bool, list[np.ndarray], list[np.ndarray]]:
    """ env done ， (env_done_success, goal_success, replay, replay_wrist)"""
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    t = 0
    done_success = False
    build_element = build_policy_input_builder(
        resize_size=resize_size,
        prompt=prompt,
        mem_policy=mem_policy,
        mem_obs_steps=mem_obs_steps,
    )

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            element, rgb, wrist_rgb = build_element(obs)
            replay.append(rgb)
            if wrist_rgb is not None:
                replay_wrist.append(wrist_rgb)

            if not action_plan:
                out = client.infer(element)
                actions = np.asarray(out["actions"])
                action_plan.extend(actions[:replan_steps])

            action = action_plan.popleft()
            obs, _, done, info = env.step(action.tolist())
            if done:
                done_success = True
                break
            t += 1
    except Exception as e:
        logging.exception(f"Episode failed: {e}")

    goal_success = check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False
    return done_success, goal_success, replay, replay_wrist


def get_video_basename(task_id: int | str, ep: int, seed: int, outcome: int | float | bool) -> str:
    tid, key = _resolve_task_id(task_id)
    task_tag = f"task{tid}" if tid is not None else str(task_id)
    # Backward compatible:
    # - score mode: success when score >= 100
    # - goal mode: success when outcome is True
    if isinstance(outcome, bool):
        succ = outcome
    else:
        try:
            succ = float(outcome) >= 100.0
        except Exception:
            succ = bool(outcome)
    return f"{task_tag}_{'success' if succ else 'failure'}_ep{ep}"


def run_eval(
    task_id: int | str,
    num_trials_per_task: int,
    host: str,
    port: int,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    video_out_path: str,
    seed: int,
    stage_checks: list[tuple[str, Callable[[Any], bool]]] | None = None,
    stage_checks_sequential: bool = True,
    seed_everywhere_fn: Callable[[int], None] | None = None,
    mem_policy: bool = False,
    mem_obs_steps: int = 4,
    prompt_override: str | None = None,
) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    tid, task_key = _resolve_task_id(task_id)
    bddl_path = _resolve_bddl_path(task_id)
    prompt = prompt_override if prompt_override is not None else resolve_prompt(task_key, bddl_path.stem)
    video_dir = Path(video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Using BDDL: {bddl_path}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Video output: {video_dir}")

    OffScreenRenderEnv = _get_env_class()
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=256,
        camera_widths=256,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        initialization_noise=None,
    )
    client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    stage_checks_list = list(stage_checks or [])
    use_stage_check = len(stage_checks_list) > 0
    goal_monitor_dict = _build_goal_monitor_dict(bddl_path)

    total_score = 0.0
    stage_totals = {name: 0 for name, _ in stage_checks_list} if use_stage_check else {}
    goal_succ_cnt = 0
    env_done_cnt = 0

    for ep in tqdm.tqdm(range(num_trials_per_task), desc=f"task{task_id}"):
        current_seed = seed + ep
        if seed_everywhere_fn:
            seed_everywhere_fn(current_seed)
        try:
            env.seed(current_seed)
        except AttributeError:
            pass

        if use_stage_check:
            score, stage_done, goal_success, replay, replay_wrist = run_episode_with_stages(
                env=env,
                client=client,
                prompt=prompt,
                resize_size=resize_size,
                replan_steps=replan_steps,
                num_steps_wait=num_steps_wait,
                max_steps=max_steps,
                stage_checks=stage_checks_list,
                goal_monitor_dict=goal_monitor_dict,
                stage_checks_sequential=stage_checks_sequential,
                mem_policy=mem_policy,
                mem_obs_steps=mem_obs_steps,
            )
            total_score += score
            for name in stage_done:
                stage_totals[name] += int(stage_done[name])
            goal_succ_cnt += int(goal_success)
            base_name = get_video_basename(task_id, ep, current_seed, int(score))
        else:
            env_done_success, goal_success, replay, replay_wrist = run_episode_simple(
                env=env,
                client=client,
                prompt=prompt,
                resize_size=resize_size,
                replan_steps=replan_steps,
                num_steps_wait=num_steps_wait,
                max_steps=max_steps,
                goal_monitor_dict=goal_monitor_dict,
                mem_policy=mem_policy,
                mem_obs_steps=mem_obs_steps,
            )
            env_done_cnt += int(env_done_success)
            goal_succ_cnt += int(goal_success)
            base_name = get_video_basename(task_id, ep, current_seed, 100 if env_done_success else 0)

        if replay:
            imageio.mimwrite(video_dir / f"{base_name}.mp4", replay, fps=10)
        if replay_wrist:
            imageio.mimwrite(video_dir / f"{base_name}_wrist.mp4", replay_wrist, fps=10)

        if use_stage_check:
            stages_str = " | ".join(f"{n}={'Y' if stage_done[n] else 'N'}" for n in stage_done)
            logging.info(
                f"Episode {ep} (seed={current_seed}): score={score:.0f}% | {stages_str} | goal={'Y' if goal_success else 'N'}"
            )
        else:
            logging.info(
                f"Episode {ep} (seed={current_seed}): env_done={'Y' if env_done_success else 'N'} | goal={'Y' if goal_success else 'N'}"
            )

    env.close()

    n = num_trials_per_task
    logging.info("============================================================")
    if use_stage_check:
        avg_score = total_score / max(1, n)
        logging.info(f" - :  = {avg_score:.1f}%")
        for name, cnt in stage_totals.items():
            logging.info(f"  {name}: {cnt}/{n} ({(cnt / max(1, n)) * 100:.0f}%)")
    else:
        env_pct = 100.0 * env_done_cnt / max(1, n)
        logging.info(f" - env done : {env_done_cnt}/{n} ({env_pct:.1f}%)")
    if goal_monitor_dict:
        goal_pct = 100.0 * goal_succ_cnt / max(1, n)
        logging.info(f" - BDDL goal : {goal_succ_cnt}/{n} ({goal_pct:.1f}%)")
    logging.info(f": {video_dir}")
    logging.info("============================================================")
