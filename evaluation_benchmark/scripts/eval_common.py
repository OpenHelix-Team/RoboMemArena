from __future__ import annotations

from collections import deque
from pathlib import Path
import json
import logging
import os
import re
import sys
from typing import Any, Callable

import imageio
import numpy as np
import tqdm

PKG_ROOT = Path(__file__).resolve().parent.parent
LIBERO_ROOT = PKG_ROOT / "libero_fork"
DEFAULT_BDDL_BASE = PKG_ROOT / "bddl"

for path in (LIBERO_ROOT,):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from policy_adapter import BasePolicyAdapter, build_eval26_policy_input, ensure_action_chunk, load_policy_adapter


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
_PATCHED_ENV_RESOLUTION: tuple[int, int] | None = None

TASK_PROMPTS = {
    "task1": "Pick and place cookies into the basket, then pick and place tomato sauce into the same basket.",
    "task2": "Pick and place butter into the basket, then pick and place popcorn into the same basket.",
    "task3": "Pick and place cream into the basket, then pick and place chocolate into the same basket.",
    "task4": "Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the drawer that already contains an object.",
    "task5": "Open and close each drawer to check it, in this order: top, middle, bottom. Put butter into the empty drawer.",
    "task6": "Pour tomato sauce over cookies twice and place the sauce bottle into the bowl drainer.",
    "task7": "Pour tomato sauce over the frypan twice and place the sauce bottle into the bowl drainer.",
    "task8": "Pick and place chocolate into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    "task9": "Put butter into the frypan, pour tomato sauce over it twice, then place the sauce bottle into the bowl drainer.",
    "task10": "Pour wine into the mug twice.",
    "task11": "Put cookies into the top drawer, then put butter into the middle drawer.",
    "task12": "Put cookies into the middle drawer and then put chocolate into the same drawer.",
    "task13": "Put cookies into the middle drawer and then put butter into the same drawer.",
    "task14": "Put cookies into the top drawer and put chocolate into another drawer.",
    "task15": "Pick and place butter into the frypan, then pour milk over it twice.",
    "task16": "Pick milk from the table, pour it into the mug twice, then place the milk container into the bowl drainer.",
    "task17": "Put butter into the middle drawer and then put chocolate into the same drawer.",
    "task18": "Pick and place chocolate and butter from cabinet1 to cabinet2, respectively.",
    "task19": "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2, in that order.",
    "task20": "Put cookies into the microwave and then put chocolate into the location where the cookies were placed.",
    "task21": "Put butter into the microwave and then put chocolate into the location where the butter was placed.",
    "task22": "Pour tomato sauce over cookies twice, then put the cookies into the microwave.",
    "task23": "Put cream into the microwave and then put popcorn into the location where the cream was placed.",
    "task24": "Put cookies into the microwave and then put popcorn into the location where the cookies were placed.",
    "task25": "First pick and place butter from plate1 to plate2, then pick and place cream from plate1 to plate2.",
    "task26": "Pick and place chocolate and cream from plate1 to plate2, respectively.",
}


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
    tid, _ = _resolve_task_id(task_id)
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


def _stem_to_prompt(stem: str) -> str:
    parts = stem.split("_", 1)
    text = parts[1] if len(parts) == 2 else stem
    return text.replace("_", " ")


def get_prompt(task_id: str, fallback_task_name: str = "") -> str:
    """Return a readable task prompt derived from the task id or BDDL stem."""
    if task_id in TASK_PROMPTS:
        return TASK_PROMPTS[task_id]
    if fallback_task_name:
        return _stem_to_prompt(fallback_task_name)
    return _stem_to_prompt(task_id)


def _get_env_class():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv


def patch_env_resolution(camera_height: int, camera_width: int) -> None:
    global _PATCHED_ENV_RESOLUTION

    target = (camera_height, camera_width)
    if _PATCHED_ENV_RESOLUTION == target:
        return

    base_env = _get_env_class()
    orig_init = getattr(base_env, "_orig_init_for_resolution_patch", base_env.__init__)

    def patched_init(self, *args, **kwargs):
        kwargs["camera_heights"] = camera_height
        kwargs["camera_widths"] = camera_width
        return orig_init(self, *args, **kwargs)

    base_env._orig_init_for_resolution_patch = orig_init
    base_env.__init__ = patched_init
    globals()["_get_env_class"] = lambda: base_env
    _PATCHED_ENV_RESOLUTION = target


def _resolve_body_id(env: Any, name: str) -> int | None:
    variants = [name]
    if not name.endswith("_main"):
        variants.append(f"{name}_main")
    if name.endswith("_main"):
        variants.append(name[:-5])
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
    z_low: float = -0.20,
    z_high: float = 0.20,
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
                on_site = _is_obj_in_site_region(
                    env,
                    obj,
                    target,
                    x_thresh=0.10,
                    y_thresh=0.10,
                    z_low=-0.05,
                    z_high=0.15,
                )
                if not (on_body or on_site):
                    obj_ok = False
                    break
        if not obj_ok:
            return False
    return True


def get_video_basename(task_id: int | str, ep: int, seed: int, success_metric: bool | int | float) -> str:
    tid, _ = _resolve_task_id(task_id)
    task_tag = f"task{tid}" if tid is not None else str(task_id)
    if isinstance(success_metric, bool):
        succ = success_metric
    else:
        succ = float(success_metric) >= 100.0
    return f"{task_tag}_{'success' if succ else 'failure'}_ep{ep}_seed{seed}"


def _reset_adapter(adapter: BasePolicyAdapter) -> None:
    adapter.reset()


def _close_adapter(adapter: BasePolicyAdapter) -> None:
    close_fn = getattr(adapter, "close", None)
    if callable(close_fn):
        close_fn()


def run_episode_with_stages(
    env: Any,
    adapter: BasePolicyAdapter,
    prompt: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    post_goal_steps: int,
    stage_checks: list[tuple[str, Callable[[Any], bool]]],
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
    stage_checks_sequential: bool,
) -> tuple[float, dict[str, bool], bool, list[np.ndarray], list[np.ndarray]]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    stage_done = {name: False for name, _ in stage_checks}
    t = 0
    all_stages_logged = False
    goal_reached_t: int | None = None

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

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

            if stage_checks_sequential:
                for i, (name, check_fn) in enumerate(stage_checks):
                    if stage_done[name]:
                        continue
                    prev_all_done = all(stage_done[n] for n, _ in stage_checks[:i])
                    if prev_all_done and check_fn(env):
                        stage_done[name] = True
                        logging.info(f"  [t={t}] Stage completed: {name}")
            else:
                for name, check_fn in stage_checks:
                    if not stage_done[name] and check_fn(env):
                        stage_done[name] = True
                        logging.info(f"  [t={t}] Stage completed: {name}")

            if all(stage_done.values()) and not all_stages_logged:
                logging.info(f"  [t={t}] All stages completed.")
                all_stages_logged = True

            goal_success = bool(stage_done) and all(stage_done.values())
            if goal_success and goal_reached_t is None:
                goal_reached_t = t
                logging.info(
                    f"  [t={t}] Required stages completed. Continuing {post_goal_steps} more steps before exit."
                )

            if done:
                break
            if goal_reached_t is not None and (t - goal_reached_t) >= post_goal_steps:
                break
            t += 1
    except Exception as exc:
        logging.exception(f"Episode failed: {exc}")

    num_done = sum(1 for name, _ in stage_checks if stage_done[name])
    score = 100.0 * num_done / len(stage_checks)
    goal_success = bool(stage_done) and all(stage_done.values())
    return score, stage_done, goal_success, replay, replay_wrist


def run_episode_simple(
    env: Any,
    adapter: BasePolicyAdapter,
    prompt: str,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    post_goal_steps: int,
    goal_monitor_dict: dict[str, list[tuple[str, str]]],
) -> tuple[bool, bool, list[np.ndarray], list[np.ndarray]]:
    obs = env.reset()
    replay: list[np.ndarray] = []
    replay_wrist: list[np.ndarray] = []
    action_plan: deque[np.ndarray] = deque()
    t = 0
    done_success = False
    goal_reached_t: int | None = None

    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

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
            goal_success = check_goal_success(env, goal_monitor_dict) if goal_monitor_dict else False
            if goal_success and goal_reached_t is None:
                goal_reached_t = t
                logging.info(f"  [t={t}] Goal reached. Continuing {post_goal_steps} more steps before exit.")
            if done:
                done_success = True
                break
            if goal_reached_t is not None and (t - goal_reached_t) >= post_goal_steps:
                break
            t += 1
    except Exception as exc:
        logging.exception(f"Episode failed: {exc}")

    goal_success = goal_reached_t is not None
    return done_success, goal_success, replay, replay_wrist


def run_eval(
    task_id: int | str,
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
    stage_checks: list[tuple[str, Callable[[Any], bool]]] | None = None,
    stage_checks_sequential: bool = True,
    seed_everywhere_fn: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    if adapter is None:
        adapter = load_policy_adapter(adapter_spec or "", **(adapter_kwargs or {}))
        owns_adapter = True
    else:
        owns_adapter = False

    tid, task_key = _resolve_task_id(task_id)
    bddl_path = _resolve_bddl_path(task_id)
    prompt = get_prompt(task_key, bddl_path.stem)
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

    stage_checks_list = list(stage_checks or [])
    use_stage_check = len(stage_checks_list) > 0
    goal_monitor_dict = _build_goal_monitor_dict(bddl_path)

    total_score = 0.0
    stage_totals = {name: 0 for name, _ in stage_checks_list} if use_stage_check else {}
    goal_succ_cnt = 0
    env_done_cnt = 0
    tsr_succ_cnt = 0
    episodes: list[dict[str, Any]] = []

    try:
        for ep in tqdm.tqdm(range(num_trials_per_task), desc=f"task{task_id}"):
            current_seed = seed + ep
            if seed_everywhere_fn:
                seed_everywhere_fn(current_seed)
            try:
                env.seed(current_seed)
            except AttributeError:
                pass
            _reset_adapter(adapter)

            if use_stage_check:
                score, stage_done, goal_success, replay, replay_wrist = run_episode_with_stages(
                    env=env,
                    adapter=adapter,
                    prompt=prompt,
                    resize_size=resize_size,
                    replan_steps=replan_steps,
                    num_steps_wait=num_steps_wait,
                    max_steps=max_steps,
                    post_goal_steps=post_goal_steps,
                    stage_checks=stage_checks_list,
                    goal_monitor_dict=goal_monitor_dict,
                    stage_checks_sequential=stage_checks_sequential,
                )
                total_score += score
                for name in stage_done:
                    stage_totals[name] += int(stage_done[name])
                tsr_success = bool(stage_done) and all(stage_done.values())
                tsr_succ_cnt += int(tsr_success)
                goal_succ_cnt += score / 100.0
                base_name = get_video_basename(task_id, ep, current_seed, tsr_success)
                ep_summary = {
                    "ep": ep,
                    "seed": current_seed,
                    "TSR": 100.0 if tsr_success else 0.0,
                    "CSR": float(score),
                    "stage_done": stage_done,
                }
            else:
                env_done_success, goal_success, replay, replay_wrist = run_episode_simple(
                    env=env,
                    adapter=adapter,
                    prompt=prompt,
                    resize_size=resize_size,
                    replan_steps=replan_steps,
                    num_steps_wait=num_steps_wait,
                    max_steps=max_steps,
                    post_goal_steps=post_goal_steps,
                    goal_monitor_dict=goal_monitor_dict,
                )
                env_done_cnt += int(env_done_success)
                tsr_success = bool(env_done_success)
                tsr_succ_cnt += int(tsr_success)
                goal_succ_cnt += int(goal_success)
                base_name = get_video_basename(
                    task_id,
                    ep,
                    current_seed,
                    goal_success if goal_monitor_dict else env_done_success,
                )
                ep_summary = {
                    "ep": ep,
                    "seed": current_seed,
                    "TSR": 100.0 if tsr_success else 0.0,
                    "CSR": 100.0 if goal_success else 0.0,
                    "env_done_success": bool(env_done_success),
                }

            if replay:
                imageio.mimwrite(video_dir / f"{base_name}.mp4", replay, fps=10)
            if replay_wrist:
                imageio.mimwrite(video_dir / f"{base_name}_wrist.mp4", replay_wrist, fps=10)

            episodes.append(ep_summary)

            if use_stage_check:
                stages_str = " | ".join(f"{n}={'Y' if ep_summary['stage_done'][n] else 'N'}" for n in ep_summary["stage_done"])
                logging.info(
                    f"Episode {ep} (seed={current_seed}): TSR={ep_summary['TSR']:.0f}% | "
                    f"CSR={ep_summary['CSR']:.0f}% | {stages_str}"
                )
            else:
                logging.info(
                    f"Episode {ep} (seed={current_seed}): TSR={ep_summary['TSR']:.0f}% | CSR={ep_summary['CSR']:.0f}%"
                )
    finally:
        env.close()
        if owns_adapter:
            _close_adapter(adapter)

    n = num_trials_per_task
    logging.info("============================================================")
    if use_stage_check:
        avg_score = total_score / max(1, n)
        logging.info(f"Final result - average stage success rate = {avg_score:.1f}%")
        for name, cnt in stage_totals.items():
            logging.info(f"  {name}: {cnt}/{n} ({(cnt / max(1, n)) * 100:.0f}%)")
    else:
        avg_score = 100.0 * env_done_cnt / max(1, n)
        logging.info(f"Final result - env done success rate: {env_done_cnt}/{n} ({avg_score:.1f}%)")
    tsr_pct = 100.0 * tsr_succ_cnt / max(1, n)
    logging.info(f"Final result - TSR all-stage success rate: {tsr_succ_cnt}/{n} ({tsr_pct:.1f}%)")
    goal_pct = 100.0 * goal_succ_cnt / max(1, n) if use_stage_check else (100.0 * goal_succ_cnt / max(1, n) if goal_monitor_dict else 0.0)
    if use_stage_check:
        logging.info(f"Final result - CSR stage completion rate: {goal_pct:.1f}%")
    elif goal_monitor_dict:
        logging.info(f"Final result - BDDL goal success rate: {goal_succ_cnt}/{n} ({goal_pct:.1f}%)")
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


def parse_adapter_kwargs(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("adapter kwargs must be a JSON object")
    return data
