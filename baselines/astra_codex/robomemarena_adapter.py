"""Astra waypoint policy bridged to the unchanged RoboMemArena step loop."""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from policy_adapter import BasePolicyAdapter


SYSTEM = """You control a simulated Franka Panda by choosing end-effector waypoints.
Answer with ONE JSON object:
{"description":"what happened and what you will do next",
 "waypoint":[x,y,z,roll,pitch,yaw],"gripper":-1}
World coordinates are in metres, +z is up. The robot base pose is reported below;
use that pose and the camera calibration rather than assumptions from another environment.
The TCP is the Panda controller's grip site between the fingertips. Its orientation
uses extrinsic XYZ roll, pitch, yaw in radians: R = Rz(yaw) Ry(pitch) Rx(roll).
The reported TCP pose and commanded pose use exactly this same frame and convention.
Gripper: -1 = close, +1 = open, 0 = keep. The arm first moves toward your waypoint
using feedback OSC control, then applies a nonzero gripper command for 25 control
steps. With 0 the existing gripper target is unchanged. Identical waypoints are
allowed, including gripper-only actions. There is no collision-avoiding path planner;
choose intermediate waypoints for clearance. A motion is limited to 80 control steps;
after a timeout you receive the actual pose and can correct it.
Simulation runs at 20 control steps per second and advances only while actions execute.
Each observation message contains front-camera frames since the preceding request,
one every 3 control steps plus the last, followed by the current wrist image.
Images are upright RGB; calibration is for the displayed image dimensions.
K is pinhole intrinsics and [R|t] maps WORLD to CAMERA coordinates in the OpenCV
convention (x right, y down, z forward): pixel ~ K (R X_world + t).
You receive only images, the robot's own state, and camera calibration, without depth
or object positions. Remember previous observations when the task requires it.
The official evaluator owns episode completion and the control-step budget.
If you believe the task is complete, answer with "waypoint":null; this stops further
model decisions and returns zero arm motion with the gripper target unchanged.
The official evaluator continues stepping and observing until its own termination.
"""


def parse_waypoint(text: str) -> tuple[np.ndarray | None, int]:
    """Accept the upstream JSON / fenced JSON form, rejecting non-finite poses."""
    if "```" in text:
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        answer = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, TypeError) as exc:
        raise ValueError("unparseable_model_answer") from exc
    if not isinstance(answer, dict):
        raise ValueError("invalid_model_answer")
    if "waypoint" in answer and answer["waypoint"] is None:
        return None, 0
    waypoint, gripper = answer.get("waypoint"), answer.get("gripper", 0)
    if (not isinstance(waypoint, list) or len(waypoint) != 6
            or any(type(v) not in (int, float) for v in waypoint)
            or type(gripper) not in (int, float) or gripper not in (-1, 0, 1)):
        raise ValueError("invalid_waypoint")
    pose = np.asarray(waypoint, dtype=np.float64)
    if not np.isfinite(pose).all():
        raise ValueError("nonfinite_waypoint")
    return pose, int(gripper)


def pose_error(position, orientation, waypoint) -> tuple[np.ndarray, np.ndarray]:
    target_rotation = Rotation.from_euler("xyz", np.asarray(waypoint)[3:]).as_matrix()
    # robosuite set_goal_orientation left-multiplies its current world rotation.
    rotation_error = Rotation.from_matrix(target_rotation @ np.asarray(orientation).T).as_rotvec()
    return np.asarray(waypoint)[:3] - np.asarray(position), rotation_error


def osc_action(position, orientation, waypoint, gripper, input_min, input_max, output_min, output_max):
    """Invert the live OSC controller scaling, preserving its clipping limits."""
    delta = np.concatenate(pose_error(position, orientation, waypoint))
    lo, hi, out_lo, out_hi = [np.broadcast_to(np.asarray(x, dtype=float), (6,))
                              for x in (input_min, input_max, output_min, output_max)]
    if not all(np.isfinite(x).all() for x in (delta, lo, hi, out_lo, out_hi)):
        raise ValueError("nonfinite_controller_state")
    if np.any(hi <= lo) or np.any(out_hi <= out_lo):
        raise ValueError("invalid_controller_scale")
    scaled = (delta - (out_hi + out_lo) / 2) * (hi - lo) / (out_hi - out_lo) + (hi + lo) / 2
    return np.concatenate([np.clip(scaled, lo, hi), [-int(gripper)]]).astype(np.float32)


def camera_calibration(sim, name: str, raw_shape, image_shape) -> dict:
    """OpenCV calibration after official flipud and possibly anisotropic resize."""
    camera_id = sim.model.camera_name2id(name)
    raw_h, raw_w = raw_shape[:2]
    height, width = image_shape[:2]
    focal = 0.5 * raw_h / np.tan(np.deg2rad(sim.model.cam_fovy[camera_id]) / 2)
    intrinsic = np.array([[focal, 0, raw_w / 2], [0, focal, raw_h / 2], [0, 0, 1]])
    intrinsic = np.diag([width / raw_w, height / raw_h, 1]) @ intrinsic
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = sim.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    camera_to_world[:3, 3] = sim.data.cam_xpos[camera_id]
    return {"K": intrinsic.tolist(), "world_to_camera": np.linalg.inv(camera_to_world)[:3].tolist(),
            "image_size_wh": [width, height]}


class RoboMemArenaAdapter(BasePolicyAdapter):
    def __init__(self, run_dir: str, max_model_calls: int = 150, timeout: float = 600):
        self.run_dir = Path(run_dir).resolve()
        self.max_model_calls, self.timeout = int(max_model_calls), float(timeout)
        if self.max_model_calls < 1 or self.timeout <= 0:
            raise ValueError("max_model_calls and timeout must be positive")
        for name in ("requests", "responses", "observations"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        self.env = None
        self.request_id, self.episode = 0, -1
        self.reset()
        self.episode = -1  # The evaluator's first reset starts episode zero.

    def reset(self):
        self.episode += 1
        self.steps = self.model_calls = self.motion_steps = self.gripper_steps = 0
        self.target = None
        self.target_gripper = 0
        self.phase, self.previous_outcome = "motion", "initial_observation"
        self.frames = []
        self.latest_obs = None
        self.failure_reason = None
        self.finished = False

    def bind_env(self, env):
        from robosuite import macros

        if macros.IMAGE_CONVENTION != "opengl":
            raise RuntimeError("expected_opengl_raw_images_for_official_flipud")
        self.env = env

    def _trace(self, event, **data):
        row = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "episode": self.episode, "policy_steps": self.steps, "model_calls": self.model_calls,
               "event": event, **data}
        with (self.run_dir / "trace.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")

    def observe(self, obs, prompt, resize_size):
        self.latest_obs = obs
        if not self.finished:
            self.frames.append((self.steps, np.asarray(obs["observation/image"]).copy()))

    def _tcp(self):
        if self.env is None:
            raise RuntimeError("bind_env must be called before inference")
        robot = self.env.robots[0]
        site = robot.eef_site_id
        return (np.asarray(self.env.sim.data.site_xpos[site]).copy(),
                np.asarray(self.env.sim.data.site_xmat[site]).reshape(3, 3).copy())

    def _state(self, obs):
        position, orientation = self._tcp()
        robot, sim = self.env.robots[0], self.env.sim
        root_id = sim.model.body_name2id(robot.robot_model.root_body)
        raw = obs["_raw_obs"]
        return {
            "tcp_pose_xyz_rpy": np.concatenate([position, Rotation.from_matrix(orientation).as_euler("xyz")]).tolist(),
            "finger_joint_positions_m": np.asarray(raw["robot0_gripper_qpos"]).tolist(),
            "robot_base_position_world_m": np.asarray(sim.data.body_xpos[root_id]).tolist(),
            "robot_base_rotation_world": np.asarray(sim.data.body_xmat[root_id]).reshape(3, 3).tolist(),
            "elapsed_control_steps": int(self.env.env.timestep),
            "previous_waypoint_outcome": self.previous_outcome,
            "front_camera": camera_calibration(sim, "agentview", raw["agentview_image"].shape,
                                                obs["observation/image"].shape),
            "wrist_camera": camera_calibration(sim, "robot0_eye_in_hand", raw["robot0_eye_in_hand_image"].shape,
                                                obs["observation/wrist_image"].shape),
        }

    def _ask(self, text, images):
        self.request_id += 1
        request_path = self.run_dir / "requests" / f"{self.request_id:06d}.json"
        response_path = self.run_dir / "responses" / request_path.name
        if request_path.exists() or response_path.exists():
            raise RuntimeError(f"broker_id_already_exists:{self.request_id}")
        payload = {"text": text, "images": images, "episode": self.episode}
        temporary = request_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(request_path)
        deadline = time.monotonic() + self.timeout
        while not response_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"broker_timeout:{self.request_id}")
            time.sleep(0.1)
        response = json.loads(response_path.read_text())
        self._trace("model_response", request_id=self.request_id, response=response)
        if response.get("error"):
            raise RuntimeError(f"broker_error:{response['error']}")
        if not isinstance(response.get("text"), str):
            raise ValueError("broker_response_missing_text")
        return response["text"]

    def _new_waypoint(self, obs, prompt):
        if self.model_calls >= self.max_model_calls:
            raise RuntimeError("model_call_limit")
        self.model_calls += 1
        if not self.frames:
            self.frames = [(self.steps, np.asarray(obs["observation/image"]).copy())]
        indices = sorted(set(range(0, len(self.frames), 3)) | {len(self.frames) - 1})
        selected = [self.frames[i] for i in indices]
        images = []
        for step, frame in selected:
            path = self.run_dir / "observations" / f"ep{self.episode:03d}_front_{step:06d}.png"
            Image.fromarray(frame).save(path)
            images.append(str(path))
        wrist_path = self.run_dir / "observations" / f"ep{self.episode:03d}_wrist_{self.steps:06d}.png"
        Image.fromarray(np.asarray(obs["observation/wrist_image"])).save(wrist_path)
        images.append(str(wrist_path))
        text = (SYSTEM + "\nTask goal: " + prompt + "\nFront frames at policy steps: "
                + json.dumps([step for step, _ in selected]) + "; final attached image is the current wrist.\n"
                + "Robot state and calibration: " + json.dumps(self._state(obs), allow_nan=False))
        self._trace("model_request", request_id=self.request_id + 1, images=images)
        reply = self._ask(text, images)
        self.target, self.target_gripper = parse_waypoint(reply)
        self.motion_steps = self.gripper_steps = 0
        self.phase = "motion"
        self.frames.clear()
        if self.target is None:
            self.finished = True
            self._trace("policy_stop", reason="declared_complete")
            return
        self._trace("waypoint", target=self.target.tolist(), gripper=self.target_gripper)

    def infer_actions(self, obs, prompt, resize_size):
        try:
            if self.latest_obs is not obs:
                self.observe(obs, prompt, resize_size)
            if self.target is not None:
                finished = self.phase == "gripper" and self.gripper_steps >= 25
                timed_out = self.phase == "motion" and self.motion_steps >= 80
                if finished or timed_out:
                    self.previous_outcome = "reached_and_gripper_applied" if finished else "motion_timeout"
                    self._trace("waypoint_end", outcome=self.previous_outcome, target=self.target.tolist())
                    self.target = None
            if self.target is None and not self.finished:
                self._new_waypoint(obs, prompt)
            if self.finished:
                action = np.zeros((1, 7), dtype=np.float32)
                self._trace("hold", action=action[0].tolist(), reason="declared_complete")
                self.steps += 1
                return action
            position, orientation = self._tcp()
            dp, dr = pose_error(position, orientation, self.target)
            if self.phase == "motion" and np.linalg.norm(dp) <= 0.005 and np.linalg.norm(dr) <= 0.05:
                self.phase = "gripper"
                self._trace("waypoint_reached", position_error_m=float(np.linalg.norm(dp)),
                            orientation_error_rad=float(np.linalg.norm(dr)))
            gripper = self.target_gripper if self.phase == "gripper" else 0
            controller = self.env.robots[0].controller
            if not controller.use_delta or not controller.use_ori or controller.impedance_mode != "fixed":
                raise RuntimeError("expected_fixed_delta_OSC_POSE_controller")
            action = osc_action(position, orientation, self.target, gripper,
                                controller.input_min, controller.input_max, controller.output_min, controller.output_max)
            self._trace("action", action=action.tolist(), target=self.target.tolist(), phase=self.phase,
                        position_error_m=float(np.linalg.norm(dp)), orientation_error_rad=float(np.linalg.norm(dr)))
            self.steps += 1
            if self.phase == "gripper":
                self.gripper_steps += 1
            else:
                self.motion_steps += 1
            return action[None, :]
        except Exception as exc:
            self.failure_reason = f"{type(exc).__name__}:{exc}"
            self._trace("error", error=self.failure_reason)
            error_path = self.run_dir / "error.json"
            if not error_path.exists():
                error_path.write_text(json.dumps({"episode": self.episode, "error": self.failure_reason}))
            raise

    def close(self):
        self._trace("closed")


def build_adapter(**kwargs):
    return RoboMemArenaAdapter(**kwargs)
