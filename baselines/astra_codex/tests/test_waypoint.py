"""CPU-only checks for the waypoint bridge; no simulator or model is created."""
from pathlib import Path
import ast
import os
import importlib.util
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path(os.environ.get("ASTRA_BENCHMARK_ROOT", ROOT.parents[1] / "evaluation_benchmark")) / "scripts")]
from robomemarena_adapter import RoboMemArenaAdapter, camera_calibration, osc_action, parse_waypoint, pose_error


class WaypointTest(unittest.TestCase):
    def test_declared_complete_holds_without_further_model_requests(self):
        with TemporaryDirectory() as run_dir:
            adapter = RoboMemArenaAdapter(run_dir)
            adapter.reset()
            adapter._state = Mock(return_value={})
            adapter._ask = Mock(return_value='{"waypoint":null}')
            image = np.zeros((2, 2, 3), dtype=np.uint8)
            for _ in range(3):
                obs = {"observation/image": image, "observation/wrist_image": image}
                adapter.observe(obs, "test task", 2)
                action = adapter.infer_actions(obs, "test task", 2)
                np.testing.assert_array_equal(action, np.zeros((1, 7), dtype=np.float32))
            adapter._ask.assert_called_once()
            self.assertTrue(adapter.finished)
            self.assertEqual(adapter.model_calls, 1)
            self.assertFalse((Path(run_dir) / "error.json").exists())
            events = [json.loads(line)["event"] for line in (Path(run_dir) / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(events.count("policy_stop"), 1)
            self.assertEqual(events.count("hold"), 3)
            adapter.reset()
            self.assertFalse(adapter.finished)

    def test_native_gripper_zero_preserves_motor_target(self):
        # Load the installed method itself without importing a graphics runtime.
        package = Path(importlib.util.find_spec("robosuite").origin).parent
        source = package / "models/grippers/panda_gripper.py"
        gripper_class = next(node for node in ast.parse(source.read_text()).body
                             if isinstance(node, ast.ClassDef) and node.name == "PandaGripper")
        method = next(node for node in gripper_class.body
                      if isinstance(node, ast.FunctionDef) and node.name == "format_action")
        namespace = {"np": np}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        gripper = SimpleNamespace(dof=1, speed=0.01, current_action=np.array([-0.4, 0.4]))
        before = gripper.current_action.copy()
        result = namespace["format_action"](gripper, np.array([0.]))
        np.testing.assert_array_equal(result, before)
        np.testing.assert_array_equal(gripper.current_action, before)

    def test_world_rotation_and_scaling(self):
        current = Rotation.from_euler("xyz", [0.2, -0.6, 1.3]).as_matrix()
        target = np.array([0.01, -0.02, 0.03, -0.2, -0.5, 1.1])
        out_max = np.array([0.05] * 3 + [0.5] * 3)
        action = osc_action(np.zeros(3), current, target, -1, -1, 1, -out_max, out_max)
        np.testing.assert_allclose(action[:3], [0.2, -0.4, 0.6], atol=1e-6)
        reached = Rotation.from_rotvec(action[3:6] * out_max[3:]).as_matrix() @ current
        np.testing.assert_allclose(reached, Rotation.from_euler("xyz", target[3:]).as_matrix(), atol=1e-6)
        self.assertEqual(action[-1], 1)  # Upstream close -> robosuite close.
        self.assertEqual(osc_action(np.zeros(3), current, target, 1, -1, 1, -out_max, out_max)[-1], -1)
        self.assertEqual(osc_action(np.zeros(3), current, target, 0, -1, 1, -out_max, out_max)[-1], 0)

    def test_wrap_clipping_and_invalid_answers(self):
        current = Rotation.from_euler("xyz", [0, 0, np.pi - 0.01]).as_matrix()
        target = [10, 0, 0, 0, 0, -np.pi + 0.01]
        _, rotation = pose_error(np.zeros(3), current, target)
        np.testing.assert_allclose(rotation, [0, 0, 0.02], atol=1e-8)
        out_max = np.array([0.05] * 3 + [0.5] * 3)
        action = osc_action(np.zeros(3), current, target, 0, -1, 1, -out_max, out_max)
        self.assertEqual(action[0], 1)
        pose, gripper = parse_waypoint('```json\n{"waypoint":[0,0,0,1,2,3],"gripper":-1}\n```')
        self.assertEqual(gripper, -1)
        np.testing.assert_equal(pose, [0, 0, 0, 1, 2, 3])
        self.assertEqual(parse_waypoint('{"waypoint":null}'), (None, 0))
        for invalid in ['{"waypoint":[0,0,0,0,0,NaN]}',
                        '{"waypoint":[0,0,0,0,0,0],"gripper":true}', 'not json']:
            with self.assertRaises(ValueError):
                parse_waypoint(invalid)

    def test_upright_camera_projection_after_resize(self):
        sim = SimpleNamespace(
            model=SimpleNamespace(camera_name2id=lambda _: 0, cam_fovy=np.array([60.])),
            data=SimpleNamespace(cam_xmat=np.eye(3)[None].reshape(1, 9), cam_xpos=np.array([[1., 2., 3.]])))
        calibration = camera_calibration(sim, "camera", (480, 640, 3), (256, 256, 3))
        k = np.array(calibration["K"])
        world_to_camera = np.array(calibration["world_to_camera"])
        np.testing.assert_allclose(k[0, 0] / k[1, 1], 480 / 640)
        center = k @ (world_to_camera @ np.array([1., 2., 2., 1.]))
        np.testing.assert_allclose(center[:2] / center[2], [128, 128])
        above = k @ (world_to_camera @ np.array([1., 2.1, 2., 1.]))
        self.assertLess(above[1] / above[2], 128)  # Upright image, no second vertical flip.


if __name__ == "__main__":
    unittest.main()
