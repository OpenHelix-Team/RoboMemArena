from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_tasks2_26 as eval_tasks  # noqa: E402
import task2_26_reference_stage as stages  # noqa: E402


class _FakeModel:
    body_names = ["tomato_sauce_1"]

    @staticmethod
    def body_name2id(name: str) -> int:
        if name not in {"tomato_sauce_1", "tomato_sauce_1_main"}:
            raise KeyError(name)
        return 0


class _FakeData:
    def __init__(self) -> None:
        self.body_xmat = np.eye(3, dtype=np.float64).reshape(1, 9)


class _FakeSim:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.data = _FakeData()


class _TiltEnv:
    def __init__(self) -> None:
        self.sim = _FakeSim()

    def set_tilt(self, angle: float) -> None:
        c, s = np.cos(angle), np.sin(angle)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        self.sim.data.body_xmat[0] = rot_x.reshape(9)


class _RolloutEnv:
    def reset(self):
        return {"step": 0}

    def step(self, action):
        del action
        return {"step": 1}, 0.0, False, {}


class _Adapter:
    @staticmethod
    def infer_actions(**kwargs):
        del kwargs
        return np.zeros((1, 7), dtype=np.float32)


class CountingPourStageTest(unittest.TestCase):
    def test_task22_requires_microwave_stages_but_not_final_close(self) -> None:
        names = [spec.name for spec in stages._task_specs(22)]
        self.assertEqual(
            names,
            [
                "01_Lift_Tomato_Sauce",
                "02_Pour_One",
                "03_Pour_Two",
                "04_Place_Tomato_Aside",
                "05_Open_Microwave",
                "06_Place_Cookies_Microwave",
                "07_Close_Microwave",
            ],
        )

        stage_done = {name: True for name in names}
        stage_done["07_Close_Microwave"] = False
        self.assertEqual(stages._counted_stage_names(22, stage_done), names[:-1])
        self.assertTrue(stages._stage_success_from_stage_done(22, stage_done))
        self.assertEqual(stages._stage_score_pct(22, stage_done), 100.0)

        stage_done["05_Open_Microwave"] = False
        self.assertFalse(stages._stage_success_from_stage_done(22, stage_done))

    def test_task22_ends_after_required_stages_without_close(self) -> None:
        names = [spec.name for spec in stages._task_specs(22)]
        specs = [
            stages.StageSpec(name, lambda env, state, start: state["step_idx"] > start)
            for name in names[:-1]
        ]
        specs.append(stages.StageSpec(names[-1], lambda env, state, start: False))

        def update_state(obs, state):
            del obs
            state["step_idx"] += 1

        with (
            mock.patch.object(eval_tasks, "_build_initial_state", return_value={"step_idx": 0}),
            mock.patch.object(eval_tasks, "_update_state", side_effect=update_state),
            mock.patch.object(eval_tasks, "_is_counting_pour_task", return_value=True),
            mock.patch.object(eval_tasks, "_extra_pour_check", return_value=lambda env, state, start: False),
            mock.patch.object(
                eval_tasks,
                "build_eval26_policy_input",
                return_value=({}, np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)),
            ),
        ):
            _, stage_done, _, diagnostics, _, _ = eval_tasks.run_episode_with_stateful_stages(
                task_id=22,
                env=_RolloutEnv(),
                adapter=_Adapter(),
                prompt="test",
                resize_size=2,
                replan_steps=1,
                num_steps_wait=0,
                max_steps=20,
                post_goal_steps=200,
                stage_specs=specs,
                goal_monitor_dict={},
                goal_check_override=None,
                fail_on_extra_pour=False,
                extra_pour_monitor_steps=30,
            )

        self.assertTrue(diagnostics["stage_success"])
        self.assertFalse(stage_done["07_Close_Microwave"])

    def test_counting_stage_maps(self) -> None:
        expected = {
            6: ["01_Lift_Tomato_Sauce", "02_Pour_One", "03_Pour_Two"],
            7: ["01_Lift_Tomato_Sauce", "02_Pour_One", "03_Pour_Two"],
            8: ["01_Place_Pudding_Frypan", "02_Lift_Tomato_Sauce", "03_Pour_One", "04_Pour_Two"],
            9: ["01_Place_Butter_Frypan", "02_Lift_Tomato_Sauce", "03_Pour_One", "04_Pour_Two"],
            10: ["01_Lift_Wine_Bottle", "02_Pour_One", "03_Pour_Two"],
            15: ["01_Place_Butter_Frypan", "02_Lift_Milk", "03_Pour_One", "04_Pour_Two"],
            16: ["01_Lift_Milk", "02_Pour_One", "03_Pour_Two"],
            22: [
                "01_Lift_Tomato_Sauce",
                "02_Pour_One",
                "03_Pour_Two",
                "04_Place_Tomato_Aside",
                "05_Open_Microwave",
                "06_Place_Cookies_Microwave",
                "07_Close_Microwave",
            ],
        }
        actual = {task_id: [spec.name for spec in stages._task_specs(task_id)] for task_id in expected}
        self.assertEqual(actual, expected)

    def test_extra_pour_flag_defaults_on(self) -> None:
        parser = eval_tasks.build_argparser()
        default_args = parser.parse_args(["--task-id", "6", "--adapter-spec", "adapter.py:build"])
        disabled_args = parser.parse_args(
            ["--task-id", "6", "--adapter-spec", "adapter.py:build", "--no-fail-on-extra-pour"]
        )
        self.assertTrue(default_args.fail_on_extra_pour)
        self.assertEqual(default_args.extra_pour_monitor_steps, 30)
        self.assertFalse(disabled_args.fail_on_extra_pour)

    def test_body_pour_requires_tilt_and_return(self) -> None:
        env = _TiltEnv()
        state = {"step_idx": 0, "tilt_angles": []}
        check = stages._body_pour_stage("tomato_sauce_1")
        detected = False
        for step, angle in enumerate([0, 0, 0, 0, 0, 0.03, 0.08, 0.16, 0.20, 0.12, 0.09], start=1):
            state["step_idx"] = step
            env.set_tilt(angle)
            detected = check(env, state, 1)
        self.assertTrue(detected)

    def test_body_pour_does_not_count_without_return(self) -> None:
        env = _TiltEnv()
        state = {"step_idx": 0, "tilt_angles": []}
        check = stages._body_pour_stage("tomato_sauce_1")
        detected = False
        for step, angle in enumerate([0, 0, 0, 0, 0, 0.03, 0.08, 0.16, 0.20, 0.18, 0.17], start=1):
            state["step_idx"] = step
            env.set_tilt(angle)
            detected = check(env, state, 1)
        self.assertFalse(detected)

    def test_counting_task_skips_goal_and_waits_for_monitor(self) -> None:
        specs = [
            stages.StageSpec("01_Lift_Tomato_Sauce", lambda env, state, start: state["step_idx"] > start),
            stages.StageSpec("02_Pour_One", lambda env, state, start: state["step_idx"] > start),
            stages.StageSpec("03_Pour_Two", lambda env, state, start: state["step_idx"] > start),
        ]

        def update_state(obs, state):
            del obs
            state["step_idx"] += 1

        with (
            mock.patch.object(eval_tasks, "_build_initial_state", return_value={"step_idx": 0}),
            mock.patch.object(eval_tasks, "_update_state", side_effect=update_state),
            mock.patch.object(eval_tasks, "_is_counting_pour_task", return_value=True),
            mock.patch.object(eval_tasks, "_extra_pour_check", return_value=lambda env, state, start: False),
            mock.patch.object(eval_tasks.ec, "check_goal_success", side_effect=AssertionError("goal checker called")),
            mock.patch.object(
                eval_tasks,
                "build_eval26_policy_input",
                return_value=({}, np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)),
            ),
        ):
            _, _, goal_success, diagnostics, _, _ = eval_tasks.run_episode_with_stateful_stages(
                task_id=6,
                env=_RolloutEnv(),
                adapter=_Adapter(),
                prompt="test",
                resize_size=2,
                replan_steps=1,
                num_steps_wait=0,
                max_steps=10,
                post_goal_steps=200,
                stage_specs=specs,
                goal_monitor_dict={"must_not_be_used": []},
                goal_check_override=None,
                fail_on_extra_pour=True,
                extra_pour_monitor_steps=2,
            )

        self.assertIsNone(goal_success)
        self.assertTrue(diagnostics["stage_success"])
        self.assertEqual(diagnostics["pour_2_step"], 2)
        self.assertEqual(diagnostics["extra_monitor_end_step"], 4)

    def test_third_pour_fails_episode(self) -> None:
        specs = [
            stages.StageSpec("01_Lift_Tomato_Sauce", lambda env, state, start: state["step_idx"] > start),
            stages.StageSpec("02_Pour_One", lambda env, state, start: state["step_idx"] > start),
            stages.StageSpec("03_Pour_Two", lambda env, state, start: state["step_idx"] > start),
        ]

        def update_state(obs, state):
            del obs
            state["step_idx"] += 1

        with (
            mock.patch.object(eval_tasks, "_build_initial_state", return_value={"step_idx": 0}),
            mock.patch.object(eval_tasks, "_update_state", side_effect=update_state),
            mock.patch.object(eval_tasks, "_is_counting_pour_task", return_value=True),
            mock.patch.object(eval_tasks, "_extra_pour_check", return_value=lambda env, state, start: state["step_idx"] > start),
            mock.patch.object(
                eval_tasks,
                "build_eval26_policy_input",
                return_value=({}, np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)),
            ),
        ):
            _, _, _, diagnostics, _, _ = eval_tasks.run_episode_with_stateful_stages(
                task_id=6,
                env=_RolloutEnv(),
                adapter=_Adapter(),
                prompt="test",
                resize_size=2,
                replan_steps=1,
                num_steps_wait=0,
                max_steps=10,
                post_goal_steps=200,
                stage_specs=specs,
                goal_monitor_dict={},
                goal_check_override=None,
                fail_on_extra_pour=True,
                extra_pour_monitor_steps=2,
            )

        self.assertFalse(diagnostics["stage_success"])
        self.assertTrue(diagnostics["extra_pour_detected"])
        self.assertEqual(diagnostics["failure_reason"], "extra_pour")


if __name__ == "__main__":
    unittest.main()
