"""Task11 uses cookies/top and butter/middle, including legacy entrypoints."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Put cookies into the top drawer, then put butter into the middle drawer."


class Task11ContractTest(unittest.TestCase):
    def test_prompt_and_bddl_entrypoints_agree(self):
        for relative in ("scripts/eval_common.py", "openpi_minimal_runtime/task_prompts.py"):
            tree = ast.parse((ROOT / relative).read_text())
            prompts = next(ast.literal_eval(node.value) for node in tree.body
                           if isinstance(node, ast.Assign) and any(
                               isinstance(target, ast.Name) and target.id == "TASK_PROMPTS"
                               for target in node.targets))
            self.assertEqual(prompts["task11"], PROMPT, relative)
            self.assertNotIn(PROMPT, [value for key, value in prompts.items() if key != "task11"])
        for base in (ROOT, ROOT.parent):
            text = (base / "bddl/11_cookies_top_butter_middle.bddl").read_text()
            self.assertIn(f"(:language {PROMPT})", text)
            goal = text.split("(:goal", 1)[1]
            self.assertIn("(In cookies_1 wooden_cabinet_1_top_region)", goal)
            self.assertIn("(In butter_1 wooden_cabinet_1_middle_region)", goal)
            self.assertNotIn("chocolate_pudding_1", goal)

    def test_reference_semantics_keep_legacy_action_labels(self):
        for relative in ("async_vlm26_reference/fullvlm_v2_26_memory_tasks.json",
                         "reference_evaluation/tasks2_26_vlm5_reference/fullvlm_v2_26_memory_tasks.json"):
            config = json.loads((ROOT / relative).read_text())
            task = next(task for task in config["tasks"] if task["task_id"] == 11)
            self.assertEqual(task["brief_description"], PROMPT)
            for key in ("task_name", "task_block", "scene_description"):
                self.assertIn("butter", task[key].lower())
                self.assertNotIn("chocolate", task[key].lower())
            self.assertIn("historical", task["notes"])
            self.assertEqual(task["primitive_order"][5:7], [
                {"stem": "pick_chocolate", "label": "pick chocolate"},
                {"stem": "place_chocolate", "label": "place chocolate"},
            ])

    def test_butter_required_and_final_drawer_close_optional(self):
        # Compile the actual stage factories without importing simulator/model runtimes.
        names = {"_in_container_site", "_task_specs", "_optional_final_stage_name",
                 "_counted_stage_names", "_stage_success_from_stage_done", "_stage_score_pct"}
        for relative in ("scripts/task2_26_reference_stage.py",
                         "openpi_minimal_runtime/retry_tasks2_26_stage_from_anygrasp.py"):
            with self.subTest(source=relative):
                tree = ast.parse((ROOT / relative).read_text())
                selected = [node for node in tree.body
                            if isinstance(node, ast.FunctionDef) and node.name in names]
                namespace = {
                    "Any": Any, "Callable": Callable,
                    "StageSpec": lambda name, check_fn: SimpleNamespace(name=name, check_fn=check_fn),
                    "_drawer_open_abs": lambda *args: lambda *args: True,
                    "_drawer_closed_abs": lambda *args: lambda *args: True,
                }
                # Preserve the actual optional-stage mapping as well as the functions.
                selected += [node for node in tree.body if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "DRAWER_TASK_OPTIONAL_FINAL_STAGE"
                    for target in node.targets)]
                exec(compile(ast.Module(body=selected, type_ignores=[]), relative, "exec"), namespace)
                sites = {"wooden_cabinet_1_top_region": np.array([0., 0., 1.]),
                         "wooden_cabinet_1_middle_region": np.array([0., 0., .5])}
                top = sites["wooden_cabinet_1_top_region"] + np.array([0., 0., .05])
                middle = sites["wooden_cabinet_1_middle_region"] + np.array([0., 0., .05])
                positions = {"cookies_1": top, "butter_1": top,
                             "chocolate_pudding_1": middle}
                namespace["_current_body_pos"] = lambda env, name: positions.get(name)
                namespace["_current_site_pos"] = lambda env, name: sites.get(name)
                specs = namespace["_task_specs"](11)
                self.assertEqual(len(specs), 6)
                self.assertEqual(specs[4].name, "05_Place_Butter_Middle_Drawer")
                done = {spec.name: True for spec in specs}
                done[specs[1].name] = specs[1].check_fn(None, {}, 0)
                self.assertTrue(done[specs[1].name])
                done[specs[4].name] = specs[4].check_fn(None, {}, 0)
                self.assertFalse(done[specs[4].name])
                self.assertFalse(namespace["_stage_success_from_stage_done"](11, done))
                self.assertEqual(namespace["_stage_score_pct"](11, done), 80.)
                positions["butter_1"], positions["chocolate_pudding_1"] = (
                    positions["chocolate_pudding_1"], positions["butter_1"])
                done[specs[4].name] = specs[4].check_fn(None, {}, 0)
                done[specs[-1].name] = False
                self.assertTrue(done[specs[4].name])
                self.assertTrue(namespace["_stage_success_from_stage_done"](11, done))
                self.assertEqual(namespace["_stage_score_pct"](11, done), 100.)


if __name__ == "__main__":
    unittest.main()
