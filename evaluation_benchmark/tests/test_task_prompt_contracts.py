"""Check public evaluation defaults without importing simulator runtimes."""
import argparse
import ast
from pathlib import Path
import unittest


class TaskPromptContractTest(unittest.TestCase):
    def test_requested_object_and_order_wording(self):
        source = Path(__file__).resolve().parents[1] / "scripts/eval_common.py"
        assignments = [node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.Assign)]
        prompts = next(ast.literal_eval(node.value) for node in assignments
                       if any(isinstance(t, ast.Name) and t.id == "TASK_PROMPTS"
                              for t in node.targets))
        self.assertEqual(len(prompts), 26)
        for task in (4, 5):
            self.assertIn("in this order: top, middle, bottom", prompts[f"task{task}"])
            self.assertIn("Open and close each drawer", prompts[f"task{task}"])
        self.assertEqual(prompts["task11"],
                         "Put cookies into the top drawer, then put butter into the middle drawer.")
        self.assertEqual(prompts["task19"],
                         "Pick and place tomato sauce, milk, and orange juice from cabinet1 to cabinet2, in that order.")
        self.assertEqual(prompts["task25"],
                         "First pick and place butter from plate1 to plate2, then pick and place cream from plate1 to plate2.")

    def test_public_sweep_defaults_to_50_trials_and_docs_match(self):
        root = Path(__file__).resolve().parents[1]
        source = root / "scripts/run_all_tasks1_26.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "build_argparser")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        parser = namespace["build_argparser"]()
        required = ["--adapter-spec", "adapter.py:build", "--out-root", "outputs/test"]
        args = parser.parse_args(required)
        self.assertEqual(args.num_trials_per_task, 50)
        self.assertEqual(args.seed, 50)
        self.assertEqual(args.seed + args.num_trials_per_task - 1, 99)
        self.assertEqual((args.task_end - args.task_start + 1) * args.num_trials_per_task, 1300)
        override = parser.parse_args(required + ["--num-trials-per-task", "2"])
        self.assertEqual(override.num_trials_per_task, 2)
        for doc in (root / "README.md", root / "docs/evaluate_your_model.md"):
            self.assertIn("--num-trials-per-task 50", doc.read_text())
            self.assertNotIn("--num-trials-per-task 51", doc.read_text())


if __name__ == "__main__":
    unittest.main()
