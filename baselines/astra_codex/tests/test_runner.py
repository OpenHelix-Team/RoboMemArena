"""CPU-only release contracts; synthetic scores are never benchmark results."""
import ast
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run as runner


class RunnerTests(unittest.TestCase):
    def test_renderer_patch_does_not_accumulate_over_1300_episodes(self):
        benchmark = Path(os.environ.get('ASTRA_BENCHMARK_ROOT',
                         runner.ROOT.parents[1] / 'evaluation_benchmark'))
        source = benchmark / 'scripts/task2_26_reference_stage.py'
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_patch_env_resolution')

        class Env:
            def __init__(self, **kwargs):
                self.shape = (kwargs.get('camera_heights'), kwargs.get('camera_widths'))

        ec = SimpleNamespace(_get_env_class=lambda: Env)
        namespace = {'ec': ec}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        for _ in range(1300):
            runner.initialize_renderer(namespace['_patch_env_resolution'])
            original = ec._get_env_class
            base = original()
            ec._get_env_class = lambda: base
            try:
                self.assertEqual(ec._get_env_class()().shape, (480, 640))
            finally:
                ec._get_env_class = original

    def test_default_plan_is_1300_fresh_episodes(self):
        args = runner.parser().parse_args([])
        self.assertEqual(args.num_trials, 50)
        self.assertEqual(args.tasks, list(range(1, 27)))
        self.assertEqual(list(range(args.seed, args.seed + args.num_trials)), list(range(50, 100)))
        self.assertEqual(args.num_trials * len(args.tasks), 1300)
        profiles = runner.read(runner.ROOT / 'profiles.json')['tasks']
        self.assertEqual(profiles['22']['max_steps'], 3500)
        self.assertTrue(all(p['max_steps'] == 2500 for k, p in profiles.items() if k != '22'))

    def test_aggregate_uses_task_weighting(self):
        out = runner.aggregate([{'task_id': 1, 'TSR': 100, 'CSR': 100},
                                {'task_id': 1, 'TSR': 0, 'CSR': 50},
                                {'task_id': 2, 'TSR': 100, 'CSR': 100}])
        self.assertEqual(out['episodes'], 3)
        self.assertEqual(out['TSR'], 75)
        self.assertEqual(out['CSR'], 87.5)

    def test_existing_output_cannot_be_overwritten(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'receipt.json'
            runner.write(p, {'seed': 50})
            with self.assertRaises(FileExistsError):
                runner.write(p, {'seed': 51})
            self.assertEqual(runner.read(p), {'seed': 50})

    def test_official_call_and_prompt_restoration_on_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / 'benchmark'
            args = SimpleNamespace(benchmark_root=bench)
            original_env = Mock(return_value=object)
            ec = SimpleNamespace(__file__=str(bench / 'scripts/eval_common.py'),
                                 _get_env_class=original_env, TASK_PROMPTS={'task22': 'original'})
            tasks = SimpleNamespace(__file__=str(bench / 'scripts/eval_tasks2_26.py'),
                                    _patch_env_resolution=Mock())
            policy = SimpleNamespace(__file__=str(bench / 'scripts/policy_adapter.py'))
            suite = SimpleNamespace(__file__=str(bench / 'scripts/run_all_tasks1_26.py'),
                                    _run_task=Mock(side_effect=RuntimeError('test environment failure')))

            class Base:
                failure_reason = None
                def __init__(self, run_dir):
                    self.run_dir = Path(run_dir)
                def close(self):
                    pass

            modules = {'eval_common': ec, 'eval_tasks2_26': tasks, 'policy_adapter': policy,
                       'run_all_tasks1_26': suite,
                       'robomemarena_adapter': SimpleNamespace(RoboMemArenaAdapter=Base)}
            client = Mock()
            with patch.dict(sys.modules, modules), patch.object(runner, 'setup_environment'):
                with self.assertRaisesRegex(RuntimeError, 'test environment failure'):
                    runner.evaluate_episode(args, {'prompt': 'test goal', 'max_steps': 3500,
                        'reasoning_effort': 'max'}, 22, 99, root / 'episode', client)
            call = suite._run_task.call_args.kwargs
            self.assertEqual((call['task_id'], call['seed'], call['max_steps']), (22, 99, 3500))
            self.assertEqual((call['num_trials_per_task'], call['replan_steps'],
                              call['num_steps_wait']), (1, 10, 10))
            self.assertTrue(call['fail_on_extra_pour'])
            self.assertEqual(call['extra_pour_monitor_steps'], 30)
            self.assertIs(ec._get_env_class, original_env)
            self.assertEqual(ec.TASK_PROMPTS, {'task22': 'original'})
            client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
