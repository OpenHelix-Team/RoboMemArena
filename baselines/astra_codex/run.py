"""Evaluate Astra waypoints through Codex and the pinned RoboMemArena scorer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import cache
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import pwd
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
INSTRUCTIONS = (
    'You are the robot policy. Follow the supplied robot interface. Return only the requested JSON. '
    'Use only the provided images and state; do not use external tools or inspect files beyond the input images.'
)
POLICY_ERRORS = {'unparseable_model_answer', 'invalid_model_answer',
                 'invalid_waypoint', 'nonfinite_waypoint'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', '--task', type=int, nargs='+', choices=range(1, 27),
                   default=list(range(1, 27)))
    p.add_argument('--num-trials', '--num-trials-per-task', type=int, default=50)
    p.add_argument('--seed', type=int, default=50, help='First episode seed')
    p.add_argument('--benchmark-root', type=Path,
                   default=ROOT.parents[1] / 'evaluation_benchmark')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--reasoning-effort', choices=['medium', 'high', 'xhigh', 'max', 'ultra'])
    p.add_argument('--max-steps', type=int)
    p.add_argument('--codex-bin', help='Codex executable; otherwise CODEX_BIN or PATH')
    p.add_argument('--check', action='store_true', help='Validate sources/plan without a model or GPU')
    return p


def settings(args):
    if args.num_trials < 1 or args.seed < 0 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError('Trial count and step budget must be positive; seed must be nonnegative')
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError('Each task may be selected only once')
    config, provenance = read(ROOT / 'profiles.json'), read(ROOT / 'provenance.json')
    if set(config['tasks']) != {str(t) for t in range(1, 27)}:
        raise ValueError('Profiles must cover exactly tasks 1-26')
    for relative, digest in provenance['unchanged_policy_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Policy source differs from released version: {relative}')
    for relative, digest in provenance['benchmark_files_sha256'].items():
        path = args.benchmark_root / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f'Pinned benchmark mismatch: {relative}. Use benchmark commit '
                             + provenance['benchmark_commit'] + ' as documented in README.md')
    selected = {}
    for task in args.tasks:
        profile = dict(config['tasks'][str(task)])
        if args.reasoning_effort is not None:
            profile['reasoning_effort'] = args.reasoning_effort
        if args.max_steps is not None:
            profile['max_steps'] = args.max_steps
        selected[task] = profile
    return config, provenance, selected


def setup_environment(benchmark, run):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    config = run / 'libero_config'
    config.mkdir()
    lib = benchmark / 'libero_fork/libero'
    paths = {'benchmark_root': str(lib), 'bddl_files': str(lib / 'bddl_files'),
             'init_states': str(lib / 'init_files'), 'datasets': str(lib.parent / 'datasets'),
             'assets': str(lib / 'assets')}
    (config / 'config.yaml').write_text(''.join(f'{k}: {json.dumps(v)}\n' for k, v in paths.items()))
    os.environ['LIBERO_CONFIG_PATH'] = str(config)
    sys.path[:0] = [str(benchmark / 'scripts'), str(benchmark / 'libero_fork')]


@cache
def initialize_renderer(patch_resolution):
    # The pinned patch wraps Env.__init__; apply it once, not once per episode.
    patch_resolution()


def evaluate_episode(args, profile, task, seed, run, client):
    """One fresh episode; no retry or implicit conversation reset after failure."""
    run.mkdir(parents=True, exist_ok=False)
    setup_environment(args.benchmark_root, run)
    import eval_common as ec
    import eval_tasks2_26 as tasks26
    import policy_adapter
    import run_all_tasks1_26 as suite
    from robomemarena_adapter import RoboMemArenaAdapter

    for module in (ec, tasks26, policy_adapter, suite):
        if not Path(module.__file__).resolve().is_relative_to(args.benchmark_root):
            raise RuntimeError('Imported evaluator outside the pinned benchmark: ' + module.__name__)
    initialize_renderer(tasks26._patch_env_resolution)
    broker = run / 'broker'

    class LocalPolicy(RoboMemArenaAdapter):
        # Only transport changes: original frame selection, prompt and OSC are inherited.
        def _ask(self, text, images):
            self.request_id += 1
            write(self.run_dir / 'requests' / f'{self.request_id:06d}.json',
                  {'text': text, 'images': images, 'episode': self.episode,
                   'image_sha256': [sha(p) for p in images]})
            if client.thread_id is None:
                client.start(INSTRUCTIONS)
            answer = {'text': client.ask(text, image_paths=images), 'thread_id': client.thread_id}
            write(self.run_dir / 'responses' / f'{self.request_id:06d}.json', answer)
            self._trace('model_response', request_id=self.request_id, response=answer)
            return answer['text']

    adapter = LocalPolicy(str(broker))
    original_env, original_prompts = ec._get_env_class, dict(ec.TASK_PROMPTS)
    env_class = original_env()

    def factory(*a, **kw):
        env = env_class(*a, **kw)
        adapter.bind_env(env)
        return env

    errors = []

    class CaptureErrors(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.ERROR:
                errors.append(self.format(record))

    capture = CaptureErrors()
    ec._get_env_class = lambda: factory
    ec.TASK_PROMPTS[f'task{task}'] = profile['prompt']
    logging.getLogger().addHandler(capture)
    try:
        write(run / 'manifest.json', {
            'created_at': datetime.now(timezone.utc).isoformat(), 'task': task, 'seed': seed,
            'model': 'gpt-6-astra', **profile, 'replan_steps': 10,
            'benchmark_commit': read(ROOT / 'provenance.json')['benchmark_commit'],
            'producer_commit': subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'code_sha256': {p.name: sha(p) for p in ROOT.glob('*.py')},
            'hostname': platform.node(), 'unix_user': pwd.getpwuid(os.getuid()).pw_name,
            'slurm': {k: os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_JOB_ACCOUNT',
                'SLURM_JOB_PARTITION', 'SLURM_JOB_NODELIST', 'CUDA_VISIBLE_DEVICES',
                'MUJOCO_EGL_DEVICE_ID')}, 'python': sys.executable})
        result = suite._run_task(task_id=task, num_trials_per_task=1, resize_size=256,
            replan_steps=10, num_steps_wait=10, max_steps=profile['max_steps'],
            post_goal_steps=200, video_dir=run / f'videos/task{task}', seed=seed,
            adapter=adapter, fail_on_extra_pour=True, extra_pour_monitor_steps=30)
        suite._write_outputs(run, [result], seed)
        model_error = (adapter.failure_reason or '').removeprefix('ValueError:')
        malformed = len(errors) == 1 and model_error in POLICY_ERRORS
        policy_budget = len(errors) == 1 and adapter.failure_reason == 'RuntimeError:model_call_limit'
        infrastructure = bool(errors) and not (malformed or policy_budget)
        write(run / 'completion.json', {'result': result, 'official_scoring_completed': True,
            'valid_scientific_episode': not infrastructure, 'runtime_errors': errors,
            'model_output_failure': malformed, 'policy_budget_exhausted': policy_budget,
            'infrastructure_error': infrastructure})
        if infrastructure:
            raise RuntimeError('Infrastructure interrupted this episode; preserve it and stop the run')
        return result
    finally:
        logging.getLogger().removeHandler(capture)
        ec._get_env_class = original_env
        ec.TASK_PROMPTS.clear()
        ec.TASK_PROMPTS.update(original_prompts)
        adapter.close()
        client.close()


def aggregate(results):
    tasks = []
    for task in sorted({r['task_id'] for r in results}):
        group = [r for r in results if r['task_id'] == task]
        tasks.append({'task': task, 'episodes': len(group),
                      'TSR': sum(r['TSR'] for r in group) / len(group),
                      'CSR': sum(r['CSR'] for r in group) / len(group)})
    return {'episodes': len(results), 'tasks': tasks,
            'TSR': sum(t['TSR'] for t in tasks) / len(tasks),
            'CSR': sum(t['CSR'] for t in tasks) / len(tasks)}


def main(argv=None):
    args = parser().parse_args(argv)
    args.benchmark_root = args.benchmark_root.resolve()
    config, provenance, profiles = settings(args)
    planned = len(args.tasks) * args.num_trials
    if args.check:
        print(json.dumps({'status': 'PASS', 'planned_episodes': planned,
                          'trials_per_task': args.num_trials, 'seed_start': args.seed,
                          'benchmark_commit': provenance['benchmark_commit']}))
        return
    if args.run_dir is None:
        raise ValueError('--run-dir is required for evaluation')
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    from codex_backend import SharedCodexRuntime
    runtimes, results = {}, []
    write(args.run_dir / 'run_manifest.json', {'model': config['model'],
        'benchmark_commit': provenance['benchmark_commit'], 'tasks': args.tasks,
        'trials_per_task': args.num_trials, 'seed_start': args.seed,
        'planned_episodes': planned, 'profiles': profiles,
        'metrics': {'TSR': 'Full-task success rate', 'CSR': 'Mean required-stage completion'}})
    try:
        for task in args.tasks:
            profile = profiles[task]
            effort = profile['reasoning_effort']
            if effort not in runtimes:
                runtime = SharedCodexRuntime(args.run_dir / 'runtimes' / effort,
                    codex_bin=args.codex_bin, reasoning_effort=effort)
                runtimes[effort] = runtime
                runtime.start()
            for trial in range(args.num_trials):
                seed = args.seed + trial
                run = args.run_dir / f'task{task:02d}_seed{seed:04d}'
                client = runtimes[effort].backend(run / 'broker/codex_trace')
                try:
                    result = evaluate_episode(args, profile, task, seed, run, client)
                finally:
                    client.close()
                results.append(result)
                print(json.dumps({'task': task, 'seed': seed, 'TSR': result['TSR'],
                                  'CSR': result['CSR'], 'completed': len(results),
                                  'planned': planned}), flush=True)
        report = aggregate(results)
        report.update(status='COMPLETE', planned_episodes=planned)
        write(args.run_dir / 'summary.json', report)
    except BaseException as exc:
        write(args.run_dir / 'failure.json', {'status': 'INCOMPLETE',
            'completed_episodes': len(results), 'planned_episodes': planned,
            'error': f'{type(exc).__name__}: {exc}'})
        raise
    finally:
        for runtime in runtimes.values():
            runtime.close()


if __name__ == '__main__':
    main()
