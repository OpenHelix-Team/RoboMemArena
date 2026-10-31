# Evaluate Your Model on RoboMemArena

This page explains the single integration interface for evaluating your model on the RoboMemArena 1-26 task benchmark.

The benchmark code handles the evaluation loop:

1. create the RoboMemArena/LIBERO environment
2. load the task BDDL and task prompt
3. reset the environment
4. collect the current observation
5. pass `obs + prompt` to your adapter
6. execute the returned action chunk
7. record videos and compute CSR/TSR

Your model only needs to provide an adapter that returns actions.

## Adapter Interface

Implement the adapter interface in:

```text
evaluation_benchmark/scripts/example_policy_adapter_template.py
```

The required method is:

```python
def infer_actions(self, obs, prompt, resize_size):
    """Return a numpy action chunk with shape [horizon, action_dim]."""
```

The adapter can contain any model architecture. For example:

- a single policy that directly maps `obs + prompt` to an action chunk
- a VLM planner that chooses a subtask, followed by a VLA policy that outputs actions
- a policy server client that forwards `obs + prompt` to a remote model
- custom preprocessing for images, states, prompts, or action dimensions

All of them use the same external contract: `obs + prompt -> action chunk`.

## Run the 1-26 Evaluation

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/your_checkpoint"}' \
  --num-trials-per-task 50 \
  --seed 50 \
  --out-root outputs/your_model_eval_1_26
```

For counting-pour tasks, strict extra-pour rejection is enabled by default. The
second completed pour starts a 30-environment-step post-stage monitor; a third
completed pour during that window fails the episode. Use
`--no-fail-on-extra-pour` only for diagnostic runs, or change the window with
`--post-stage-steps` / `--extra-pour-monitor-steps`.

This wrapper records every episode. It does not retry seeds and does not filter for non-zero stage scores.
The sweep uses the benchmark reference stage/goal checkers, so external model evaluation follows the same 1-26 scoring setting as the reference evaluation path.

Task11 places cookies in the top drawer and butter in the middle drawer. See the
[Task11 instruction and scoring contract](task11_contract.md) for the corrected
prompt, unchanged scoring predicates, and legacy VLA label compatibility.

`--adapter-spec` points to a Python file and factory function:

```text
/abs/path/to/your_adapter.py:build_adapter
```

The factory should return an object with `reset()` and `infer_actions(obs, prompt, resize_size)`.

## VLM/VLA Models

A VLM/VLA system should still be exposed through the same adapter interface. Inside the adapter, you can run the VLM planner, maintain a subtask buffer, call a VLA policy server, convert observations, and return the final action chunk.

For a reference implementation of that kind of stack, inspect:

```text
evaluation_benchmark/async_vlm26_reference/
evaluation_benchmark/reference_evaluation/
```

Those folders show one way to connect a VLM planner and VLA policy to RoboMemArena. They are examples of the same benchmark setting, not a separate scoring definition.

Useful environment variables for the reference VLM/VLA stack include:

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional if using a bundled minimal OpenPI runtime
export OPENPI_INFERENCE_ROOT=/abs/path/to/openpi_inference
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm_checkpoint
export VLA_CKPT=/abs/path/to/vla_checkpoint
export VLA_CONFIG=<your_vla_config_name>  # optional; default runner value is pi05_robomemarena
```

## Metrics

- `CSR`: average stage completion rate. In output files this is stored in the legacy-compatible `goal_success_rate` / `goal_success_rate_pct` field.
- `stage_success_rate`: strict required-stage completion rate. This is the task success rate (TSR): an episode has TSR=1 only when every required reference stage is completed.
- `average_score_pct`: diagnostic partial stage/process completion score.

All tasks use stage-based scoring. Tasks 6, 7, 8, 9, 10, 15, 16, and 22 are
counting-pour tasks; their `stage_success` additionally requires the default
30-step monitor to finish without detecting a third pour. A pour is counted
from the manipulated object body: at least `0.15 rad` away from its stage
baseline followed by a return to within `0.10 rad`. Microwave tasks do not
require the final microwave-close stage.

Tasks 1, 2, and 3 accept either `basket_1` or `basket_2`. Their object pairs are
cookies and tomato sauce (Task 1), butter and popcorn (Task 2), and cream cheese
and chocolate pudding (Task 3). The first stage checks that the first object is
in either basket; the second checks that both objects are currently in the same
basket. Placing them in different baskets, or removing the first object before
placing the second, fails the second stage. The existing placement distance and
height thresholds and stage names are unchanged.
These stage rules also determine goal success, overriding the scenes' fixed
`basket_1` BDDL goals. The canonical public/reference scorer and the legacy
Task2-26 retry runner use the same basket rules. Results from older fixed-basket
scorer versions must be reported separately.

For official reporting, use the same adapter and scoring path consistently across all 1-26 tasks.
