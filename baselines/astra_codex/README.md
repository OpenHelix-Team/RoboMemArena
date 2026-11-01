# Astra (Codex) on RoboMemArena

A visual waypoint policy for the 26 RoboMemArena tasks. `gpt-6-astra` receives
front-camera history, the current wrist image, robot proprioception and camera
calibration, then returns a Cartesian waypoint and gripper command. The existing
OSC controller executes the waypoint. This baseline requires no policy training.

The model uses Codex's native app-server with ChatGPT login. No API-key fallback
is used. Each episode has its own ephemeral conversation; observation history
is retained within the episode and cleared between episodes.

## Setup

1. Install the RoboMemArena environment and dependencies following
   [the benchmark instructions](../../evaluation_benchmark/README.md). Use Python
   3.10 or newer, an EGL-capable GPU, NumPy, SciPy, Pillow, OpenCV, MuJoCo and
   robosuite as required by that environment.
   `requirements.txt` records the policy-side versions used with the pinned
   benchmark; the environment also used PyTorch 2.7.1. Follow the benchmark setup
   for LIBERO assets and its remaining dependencies.
2. Install a Codex CLI that supports the `gpt-6-astra` model and the reasoning
   efforts in `profiles.json`. Run `codex login` as the user who will run inference.
   Model and effort availability depend on that account's access.
   The evaluated transport used `codex-cli 0.162.0-alpha.17.2`.
3. Create a separate checkout of the pinned evaluation code. From the repository
   root:

   ```bash
   git worktree add --detach ../RoboMemArena-eval-c899 \
     c8994677ddbbd1c74324ba98bcf15f9b9305497f
   ```

The release checks evaluator and policy hashes before inference. This fixes the
scorer, scene definitions and preprocessing rather than silently using a later
benchmark revision. The pinned Task11 instruction has a known object mismatch
with its scorer; the released profile preserves that historical contract.

## Run 50 episodes per task

From the repository root, inside your GPU allocation or GPU-enabled environment:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Validate the configuration without a model turn or GPU initialization.
python baselines/astra_codex/run.py \
  --benchmark-root ../RoboMemArena-eval-c899/evaluation_benchmark --check

# 26 tasks × 50 episodes; seeds 50–99; fresh output directory.
python baselines/astra_codex/run.py \
  --benchmark-root ../RoboMemArena-eval-c899/evaluation_benchmark \
  --run-dir outputs/astra_codex_50 \
  --num-trials 50 --seed 50
```

If Codex is not on `PATH`, pass `--codex-bin /path/to/codex` or set `CODEX_BIN`.
On a machine with multiple GPUs, configure the EGL device for the GPU allocated
to this process before launching; CUDA and EGL indices are not interchangeable.

To evaluate one task with the same 50-episode default:

```bash
python baselines/astra_codex/run.py \
  --benchmark-root ../RoboMemArena-eval-c899/evaluation_benchmark \
  --task 22 --run-dir outputs/astra_codex_task22_50
```

`profiles.json` supplies the task-specific prompt, reasoning effort and step
budget. Most tasks use 2500 control steps; Task22 uses 3500. The runner uses
`replan_steps=10`, 10 settling steps, a 150-call policy budget, and the official
extra-pour check. `--reasoning-effort` and `--max-steps` can override profiles
for a new comparison; actual settings are written to each run manifest.

The profile combines task-specific choices made during development. Task7 uses
the gradual-pour wording, and Tasks20/21/23/24 clarify the object's new location.
The scorer and controller are unchanged. This is a recipe for new evaluation,
not a guarantee of a fixed score: stochastic model outputs and environment
variation can change outcomes. This code package includes no precomputed results.

## Outputs and failure handling

Each new run produces its own manifests, model exchanges, controller traces,
videos and official episode summaries. After all requested episodes finish,
`summary.json` reports task-weighted means:

- **TSR:** full-task success, requiring all mandatory stages and no forbidden pour.
- **CSR:** cumulative completion over the required stages.

Scientific failures remain in the denominator. Transport, authentication or
environment interruptions are preserved as incomplete attempts and stop the
run; a model call is never silently replayed. Reusing an existing run directory
is refused so that histories and attempt lineages cannot be mixed.

## Implementation

- `codex_backend.py`: native ChatGPT-authenticated transport and isolated threads.
- `robomemarena_adapter.py`: RGB/history input, waypoint parsing and feedback OSC.
- `run.py`: portable local transport and pinned official evaluation loop.
- `profiles.json`: task prompts, effort and control-step budgets.
- `provenance.json`: benchmark pin and unchanged policy source hashes.

The backend and controller are byte-identical to the evaluated implementation.
The portable runner replaces the cluster's HTTP forwarding and account-specific
scheduling with local calls that preserve the supplied text and image bytes.
It never supplies object identities, object poses, depth or scorer thresholds
to the policy. Both cameras follow the benchmark's single vertical flip and
resize convention; no second flip is applied by the adapter.

CPU-only checks, without model calls:

```bash
ASTRA_BENCHMARK_ROOT=../RoboMemArena-eval-c899/evaluation_benchmark \
  python -m unittest discover -s baselines/astra_codex/tests -v
```

Based on [embodied-astra-robomme](https://github.com/jackbjed/embodied-astra-robomme).
The upstream waypoint method is adapted to RoboMemArena's environment and OSC
execution. The MIT license and upstream copyright notice are included in
`LICENSE`; RoboMemArena and its dependencies retain their respective licenses.
