from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OFFICIAL_COMMIT = "d9f83ac5182e25ad7f0a301a77a0b667f2392df1"
OFFICIAL_SCORE_RE = re.compile(
    r"\[OFFICIAL_SCORE\]\s+task=(?P<task>\d+)\s+"
    r"average_score_pct=(?P<score>[0-9.]+)\s+"
    r"stage_success=(?P<stage_success>[01])\s+"
    r"goal_success=(?P<goal_success>[01])\s+"
    r"stage_done_json=(?P<stage_done>\{[^\r\n]*\})"
)
EPISODE_SCORE_RE = re.compile(
    r"Episode\s+(?P<episode>\d+)\s+seed=(?P<seed>\d+)\s+"
    r"stage_score=(?P<score>[0-9.]+)\s+"
    r"stage_success=(?P<stage_success>[01])\s+"
    r"goal=(?P<goal>[^\s]+).*?\|\s*(?P<stages>[^\r\n]+)"
)
EPISODE_ID_RE = re.compile(r"Episode\s+(?P<episode>\d+)\s*(?:\(|\s)seed=(?P<seed>\d+)")
STAGE_TOKEN_RE = re.compile(
    r"(?:^|\|\s*)(?P<name>[^|=]+?)=(?P<done>[YN])(?=\s*(?:\||$))"
)


@dataclass(frozen=True)
class ScoreCandidate:
    log_path: Path
    task_id: int
    score: float
    stage_success: bool
    goal_success: bool
    stage_done: dict[str, bool]
    source_seed: int
    source_episode: int


@dataclass(frozen=True)
class SourceIdentity:
    seed: int
    episode: int


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def _repo_root() -> Path:
    lock_path = Path(_required_env("OFFICIAL_SCORER_LOCK")).resolve()
    return lock_path.parents[2]


def _source_command() -> list[str]:
    raw_json = os.environ.get("SOURCE_COMMAND_JSON", "").strip()
    if raw_json:
        value = json.loads(raw_json)
        if not isinstance(value, list) or not value:
            raise RuntimeError("SOURCE_COMMAND_JSON must be a non-empty JSON list")
        return [str(item) for item in value]
    raw_shell = _required_env("SOURCE_COMMAND")
    return shlex.split(raw_shell)


def _runtime_env(repo_root: Path, output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    task_id = _required_env("TASK_ID")
    episode_index = _required_env("EPISODE_INDEX")
    seed = _required_env("SEED")
    port = int(
        env.get(
            "PORT",
            str(15000 + int(task_id) * 100 + int(episode_index)),
        )
    )
    run_id = f"task{int(task_id):02d}_ep{int(episode_index):03d}_seed{seed}"
    official_runtime_root = Path(_required_env("OFFICIAL_RUNTIME_ROOT")).resolve()
    _verify_official_runtime(repo_root, official_runtime_root)
    official_scripts = official_runtime_root / "evaluation_benchmark/scripts"
    official_bddl = official_runtime_root / "bddl"
    legacy_root = output_dir / "legacy_output"
    if legacy_root.exists() and any(legacy_root.iterdir()):
        raise RuntimeError(f"legacy output directory is not empty: {legacy_root}")
    legacy_root.mkdir(parents=True, exist_ok=True)
    norm_path = Path(_required_env("NORM_STATS_PATH")).resolve()
    env.update(
        {
            "TASK_ID": task_id,
            "EPISODE_INDEX": episode_index,
            "SEED": seed,
            "OFFICIAL_SEED": seed,
            "NUM_TRIALS": "1",
            "OFFICIAL_NUM_TRIALS": "1",
            "NUM_TRIALS_PER_TASK": "1",
            "RUN_ID": run_id,
            "OFFICIAL_RUN_STAMP": run_id,
            "STAMP": run_id,
            "PORT": str(port),
            "OUTPUT_DIR": str(output_dir),
            "OUT_ROOT": str(legacy_root),
            "OUTPUT_ROOT": str(legacy_root),
            "OUTPUT_ROOT_OVERRIDE": str(legacy_root),
            "OFFICIAL_OUTPUT_ROOT": str(legacy_root),
            "VLA_POLICY": _required_env("VLA_CHECKPOINT"),
            "VLA_CHECKPOINT": _required_env("VLA_CHECKPOINT"),
            "VLA_CKPT": _required_env("VLA_CHECKPOINT"),
            "VLM_CKPT": _required_env("VLM_CHECKPOINT"),
            "VLM_CHECKPOINT": _required_env("VLM_CHECKPOINT"),
            "VLA_NORM_FILE": _required_env("NORM_STATS_PATH"),
            "VLA_NORM_SHA256": env.get("NORM_STATS_SHA256", ""),
            "NORM_STATS_PATH": _required_env("NORM_STATS_PATH"),
            "VLA_REPO_ID": str(norm_path.parent),
            "BDDL_PATH": _required_env("BDDL_PATH"),
            "MAX_STEPS": _required_env("MAX_STEPS"),
            "OFFICIAL_MAX_STEPS": _required_env("MAX_STEPS"),
            "REPLAN_STEPS": _required_env("REPLAN_STEPS"),
            "OFFICIAL_REPLAN_STEPS": _required_env("REPLAN_STEPS"),
            "ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR": str(official_scripts),
            "ROBOMEMARENA_OFFICIAL_BDDL_DIR": str(official_bddl),
            "ROBOMEMARENA_ROOT_BDDL_DIR": str(repo_root / "bddl"),
            "OFFICIAL_ROOT": str(official_runtime_root),
            "ROBOMEMARENA_REMOTE_ROOT": str(official_runtime_root),
            "SOURCE_ROOT": str(official_runtime_root),
            "OPENPI_ROOT": _required_env("OPENPI_ROOT"),
            "INFER_ROOT": _required_env("INFER_ROOT"),
            "OPENPI_INFERENCE_ROOT": _required_env("OPENPI_INFERENCE_ROOT"),
            "TARGET_LIBERO_PATH": _required_env("TARGET_LIBERO_PATH"),
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_official_runtime(repo_root: Path, runtime_root: Path) -> None:
    lock_path = repo_root / "evaluation_benchmark/sync_eval_26/upstream_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_commit = str(lock["commit"])
    if expected_commit != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"adapter scorer constant mismatch: "
            f"lock={expected_commit} adapter={OFFICIAL_COMMIT}"
        )
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=runtime_root,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"official runtime commit mismatch: "
            f"expected={expected_commit} actual={actual_commit}"
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=runtime_root,
        text=True,
    ).strip():
        raise RuntimeError(f"official runtime checkout is dirty: {runtime_root}")
    for relative, expected_hash in lock["files"].items():
        path = runtime_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"official runtime hash mismatch: {relative}: "
                f"expected={expected_hash} actual={actual_hash}"
            )


def _score_candidates(
    log_paths: list[Path],
    task_id: int,
    *,
    started_ns: int,
    summary_identities: dict[Path, SourceIdentity],
) -> list[ScoreCandidate]:
    candidates: list[ScoreCandidate] = []
    for path in sorted(log_paths):
        if not path.is_file() or path.stat().st_mtime_ns < started_ns:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        summary_identity = summary_identities.get(path.resolve())
        log_identities = _identities_from_text(text)
        if summary_identity is not None:
            if log_identities and log_identities != {summary_identity}:
                raise RuntimeError(
                    f"official summary/log identity mismatch for {path}: "
                    f"summary={summary_identity} log={sorted(log_identities, key=lambda item: (item.seed, item.episode))}"
                )
            identity = summary_identity
        else:
            identity = _identity_from_identities(log_identities)
            if identity is None:
                continue
        matches = list(OFFICIAL_SCORE_RE.finditer(text))
        if matches and identity is not None:
            match = matches[-1]
            candidates.append(
                ScoreCandidate(
                    log_path=path,
                    task_id=int(match.group("task")),
                    score=float(match.group("score")),
                    stage_success=bool(int(match.group("stage_success"))),
                    goal_success=bool(int(match.group("goal_success"))),
                    stage_done={
                        str(key): bool(value)
                        for key, value in json.loads(match.group("stage_done")).items()
                    },
                    source_seed=identity.seed,
                    source_episode=identity.episode,
                )
            )
            continue
        episode_matches = list(EPISODE_SCORE_RE.finditer(text))
        if not episode_matches:
            continue
        match = episode_matches[-1]
        stage_done = {
            token.group("name").strip(): token.group("done") == "Y"
            for token in STAGE_TOKEN_RE.finditer(match.group("stages"))
        }
        if stage_done:
            candidates.append(
                ScoreCandidate(
                    log_path=path,
                    task_id=task_id,
                    score=float(match.group("score")),
                    stage_success=bool(int(match.group("stage_success"))),
                    goal_success=match.group("goal").lower()
                    not in {"0", "0.0", "0.000", "false", "n"},
                    stage_done=stage_done,
                    source_seed=int(match.group("seed")),
                    source_episode=int(match.group("episode")),
                )
            )
    return candidates


def _identities_from_text(text: str) -> set[SourceIdentity]:
    return {
        SourceIdentity(
            seed=int(match.group("seed")),
            episode=int(match.group("episode")),
        )
        for match in EPISODE_ID_RE.finditer(text)
    }


def _identity_from_identities(
    identities: set[SourceIdentity],
) -> SourceIdentity | None:
    if len(identities) != 1:
        return None
    return next(iter(identities))


def _record_identity(
    identities: dict[Path, SourceIdentity],
    *,
    legacy_root: Path,
    log_raw: str,
    seed: int,
    episode: int,
) -> None:
    log_path = Path(log_raw)
    if not log_path.is_absolute():
        log_path = legacy_root / log_path
    log_path = log_path.resolve()
    if not log_path.is_relative_to(legacy_root):
        raise RuntimeError(f"official summary log is outside legacy root: {log_path}")
    identity = SourceIdentity(seed=seed, episode=episode)
    previous = identities.get(log_path)
    if previous is not None and previous != identity:
        raise RuntimeError(
            f"conflicting official summary identity for {log_path}: "
            f"{previous} != {identity}"
        )
    identities[log_path] = identity


def _summary_identities(
    legacy_root: Path,
    *,
    started_ns: int,
    expected_task_id: int,
) -> dict[Path, SourceIdentity]:
    identities: dict[Path, SourceIdentity] = {}
    for summary_path in sorted(legacy_root.rglob("official_summary.json")):
        if summary_path.stat().st_mtime_ns < started_ns:
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = payload.get("episodes", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"invalid official summary episodes: {summary_path}")
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid official summary row: {summary_path}")
            if int(row["task_id"]) != expected_task_id:
                raise RuntimeError(
                    f"official summary task mismatch: "
                    f"expected={expected_task_id} row={row['task_id']}"
                )
            _record_identity(
                identities,
                legacy_root=legacy_root,
                log_raw=str(row["log"]),
                seed=int(row["seed"]),
                episode=int(row["ep"]),
            )
    for tsv_path in sorted(legacy_root.rglob("official_episodes.tsv")):
        if tsv_path.stat().st_mtime_ns < started_ns:
            continue
        with tsv_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if int(row["task_id"]) != expected_task_id:
                    raise RuntimeError(
                        f"official TSV task mismatch: "
                        f"expected={expected_task_id} row={row['task_id']}"
                    )
                _record_identity(
                    identities,
                    legacy_root=legacy_root,
                    log_raw=str(row["log"]),
                    seed=int(row["seed"]),
                    episode=int(row["ep"]),
                )
    return identities


def _video_paths(output_dir: Path) -> list[str]:
    suffixes = {".mp4", ".webm", ".avi"}
    return [
        str(path.resolve())
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _prompt_trace(log_path: Path) -> str | None:
    if not log_path.is_file():
        return None
    trace_path = log_path.with_name("prompt_trace.log")
    lines = [
        line
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if "prompt=" in line
        or "[VLM" in line
        or "subtask" in line.lower()
        or "planner" in line.lower()
    ]
    if not lines:
        return None
    trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(trace_path.resolve())


def _write_result(
    video_root: Path,
    candidate: ScoreCandidate,
) -> None:
    task_id = int(_required_env("TASK_ID"))
    scored_task = candidate.task_id
    if scored_task != task_id:
        raise RuntimeError(
            f"official score task mismatch: requested={task_id} scored={scored_task}"
        )
    stage_done_raw: Any = candidate.stage_done
    if not isinstance(stage_done_raw, dict) or not stage_done_raw:
        raise RuntimeError("official scorer returned empty stage_done_json")
    stage_done = {str(key): bool(value) for key, value in stage_done_raw.items()}
    payload = {
        "task_id": task_id,
        "episode_index": int(_required_env("EPISODE_INDEX")),
        "seed": int(_required_env("SEED")),
        "scorer_commit": OFFICIAL_COMMIT,
        "stage_done": stage_done,
        "reported_stage_score_pct": candidate.score,
        "reported_stage_success": candidate.stage_success,
        "reported_goal_success": candidate.goal_success,
        "score_log": str(candidate.log_path.resolve()),
        "source_seed": candidate.source_seed,
        "source_episode": candidate.source_episode,
        "video_paths": _video_paths(video_root),
        "prompt_trace": _prompt_trace(candidate.log_path),
        "termination_reason": "legacy_rollout_completed",
    }
    result_path = Path(_required_env("OFFICIAL_RESULT_JSON"))
    result_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    output_dir = Path(_required_env("OUTPUT_DIR")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = _repo_root()
    env = _runtime_env(repo_root, output_dir)
    started_ns = time.time_ns() - 2_000_000_000
    command = _source_command()
    cwd = Path(os.environ.get("SOURCE_CWD", str(repo_root))).resolve()
    source_log = output_dir / "source_launcher.log"
    with source_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"source launcher failed with exit {completed.returncode}: "
            f"command={command} log={source_log}"
        )

    task_id = int(_required_env("TASK_ID"))
    requested_seed = int(_required_env("SEED"))
    legacy_root = Path(env["OUT_ROOT"])
    log_paths = [source_log, *sorted(legacy_root.rglob("*.log"))]
    candidates = _score_candidates(
        log_paths,
        task_id,
        started_ns=started_ns,
        summary_identities=_summary_identities(
            legacy_root,
            started_ns=started_ns,
            expected_task_id=task_id,
        ),
    )
    candidates = [
        item
        for item in candidates
        if item.source_seed == requested_seed and item.source_episode == 0
    ]
    if not candidates:
        raise RuntimeError(
            "source launcher completed without a locked official score; "
            f"searched={output_dir}"
        )
    candidate = max(
        candidates,
        key=lambda item: item.log_path.stat().st_mtime_ns,
    )
    _write_result(legacy_root, candidate)


if __name__ == "__main__":
    main()
