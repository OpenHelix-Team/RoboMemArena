from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .models import TaskProfile


@dataclass(frozen=True)
class RolloutRequest:
    repo_root: Path
    task_id: int
    episode_index: int
    seed: int
    output_dir: Path
    profile: TaskProfile
    scorer_lock_path: Path
    runtime_env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutResult:
    return_code: int
    episode_log: Path
    official_result_path: Path
    video_paths: tuple[Path, ...]
    prompt_trace: Path | None
    termination_reason: str


class RolloutPlugin(Protocol):
    def run(self, request: RolloutRequest) -> RolloutResult: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_plugin_sources(source_hashes: dict[str, str]) -> int:
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(f"plugin source path must be absolute: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"source hash mismatch: {path}: expected={expected} actual={actual}"
            )
    return len(source_hashes)
