from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


LOCK_RELATIVE_PATH = Path("evaluation_benchmark/sync_eval_26/upstream_lock.json")


@dataclass(frozen=True)
class LockVerification:
    commit: str
    checked_files: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream_lock(
    repo_root: Path,
    *,
    lock_path: Path | None = None,
    runtime_root: Path | None = None,
    file_overrides: Mapping[str, Path] | None = None,
    verify_commit: bool = True,
) -> LockVerification:
    repo_root = repo_root.resolve()
    lock_path = lock_path or repo_root / LOCK_RELATIVE_PATH
    runtime_root = (runtime_root or repo_root).resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    commit = str(lock["commit"])

    if verify_commit:
        try:
            actual = subprocess.check_output(
                ["git", "rev-parse", commit],
                cwd=runtime_root,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"missing upstream commit {commit}") from exc
        if actual != commit:
            raise RuntimeError(
                f"upstream commit mismatch: expected={commit} actual={actual}"
            )

    overrides = dict(file_overrides or {})
    for relative, expected in lock["files"].items():
        path = overrides.get(relative, runtime_root / relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_hash = _sha256(path)
        if actual_hash != expected:
            raise RuntimeError(
                f"upstream hash mismatch: {relative}: "
                f"expected={expected} actual={actual_hash}"
            )

    return LockVerification(commit=commit, checked_files=len(lock["files"]))
