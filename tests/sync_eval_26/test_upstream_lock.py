import hashlib
import json
from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.upstream_lock import verify_upstream_lock


def _write_lock(root: Path, source: Path) -> Path:
    lock_path = root / "evaluation_benchmark" / "sync_eval_26" / "upstream_lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps(
            {
                "commit": "d9f83ac5182e25ad7f0a301a77a0b667f2392df1",
                "files": {"locked.py": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
        ),
        encoding="utf-8",
    )
    return lock_path


def test_upstream_lock_accepts_a_matching_explicit_runtime_file(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source = runtime / "locked.py"
    source.write_text("locked\n", encoding="utf-8")
    lock_path = _write_lock(tmp_path, source)

    result = verify_upstream_lock(
        tmp_path,
        lock_path=lock_path,
        runtime_root=runtime,
        verify_commit=False,
    )

    assert result.commit == "d9f83ac5182e25ad7f0a301a77a0b667f2392df1"
    assert result.checked_files == 1


def test_upstream_lock_rejects_modified_file(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    source = runtime / "locked.py"
    source.write_text("locked\n", encoding="utf-8")
    lock_path = _write_lock(tmp_path, source)
    source.write_text("modified\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="upstream hash mismatch"):
        verify_upstream_lock(
            tmp_path,
            lock_path=lock_path,
            runtime_root=runtime,
            verify_commit=False,
        )
