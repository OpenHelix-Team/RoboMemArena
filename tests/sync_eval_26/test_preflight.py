from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.preflight import (
    _verify_no_embedded_official_source,
)


def test_rejects_embedded_stale_official_checkout(tmp_path: Path) -> None:
    source = tmp_path / "rollout.py"
    source.write_text(
        'ROOT = PACK_DIR / "source" / "RoboMemArena_d9f83ac"\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="embeds stale official checkout"):
        _verify_no_embedded_official_source([source])


def test_accepts_source_root_runtime_binding(tmp_path: Path) -> None:
    source = tmp_path / "rollout.py"
    source.write_text(
        'ROOT = Path(os.environ["SOURCE_ROOT"]).resolve()\n',
        encoding="utf-8",
    )

    _verify_no_embedded_official_source([source])
