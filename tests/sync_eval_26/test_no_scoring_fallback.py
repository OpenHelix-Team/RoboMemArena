from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.scoring_bridge import OfficialScoringBridge

from .conftest import make_external_assets


@pytest.mark.parametrize(
    "path",
    [
        Path(
            "evaluation_benchmark/sync_eval_26/rollout_plugins/"
            "archived_success_pack/evaluators/"
            "eval_task1_qwen3_sync_endpose_hold_officialscore.py"
        ),
        Path(
            "evaluation_benchmark/sync_eval_26/rollout_plugins/"
            "archived_success_pack/evaluators/"
            "eval_tasks2_26_sync_endpose_hold_officialscore.py"
        ),
    ],
)
def test_archived_evaluators_require_the_injected_official_runtime(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    assert "ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR" in text
    assert "official_remote_" + "66e7894" not in text


def test_missing_official_lock_fails_without_fallback(tmp_path: Path) -> None:
    assets = make_external_assets(tmp_path)
    with pytest.raises(FileNotFoundError):
        OfficialScoringBridge(tmp_path, assets=assets)
