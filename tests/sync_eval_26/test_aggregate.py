from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.aggregate import EpisodeRecord, aggregate_task


def episode(index: int, success: bool = True) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=1,
        episode_index=index,
        seed=100 + index,
        stage_done={"a": success},
        stage_score_pct=100.0 if success else 0.0,
        stage_success=success,
        scorer_commit="d9f83ac",
        official_result_path=Path(f"/tmp/ep{index}.json"),
    )


def test_duplicate_episode_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_task([episode(0), episode(0)], expected_episodes=2)


def test_exact20_requires_twenty_valid_episodes() -> None:
    with pytest.raises(ValueError, match="expected 20"):
        aggregate_task([episode(i) for i in range(19)], expected_episodes=20)


def test_aggregate_computes_csr_and_tsr() -> None:
    result = aggregate_task([episode(0, True), episode(1, False)], expected_episodes=2)

    assert result.csr_pct == 50.0
    assert result.tsr == 0.5
    assert result.successes == 1
