from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26 import scoring_bridge
from evaluation_benchmark.sync_eval_26.scoring_bridge import OfficialScoringBridge

from .conftest import make_external_assets


REPO_ROOT = Path(__file__).resolve().parents[2]


class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


class _StageModule:
    COUNTING_POUR_TASKS = {6, 7, 10, 15, 16, 22}

    @staticmethod
    def _task_specs(task_id: int) -> list[_Spec]:
        if task_id == 22:
            return [
                _Spec("01_Lift_Tomato_Sauce"),
                _Spec("02_Pour_One"),
                _Spec("03_Pour_Two"),
            ]
        if task_id in {20, 21, 23, 24}:
            return [_Spec("01_Open_Microwave"), _Spec("02_Place_Object")]
        return [_Spec("01_Action"), _Spec("02_Place_Object")]

    @classmethod
    def _stage_score_pct(cls, task_id: int, stage_done: dict[str, bool]) -> float:
        names = [spec.name for spec in cls._task_specs(task_id)]
        return 100.0 * sum(bool(stage_done.get(name)) for name in names) / len(names)

    @classmethod
    def _stage_success_from_stage_done(
        cls, task_id: int, stage_done: dict[str, bool]
    ) -> bool:
        return all(stage_done.get(spec.name, False) for spec in cls._task_specs(task_id))

    @staticmethod
    def _extra_pour_check(_task_id: int):
        return None


@pytest.fixture
def bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OfficialScoringBridge:
    assets = make_external_assets(tmp_path)
    monkeypatch.setattr(
        scoring_bridge,
        "_load_official_stage_module",
        lambda *_args, **_kwargs: _StageModule,
    )
    return OfficialScoringBridge(REPO_ROOT, assets=assets)


@pytest.mark.parametrize("task_id", range(1, 27))
def test_all_tasks_have_official_stage_names(
    bridge: OfficialScoringBridge, task_id: int
) -> None:
    assert bridge.stage_names(task_id)


def test_microwave_close_is_not_scored(bridge: OfficialScoringBridge) -> None:
    for task_id in (20, 21, 23, 24):
        assert all(
            "Close_Microwave" not in name for name in bridge.stage_names(task_id)
        )


def test_task22_exact_required_stages(bridge: OfficialScoringBridge) -> None:
    assert bridge.stage_names(22) == [
        "01_Lift_Tomato_Sauce",
        "02_Pour_One",
        "03_Pour_Two",
    ]


def test_optional_drawer_close_is_excluded_from_score(
    bridge: OfficialScoringBridge,
) -> None:
    stage_done = {name: True for name in bridge.stage_names(12)}
    stage_done["04_Close_Middle_Drawer"] = False

    assert bridge.stage_score_pct(12, stage_done) == 100.0
    assert bridge.stage_success(12, stage_done)


def test_counting_task_membership(bridge: OfficialScoringBridge) -> None:
    assert bridge.counting_pour_task(22)
    assert not bridge.counting_pour_task(20)
