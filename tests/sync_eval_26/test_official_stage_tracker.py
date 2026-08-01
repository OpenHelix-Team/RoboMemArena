import sys
from pathlib import Path


EVALUATOR_DIR = (
    Path(__file__).resolve().parents[2]
    / "evaluation_benchmark"
    / "sync_eval_26"
    / "rollout_plugins"
    / "archived_success_pack"
    / "evaluators"
)
sys.path.insert(0, str(EVALUATOR_DIR))

from official_stage_tracker import OrderedOfficialStageTracker  # noqa: E402


class _Spec:
    def __init__(self, name: str, expected_step: int):
        self.name = name
        self.expected_step = expected_step

    def check_fn(self, _env, state, stage_start: int) -> bool:
        return state["step"] == self.expected_step and stage_start < state["step"]


class _StageModule:
    @staticmethod
    def _task_specs(_task_id: int):
        return [_Spec("01_A", 1), _Spec("02_B", 2)]

    @staticmethod
    def _build_initial_state(_env):
        return {"step": 0, "step_idx": 0}

    @staticmethod
    def _update_state(obs, state):
        state["step"] = obs
        state["step_idx"] = obs


def test_tracker_uses_official_names_and_order() -> None:
    tracker = OrderedOfficialStageTracker(_StageModule, task_id=1, env=object())

    assert tracker.done == {"01_A": False, "02_B": False}
    assert tracker.update(env=object(), obs=2) == []
    assert tracker.update(env=object(), obs=1) == ["01_A"]
    assert tracker.update(env=object(), obs=2) == ["02_B"]
    assert tracker.done == {"01_A": True, "02_B": True}
