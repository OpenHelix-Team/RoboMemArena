import dataclasses
from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.frozen_subprocess import _assert_autonomy
from evaluation_benchmark.sync_eval_26.plugin_api import (
    RolloutResult,
    verify_plugin_sources,
)


def test_plugin_result_cannot_supply_final_metrics() -> None:
    fields = {field.name for field in dataclasses.fields(RolloutResult)}
    assert "csr" not in fields
    assert "tsr" not in fields
    assert "stage_done" not in fields


def test_frozen_plugin_rejects_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "evaluator.py"
    source.write_text("print('changed')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        verify_plugin_sources({str(source): "0" * 64})


def test_autonomous_profile_rejects_oracle_flag() -> None:
    with pytest.raises(RuntimeError, match="ORACLE_FORCE_INITIAL_PROMPT"):
        _assert_autonomy("frozen-success", {"ORACLE_FORCE_INITIAL_PROMPT": "1"})


def test_experimental_profile_allows_diagnostic_flag() -> None:
    _assert_autonomy("experimental", {"ORACLE_FORCE_INITIAL_PROMPT": "1"})
