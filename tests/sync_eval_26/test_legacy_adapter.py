import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation_benchmark.sync_eval_26.legacy_adapter import (
    SourceIdentity,
    _runtime_env,
    _score_candidates,
    _summary_identities,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = REPO_ROOT / "evaluation_benchmark/sync_eval_26/legacy_adapter.py"
LOCK = REPO_ROOT / "evaluation_benchmark/sync_eval_26/upstream_lock.json"
OFFICIAL_RUNTIME_ROOT = REPO_ROOT.parent / "RoboMemArena_official_d9f83ac"


@pytest.fixture(scope="module", autouse=True)
def clean_official_runtime(tmp_path_factory: pytest.TempPathFactory):
    global OFFICIAL_RUNTIME_ROOT

    source_root = OFFICIAL_RUNTIME_ROOT
    worktree = tmp_path_factory.mktemp("official-runtime") / "d9f83ac"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        OFFICIAL_RUNTIME_ROOT = worktree
        yield
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        OFFICIAL_RUNTIME_ROOT = source_root


def test_adapter_forces_one_episode_and_extracts_official_score(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fake_source.py"
    source.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "assert os.environ['NUM_TRIALS'] == '1'",
                "assert os.environ['NUM_TRIALS_PER_TASK'] == '1'",
                "assert os.environ['PORT'] == '15103'",
                "assert os.environ['ROBOMEMARENA_OFFICIAL_SCRIPTS_DIR'].endswith("
                "'evaluation_benchmark/scripts')",
                "root = Path(os.environ['OUT_ROOT'])",
                "log = root / 'task1/ep0/sync_vlm.log'",
                "log.parent.mkdir(parents=True, exist_ok=True)",
                "log.write_text('[OFFICIAL_SCORE] task=1 average_score_pct=100.000000 '"
                "'stage_success=1 goal_success=1 stage_done_json='"
                '\'{"01_Place_Cookies_Basket":true,"02_Place_Tomato_Basket":true}\\n\')',
                "summary = {'episodes': [{'task_id': 1, 'ep': 0, "
                "'seed': int(os.environ['SEED']), 'log': str(log)}]}",
                "(root / 'official_summary.json').write_text("
                "__import__('json').dumps(summary))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = output / "official_episode.json"
    env = os.environ.copy()
    env.update(
        {
            "TASK_ID": "1",
            "EPISODE_INDEX": "3",
            "SEED": "107",
            "OUTPUT_DIR": str(output),
            "OFFICIAL_RESULT_JSON": str(result),
            "OFFICIAL_SCORER_LOCK": str(LOCK),
            "VLA_CHECKPOINT": "/tmp/vla",
            "VLM_CHECKPOINT": "/tmp/vlm",
            "NORM_STATS_PATH": "/tmp/norm.json",
            "BDDL_PATH": "/tmp/task1.bddl",
            "MAX_STEPS": "100",
            "REPLAN_STEPS": "5",
            "SOURCE_COMMAND_JSON": json.dumps([sys.executable, str(source)]),
            "OFFICIAL_RUNTIME_ROOT": str(OFFICIAL_RUNTIME_ROOT),
            "OPENPI_ROOT": str(tmp_path / "openpi"),
            "INFER_ROOT": str(tmp_path / "openpi_inference"),
            "OPENPI_INFERENCE_ROOT": str(tmp_path / "openpi_inference"),
            "TARGET_LIBERO_PATH": str(tmp_path / "libero"),
        }
    )
    completed = subprocess.run(
        [sys.executable, str(ADAPTER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["task_id"] == 1
    assert payload["episode_index"] == 3
    assert payload["seed"] == 107
    assert payload["source_seed"] == 107
    assert payload["source_episode"] == 0
    assert payload["scorer_commit"] == "d9f83ac5182e25ad7f0a301a77a0b667f2392df1"
    assert list(payload["stage_done"]) == [
        "01_Place_Cookies_Basket",
        "02_Place_Tomato_Basket",
    ]


def test_adapter_extracts_native_episode_stage_line(tmp_path: Path) -> None:
    source = tmp_path / "fake_native.py"
    source.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "root = Path(os.environ['OUT_ROOT'])",
                "log = root / 'task7/ep0/sync_vlm.log'",
                "log.parent.mkdir(parents=True, exist_ok=True)",
                "log.write_text('Episode 0 seed=100 stage_score=66.7 '"
                "'stage_success=1 goal=0.667 failure_reason=none | '"
                "'01_Lift_Tomato_Sauce=Y | 02_Pour_One=Y | 03_Pour_Two=N\\n')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    result = output / "official_episode.json"
    env = os.environ.copy()
    env.update(
        {
            "TASK_ID": "7",
            "EPISODE_INDEX": "0",
            "SEED": "100",
            "OUTPUT_DIR": str(output),
            "OFFICIAL_RESULT_JSON": str(result),
            "OFFICIAL_SCORER_LOCK": str(LOCK),
            "VLA_CHECKPOINT": "/tmp/vla",
            "VLM_CHECKPOINT": "/tmp/vlm",
            "NORM_STATS_PATH": "/tmp/norm.json",
            "BDDL_PATH": "/tmp/task7.bddl",
            "MAX_STEPS": "100",
            "REPLAN_STEPS": "1",
            "SOURCE_COMMAND_JSON": json.dumps([sys.executable, str(source)]),
            "OFFICIAL_RUNTIME_ROOT": str(OFFICIAL_RUNTIME_ROOT),
            "OPENPI_ROOT": str(tmp_path / "openpi"),
            "INFER_ROOT": str(tmp_path / "openpi_inference"),
            "OPENPI_INFERENCE_ROOT": str(tmp_path / "openpi_inference"),
            "TARGET_LIBERO_PATH": str(tmp_path / "libero"),
        }
    )

    completed = subprocess.run(
        [sys.executable, str(ADAPTER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["stage_done"] == {
        "01_Lift_Tomato_Sauce": True,
        "02_Pour_One": True,
        "03_Pour_Two": False,
    }
    assert payload["source_seed"] == 100


def test_adapter_rejects_official_score_without_seed_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fake_source.py"
    source.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "root = Path(os.environ['OUT_ROOT'])",
                "log = root / 'task1/ep0/sync_vlm.log'",
                "log.parent.mkdir(parents=True, exist_ok=True)",
                "log.write_text('[OFFICIAL_SCORE] task=1 "
                "average_score_pct=100.000000 stage_success=1 "
                "goal_success=1 stage_done_json='"
                '\'{"01_Place_Cookies_Basket":true,'
                '"02_Place_Tomato_Basket":true}\\n\')',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    env = os.environ.copy()
    env.update(
        {
            "TASK_ID": "1",
            "EPISODE_INDEX": "0",
            "SEED": "104",
            "OUTPUT_DIR": str(output),
            "OFFICIAL_RESULT_JSON": str(output / "official_episode.json"),
            "OFFICIAL_SCORER_LOCK": str(LOCK),
            "VLA_CHECKPOINT": "/tmp/vla",
            "VLM_CHECKPOINT": "/tmp/vlm",
            "NORM_STATS_PATH": "/tmp/norm.json",
            "BDDL_PATH": "/tmp/task1.bddl",
            "MAX_STEPS": "100",
            "REPLAN_STEPS": "5",
            "SOURCE_COMMAND_JSON": json.dumps([sys.executable, str(source)]),
            "OFFICIAL_RUNTIME_ROOT": str(OFFICIAL_RUNTIME_ROOT),
            "OPENPI_ROOT": str(tmp_path / "openpi"),
            "INFER_ROOT": str(tmp_path / "openpi_inference"),
            "OPENPI_INFERENCE_ROOT": str(tmp_path / "openpi_inference"),
            "TARGET_LIBERO_PATH": str(tmp_path / "libero"),
        }
    )

    completed = subprocess.run(
        [sys.executable, str(ADAPTER)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "without a locked official score" in completed.stderr


def test_runtime_env_preserves_scheduler_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = {
        "TASK_ID": "7",
        "EPISODE_INDEX": "3",
        "SEED": "100",
        "VLA_CHECKPOINT": "/tmp/vla",
        "VLM_CHECKPOINT": "/tmp/vlm",
        "NORM_STATS_PATH": "/tmp/norm/norm_stats.json",
        "BDDL_PATH": "/tmp/task.bddl",
        "MAX_STEPS": "2500",
        "REPLAN_STEPS": "5",
        "PORT": "24003",
        "OFFICIAL_RUNTIME_ROOT": str(OFFICIAL_RUNTIME_ROOT),
        "OPENPI_ROOT": str(tmp_path / "openpi"),
        "INFER_ROOT": str(tmp_path / "openpi_inference"),
        "OPENPI_INFERENCE_ROOT": str(tmp_path / "openpi_inference"),
        "TARGET_LIBERO_PATH": str(tmp_path / "libero"),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    runtime = _runtime_env(REPO_ROOT, tmp_path)

    assert runtime["PORT"] == "24003"


def test_runtime_env_forwards_profile_norm_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    values = {
        "TASK_ID": "20",
        "EPISODE_INDEX": "0",
        "SEED": "106",
        "VLA_CHECKPOINT": "/tmp/vla",
        "VLM_CHECKPOINT": "/tmp/vlm",
        "NORM_STATS_PATH": "/tmp/norm/norm_stats.json",
        "NORM_STATS_SHA256": "profile-locked-checksum",
        "BDDL_PATH": "/tmp/task.bddl",
        "MAX_STEPS": "1000",
        "REPLAN_STEPS": "10",
        "OFFICIAL_RUNTIME_ROOT": str(OFFICIAL_RUNTIME_ROOT),
        "OPENPI_ROOT": str(tmp_path / "openpi"),
        "INFER_ROOT": str(tmp_path / "openpi_inference"),
        "OPENPI_INFERENCE_ROOT": str(tmp_path / "openpi_inference"),
        "TARGET_LIBERO_PATH": str(tmp_path / "libero"),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    runtime = _runtime_env(REPO_ROOT, tmp_path)

    assert runtime["VLA_NORM_FILE"] == "/tmp/norm/norm_stats.json"
    assert runtime["VLA_NORM_SHA256"] == "profile-locked-checksum"


def test_score_candidates_rejects_summary_log_identity_mismatch(
    tmp_path: Path,
) -> None:
    log = tmp_path / "sync_vlm.log"
    log.write_text(
        "Episode 0 seed=999\n"
        "[OFFICIAL_SCORE] task=1 average_score_pct=100.000000 "
        "stage_success=1 goal_success=1 "
        'stage_done_json={"01_A":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="summary/log identity mismatch"):
        _score_candidates(
            [log],
            1,
            started_ns=0,
            summary_identities={log.resolve(): SourceIdentity(seed=104, episode=0)},
        )


def test_score_candidates_rejects_native_log_with_multiple_identities(
    tmp_path: Path,
) -> None:
    log = tmp_path / "sync_vlm.log"
    log.write_text(
        "Episode 1 seed=999 stage_score=0.0 stage_success=0 goal=0 "
        "failure_reason=none | 01_A=N\n"
        "Episode 0 seed=104 stage_score=100.0 stage_success=1 goal=1 "
        "failure_reason=none | 01_A=Y\n",
        encoding="utf-8",
    )

    candidates = _score_candidates(
        [log],
        1,
        started_ns=0,
        summary_identities={},
    )

    assert candidates == []


def test_summary_identities_requires_matching_task_and_explicit_episode(
    tmp_path: Path,
) -> None:
    log = tmp_path / "sync_vlm.log"
    log.write_text("score\n", encoding="utf-8")
    summary = tmp_path / "official_summary.json"
    summary.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "task_id": 2,
                        "seed": 104,
                        "log": str(log),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="task mismatch"):
        _summary_identities(
            tmp_path,
            started_ns=0,
            expected_task_id=1,
        )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["episodes"][0]["task_id"] = 1
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(KeyError, match="ep"):
        _summary_identities(
            tmp_path,
            started_ns=0,
            expected_task_id=1,
        )
