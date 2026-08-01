import importlib.util
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import pytest

from .conftest import make_external_assets, write_asset_config


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "evaluation_benchmark" / "sync_eval_26" / "scripts"
SCHEDULER_PATH = SCRIPTS_DIR / "schedule_exact20.py"
SLURM_EXACT20_LAUNCHER = SCRIPTS_DIR / "launch_exact20_dual_gpu_slurm.sh"
INNER_EXACT20_RUNNER = SCRIPTS_DIR / "run_exact20_dual_gpu.sh"
SPEC = importlib.util.spec_from_file_location("schedule_exact20", SCHEDULER_PATH)
assert SPEC is not None and SPEC.loader is not None
SCHEDULER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCHEDULER
SPEC.loader.exec_module(SCHEDULER)


def test_parse_tasks_supports_ranges() -> None:
    assert SCHEDULER.parse_tasks("1-3,7,20-21") == [1, 2, 3, 7, 20, 21]


def test_queue_is_task_major_and_exact() -> None:
    queue = SCHEDULER.build_work_queue([2, 3], [0, 1])

    assert isinstance(queue, deque)
    assert [(item.task_id, item.episode_index) for item in queue] == [
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ]


def test_discover_slots_requires_visible_gpu() -> None:
    with pytest.raises(RuntimeError, match="no GPU slots"):
        SCHEDULER.discover_slots("", allocated_raw="0")


def test_discover_slots_preserves_explicit_slot_ids() -> None:
    assert SCHEDULER.discover_slots(
        "2,5,7",
        allocated_raw="2,5,7",
    ) == ["2", "5", "7"]


def test_discover_slots_rejects_slots_outside_allocation() -> None:
    with pytest.raises(RuntimeError, match="outside CUDA_VISIBLE_DEVICES"):
        SCHEDULER.discover_slots("2,7", allocated_raw="2,5")


def test_discover_gpu_bundles_preserves_atomic_groups() -> None:
    assert SCHEDULER.discover_gpu_bundles(
        "0,1;2,3", allocated_raw="0,1,2,3"
    ) == [("0", "1"), ("2", "3")]


def test_assert_task_capacity_rejects_single_gpu_task1_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = make_external_assets(tmp_path)
    config = tmp_path / "assets.yaml"
    write_asset_config(config, assets)
    monkeypatch.setenv("SYNC_EVAL_ASSET_CONFIG", str(config))
    with pytest.raises(RuntimeError, match="no compatible GPU bundle for task 1"):
        SCHEDULER.assert_task_capacity(
            task_ids=[1],
            bundles=[("0",)],
            profile_dir=REPO_ROOT / "evaluation_benchmark" / "sync_eval_26" / "profiles",
        )


def test_package_scheduler_uses_the_packaged_profiles() -> None:
    content = SCHEDULER_PATH.read_text(encoding="utf-8")

    assert '"evaluation_benchmark" / "sync_eval_26" / "profiles"' in content


def test_strict_slurm_launcher_disables_cpu_affinity() -> None:
    content = SLURM_EXACT20_LAUNCHER.read_text(encoding="utf-8")

    assert "--gres=gpu:2" in content
    assert "--cpu-bind=none" in content
    assert "CPU_AFFINITY_MODE=none" in content
    assert "run_exact20_dual_gpu.sh" in content


def test_inner_runner_requires_verified_cpu_affinity_mode() -> None:
    content = INNER_EXACT20_RUNNER.read_text(encoding="utf-8")

    assert "CPU_AFFINITY_MODE" in content
    assert "must be launched with --cpu-bind=none" in content


def test_episode_claim_is_exclusive(tmp_path: Path) -> None:
    item = SCHEDULER.WorkItem(task_id=7, episode_index=3)
    first = SCHEDULER._claim_episode(tmp_path, item)
    assert first is not None
    try:
        assert SCHEDULER._claim_episode(tmp_path, item) is None
    finally:
        SCHEDULER._release_lock(first)

    second = SCHEDULER._claim_episode(tmp_path, item)
    assert second is not None
    SCHEDULER._release_lock(second)


def test_episode_claim_survives_parent_fd_close_while_child_runs(
    tmp_path: Path,
) -> None:
    item = SCHEDULER.WorkItem(task_id=7, episode_index=4)
    first = SCHEDULER._claim_episode(tmp_path, item)
    assert first is not None
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        pass_fds=(first.fileno(),),
    )
    first.close()
    try:
        assert SCHEDULER._claim_episode(tmp_path, item) is None
    finally:
        child.wait(timeout=5)
    time.sleep(0.05)
    second = SCHEDULER._claim_episode(tmp_path, item)
    assert second is not None
    SCHEDULER._release_lock(second)


def test_slot_claim_survives_parent_fd_close_while_child_runs(
    tmp_path: Path,
) -> None:
    first = SCHEDULER._claim_slot(
        "0",
        lock_root=tmp_path,
        job_id="123",
        hostname="node-a",
    )
    assert first is not None
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        pass_fds=(first.fileno(),),
    )
    first.close()
    try:
        assert (
            SCHEDULER._claim_slot(
                "0",
                lock_root=tmp_path,
                job_id="123",
                hostname="node-a",
            )
            is None
        )
    finally:
        child.wait(timeout=5)
    time.sleep(0.05)
    second = SCHEDULER._claim_slot(
        "0",
        lock_root=tmp_path,
        job_id="123",
        hostname="node-a",
    )
    assert second is not None
    SCHEDULER._release_lock(second)


def test_slot_lock_namespace_isolated_between_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNC_EVAL_SLOT_LOCK_NAMESPACE", "439272-worker1")

    assert SCHEDULER.resolve_slot_lock_namespace(None) == "439272-worker1"
    assert SCHEDULER.resolve_slot_lock_namespace("explicit-worker") == "explicit-worker"


def test_slot_namespace_cannot_bypass_physical_bundle_lock(tmp_path: Path) -> None:
    first = SCHEDULER._claim_slot(
        "0,1",
        lock_root=tmp_path,
        job_id="440039",
        hostname="node-a",
        namespace="task6",
    )
    assert first is not None
    try:
        assert (
            SCHEDULER._claim_slot(
                "0,1",
                lock_root=tmp_path,
                job_id="440039",
                hostname="node-a",
                namespace="task23",
            )
            is None
        )
    finally:
        SCHEDULER._release_lock(first)


def test_slot_lock_separates_disjoint_slurm_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLURM_STEP_ID", "451075.32")
    first = SCHEDULER._claim_slot(
        "0,1", lock_root=tmp_path, job_id="451075", hostname="node-a"
    )
    assert first is not None
    try:
        monkeypatch.setenv("SLURM_STEP_ID", "451075.33")
        second = SCHEDULER._claim_slot(
            "0,1", lock_root=tmp_path, job_id="451075", hostname="node-a"
        )
        assert second is not None
    finally:
        SCHEDULER._release_lock(first)
        if "second" in locals():
            SCHEDULER._release_lock(second)


def test_port_reservation_is_exclusive() -> None:
    port, first = SCHEDULER._reserve_port(39571)
    try:
        other_port, second = SCHEDULER._reserve_port(port)
        try:
            assert other_port != port
        finally:
            SCHEDULER._release_lock(second)
    finally:
        SCHEDULER._release_lock(first)
