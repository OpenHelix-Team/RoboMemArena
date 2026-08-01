from pathlib import Path

from evaluation_benchmark.sync_eval_26.profile_loader import (
    load_profile,
    verify_profile_assets,
)

from .conftest import make_external_assets, materialize_profile_bddls


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "evaluation_benchmark" / "sync_eval_26" / "profiles"


def _profiles(tmp_path: Path):
    assets = make_external_assets(tmp_path)
    materialize_profile_bddls(PROFILE_DIR, assets)
    return assets, {
        task_id: load_profile(PROFILE_DIR / f"task{task_id:02d}.yaml", assets=assets)
        for task_id in range(1, 27)
    }


def test_exactly_26_profiles_are_frozen() -> None:
    paths = sorted(PROFILE_DIR.glob("task[0-9][0-9].yaml"))
    assert [path.name for path in paths] == [
        f"task{task_id:02d}.yaml" for task_id in range(1, 27)
    ]


def test_all_profiles_bind_task_specific_external_assets(tmp_path: Path) -> None:
    assets, profiles = _profiles(tmp_path)
    for task_id, profile in profiles.items():
        assert profile.task_id == task_id
        assert profile.vla_checkpoint == assets.vla_checkpoint
        assert profile.norm_path == assets.norm_path
        assert profile.norm_sha256 == assets.norm_sha256
        assert profile.vlm_checkpoint == assets.vlm_checkpoint_for(task_id)
        assert profile.bddl_path.is_relative_to(assets.official_runtime_root)
        assert profile.prompt_config["source"] == "vlm"
        assert profile.prompt_config["oracle_prompt_injection"] is False


def test_all_profiles_keep_the_two_gpu_vla_vlm_topology(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    same_visible_device = {6, 10, 16}
    for task_id, profile in profiles.items():
        assert profile.runtime_topology.allocation_gpus == 2
        assert profile.runtime_topology.vla_visible_index == 0
        assert profile.runtime_topology.vlm_visible_index == (
            0 if task_id in same_visible_device else 1
        )


def test_task7_profile_forces_loopback_policy_server(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    profile = profiles[7]
    assert profile.runtime_env["HOST"] == "127.0.0.1"
    assert profile.runtime_topology.vla_visible_index == 0
    assert profile.runtime_topology.vlm_visible_index == 1


def test_task5_preserves_the_historical_replan10_control(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    assert profiles[5].replan_steps == 10


def test_task18_preserves_the_historical_pick_gate_control(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    profile = profiles[18]
    assert profile.runtime_env["ENDPOSE_PICK_GRIPPER_GATE"] == "0"
    assert profile.runtime_env["ENDPOSE_PICK_OBJECT_LIFT_GATE"] == "0"


def test_all_profile_paths_and_hashes_verify(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    for profile in profiles.values():
        verification = verify_profile_assets(profile)
        assert verification.task_id == profile.task_id
        assert verification.checked_hashes >= 5


def test_only_tasks22_and26_are_experimental(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    assert {
        task_id for task_id, profile in profiles.items() if profile.status == "experimental"
    } == {22, 26}


def test_task22_declares_the_vla_config_required_by_its_launcher(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    assert (
        profiles[22].runtime_env["VLA_CONFIG"]
        == "pi05_libero_robomemarena_fullvlm_v2_noflip_dataset"
    )


def test_fixed_seed_profiles_are_explicit(tmp_path: Path) -> None:
    _, profiles = _profiles(tmp_path)
    expected = {
        6: 100,
        7: 100,
        20: 106,
        21: 107,
        22: 104,
        23: 105,
        24: 108,
        26: 106,
    }
    actual = {
        task_id: profile.seed
        for task_id, profile in profiles.items()
        if profile.seed_mode == "fixed"
    }
    assert actual == expected
