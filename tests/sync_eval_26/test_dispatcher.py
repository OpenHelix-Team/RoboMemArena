import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from evaluation_benchmark.sync_eval_26 import dispatcher

from .conftest import make_external_assets, write_asset_config


EXPECTED_COMMIT = "d9f83ac5182e25ad7f0a301a77a0b667f2392df1"


class _Bridge:
    @staticmethod
    def stage_names(_task_id: int) -> list[str]:
        return ["01_Place_Cookies_Basket", "02_Place_Tomato_Basket"]

    @staticmethod
    def stage_score_pct(_task_id: int, stage_done: dict[str, bool]) -> float:
        return 100.0 if all(stage_done.values()) else 0.0

    @staticmethod
    def stage_success(_task_id: int, stage_done: dict[str, bool]) -> bool:
        return all(stage_done.values())


def test_dispatcher_recomputes_official_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    assets = make_external_assets(tmp_path)
    asset_config = tmp_path / "assets.yaml"
    write_asset_config(asset_config, assets)
    monkeypatch.setenv("SYNC_EVAL_ASSET_CONFIG", str(asset_config))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        dispatcher,
        "verify_upstream_lock",
        lambda *_args, **_kwargs: SimpleNamespace(commit=EXPECTED_COMMIT),
    )
    monkeypatch.setattr(dispatcher, "OfficialScoringBridge", lambda *_args, **_kwargs: _Bridge())
    monkeypatch.setattr(
        dispatcher,
        "git_state",
        lambda _repo_root: {"head": "test", "branch": "test", "status_porcelain": ""},
    )

    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "out = Path(os.environ['OFFICIAL_RESULT_JSON'])",
                "out.write_text(json.dumps({",
                "  'task_id': int(os.environ['TASK_ID']),",
                "  'episode_index': int(os.environ['EPISODE_INDEX']),",
                "  'seed': int(os.environ['SEED']),",
                f"  'scorer_commit': '{EXPECTED_COMMIT}',",
                "  'stage_done': {",
                "    '01_Place_Cookies_Basket': True,",
                "    '02_Place_Tomato_Basket': True,",
                "  },",
                "  'video_paths': [],",
                "  'termination_reason': 'required_stages_done',",
                "}) + '\\n')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bddl = assets.official_runtime_root / "evaluation_benchmark" / "bddl" / "task1.bddl"
    bddl.parent.mkdir(parents=True)
    bddl.write_text("test\n", encoding="utf-8")
    profile_path = (
        tmp_path / "evaluation_benchmark" / "sync_eval_26" / "profiles" / "task01.yaml"
    )
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        yaml.safe_dump(
            {
                "task_id": 1,
                "task_name": "task1",
                "status": "frozen-success",
                "plugin_kind": "frozen-subprocess",
                "plugin_entrypoint": str(plugin),
                "bddl_path": str(bddl),
                "vla_checkpoint": str(assets.vla_checkpoint),
                "vlm_checkpoint": str(assets.vlm_checkpoint_for(1)),
                "norm_path": str(assets.norm_path),
                "norm_sha256": assets.norm_sha256,
                "seed": 104,
                "seed_mode": "increment",
                "replan_steps": 5,
                "max_steps": 1600,
                "runtime_topology": {
                    "allocation_gpus": 2,
                    "vla_visible_index": 0,
                    "vlm_visible_index": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    result = dispatcher.dispatch_episode(
        repo_root=tmp_path,
        profile_path=profile_path,
        episode_index=2,
        output_root=tmp_path / "output",
    )

    assert result.official.stage_score_pct == 100.0
    assert result.official.stage_success
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 106
    assert manifest["vla_checkpoint"] == str(assets.vla_checkpoint)
    assert manifest["scorer_commit"] == EXPECTED_COMMIT
    assert manifest["profile_sha256"]
    assert manifest["execution"]["unix_user"]
    assert manifest["explicit_runtime_env"]["SEED"] == "106"
    assert manifest["explicit_runtime_env"]["BDDL_PATH"] == str(bddl)
    assert manifest["runtime_topology"] == {
        "required_gpus": 2,
        "visible_devices": ["0", "1"],
        "vla_device": "0",
        "vlm_device": "1",
    }
