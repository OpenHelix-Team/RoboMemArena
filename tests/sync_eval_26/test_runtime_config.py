import json
from pathlib import Path

from evaluation_benchmark.sync_eval_26.assets import ExternalAssets
from evaluation_benchmark.sync_eval_26.models import RuntimeTopology, TaskProfile
from evaluation_benchmark.sync_eval_26.runtime_config import materialize_runtime_env


def _assets(tmp_path: Path) -> ExternalAssets:
    vla = tmp_path / "assets" / "vla"
    norm = vla / "norm_stats.json"
    vlm = tmp_path / "assets" / "vlm" / "task01"
    official = tmp_path / "assets" / "official"
    openpi = tmp_path / "assets" / "openpi"
    inference = tmp_path / "assets" / "openpi_inference"
    data = tmp_path / "assets" / "fullvlm_v2"
    for path in (vla, vlm, official, openpi, inference, data):
        path.mkdir(parents=True, exist_ok=True)
    norm.write_text("norm\n", encoding="utf-8")
    return ExternalAssets(
        vla_checkpoint=vla,
        norm_path=norm,
        norm_sha256="a" * 64,
        vlm_checkpoints={1: vlm},
        official_runtime_root=official,
        openpi_root=openpi,
        openpi_inference_root=inference,
        fullvlm_data_root=data,
        h5dump_bin=None,
    )


def test_runtime_overlay_expands_assets_and_rewrites_launch_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = (
        repo
        / "evaluation_benchmark"
        / "sync_eval_26"
        / "rollout_plugins"
        / "task01"
        / "run.sh"
    )
    config = source.parent / "anchors.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        "OPENPI=${ASSET_OPENPI_ROOT}\nDATA=${ASSET_FULLVLM_DATA_ROOT}\n",
        encoding="utf-8",
    )
    config.write_text(
        '{"anchor":"${ASSET_FULLVLM_DATA_ROOT}/task1/example.hdf5"}\n',
        encoding="utf-8",
    )
    adapter = repo / "evaluation_benchmark" / "sync_eval_26" / "legacy_adapter.py"
    adapter.write_text("# plugin\n", encoding="utf-8")
    assets = _assets(tmp_path)
    profile = TaskProfile(
        task_id=1,
        task_name="task1",
        status="frozen-success",
        plugin_kind="frozen-subprocess",
        plugin_entrypoint=adapter,
        bddl_path=assets.official_runtime_root / "bddl" / "task1.bddl",
        vla_checkpoint=assets.vla_checkpoint,
        vlm_checkpoint=assets.vlm_checkpoint_for(1),
        norm_path=assets.norm_path,
        norm_sha256=assets.norm_sha256,
        seed=104,
        seed_mode="increment",
        replan_steps=5,
        max_steps=2000,
        runtime_topology=RuntimeTopology(2, 0, 1),
        runtime_env={
            "SOURCE_COMMAND_JSON": json.dumps(["bash", str(source)]),
            "SOURCE_CWD": str(source.parent),
            "ANCHORS_JSON": str(config),
        },
        source_paths=(adapter, source, config),
    )

    env = materialize_runtime_env(
        profile=profile,
        assets=assets,
        output_dir=tmp_path / "output",
        repo_root=repo,
    )

    command = json.loads(env["SOURCE_COMMAND_JSON"])
    overlay_source = Path(command[-1])
    overlay_config = Path(env["ANCHORS_JSON"])
    assert overlay_source != source
    assert overlay_source.is_file()
    assert overlay_config.is_file()
    assert str(assets.openpi_root) in overlay_source.read_text(encoding="utf-8")
    assert str(assets.fullvlm_data_root) in overlay_source.read_text(encoding="utf-8")
    assert str(assets.fullvlm_data_root) in overlay_config.read_text(encoding="utf-8")
    assert "${ASSET_" not in overlay_source.read_text(encoding="utf-8")
    assert "${ASSET_" not in overlay_config.read_text(encoding="utf-8")
    assert "${ASSET_" in source.read_text(encoding="utf-8")
