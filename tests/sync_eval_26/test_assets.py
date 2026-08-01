from pathlib import Path

import yaml

from evaluation_benchmark.sync_eval_26.assets import load_external_assets


def test_external_assets_resolve_a_distinct_vlm_for_each_task(tmp_path: Path) -> None:
    config_path = tmp_path / "assets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vla_checkpoint": "/models/vla_35999",
                "norm_path": "/models/vla_35999/assets/norm_stats.json",
                "norm_sha256": "a" * 64,
                "vlm_checkpoints": {
                    task_id: f"/models/vlm/task{task_id:02d}"
                    for task_id in range(1, 27)
                },
                "official_runtime_root": "/runtime/robomemarena",
                "openpi_root": "/runtime/openpi",
                "openpi_inference_root": "/runtime/openpi_inference",
                "fullvlm_data_root": "/data/fullvlm",
            }
        ),
        encoding="utf-8",
    )

    assets = load_external_assets(config_path)

    assert assets.vlm_checkpoint_for(1) == Path("/models/vlm/task01")
    assert assets.vlm_checkpoint_for(26) == Path("/models/vlm/task26")
