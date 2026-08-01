from __future__ import annotations

from pathlib import Path

from .assets import ASSET_TOKEN_PREFIX, ROOT_TOKEN, ExternalAssets, token_values
from .models import TaskProfile


DATA_ROOT_TOKEN = "${ASSET_FULLVLM_DATA_ROOT}"


def _render_source_text(*, text: str, repo_root: Path, assets: ExternalAssets, task_id: int) -> str:
    rendered = text
    for token, replacement in token_values(
        sync_root=repo_root,
        assets=assets,
        task_id=task_id,
    ).items():
        rendered = rendered.replace(token, replacement)
    if ROOT_TOKEN in rendered or ASSET_TOKEN_PREFIX in rendered:
        raise RuntimeError("runtime source contains an unresolved public asset token")
    return rendered


def _materialize_source_overlay(
    *,
    profile: TaskProfile,
    assets: ExternalAssets,
    output_dir: Path,
    repo_root: Path,
) -> Path:
    overlay_root = output_dir / "runtime_source"
    for source in profile.source_paths:
        source = source.resolve()
        try:
            relative = source.relative_to(repo_root)
        except ValueError as exc:
            raise RuntimeError(
                f"profile source is outside repo root and cannot be materialized: {source}"
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = overlay_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            _render_source_text(
                text=source.read_text(encoding="utf-8"),
                repo_root=repo_root,
                assets=assets,
                task_id=profile.task_id,
            ),
            encoding="utf-8",
        )
    return overlay_root


def materialize_runtime_env(
    *,
    profile: TaskProfile,
    assets: ExternalAssets,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, str]:
    """Render the public source snapshot and redirect one rollout to it."""

    repo_root = repo_root.resolve()
    overlay_root = _materialize_source_overlay(
        profile=profile,
        assets=assets,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    env = dict(profile.runtime_env)
    for key, value in list(env.items()):
        env[key] = value.replace(str(repo_root), str(overlay_root))
    env.update(
        {
            "OPENPI_ROOT": str(assets.openpi_root),
            "INFER_ROOT": str(assets.openpi_inference_root),
            "OPENPI_INFERENCE_ROOT": str(assets.openpi_inference_root),
            "SOURCE_ROOT": str(assets.official_runtime_root),
            "ROBOMEMARENA_FULLVLM_DATA_ROOT": str(assets.fullvlm_data_root),
            "TARGET_LIBERO_PATH": str(
                assets.official_runtime_root / "evaluation_benchmark/libero_fork"
            ),
        }
    )
    if assets.h5dump_bin is not None:
        env.setdefault("H5DUMP_BIN", str(assets.h5dump_bin))
    return env
