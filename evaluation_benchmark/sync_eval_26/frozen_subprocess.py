from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .assets import load_external_assets
from .plugin_api import RolloutRequest, RolloutResult, verify_plugin_sources
from .runtime_config import materialize_runtime_env


DIAGNOSTIC_ENV_PREFIXES = ("ORACLE_",)
DIAGNOSTIC_ENV_NAMES = {
    "OBJECT_ANCHOR",
    "OBJECT_MW_ANCHOR",
    "OBJECT_TELEPORT",
    "TELEPORT_OBJECT",
}
FALSE_VALUES = {"", "0", "false", "no", "off"}


def _enabled(value: str) -> bool:
    return value.strip().lower() not in FALSE_VALUES


def _assert_autonomy(profile_status: str, runtime_env: dict[str, str]) -> None:
    if profile_status == "experimental":
        return
    for name, value in runtime_env.items():
        diagnostic = (
            name.startswith(DIAGNOSTIC_ENV_PREFIXES) or name in DIAGNOSTIC_ENV_NAMES
        )
        if diagnostic and _enabled(value):
            raise RuntimeError(
                f"diagnostic runtime flag is forbidden for {profile_status}: "
                f"{name}={value}"
            )


def _command(entrypoint: Path) -> list[str]:
    if entrypoint.suffix == ".py":
        return [sys.executable, str(entrypoint)]
    if entrypoint.suffix == ".sh":
        return ["bash", str(entrypoint)]
    if os.access(entrypoint, os.X_OK):
        return [str(entrypoint)]
    raise RuntimeError(f"unsupported plugin entrypoint: {entrypoint}")


class FrozenSubprocessPlugin:
    def run(self, request: RolloutRequest) -> RolloutResult:
        profile = request.profile
        verify_plugin_sources(profile.source_hashes)
        _assert_autonomy(profile.status, profile.runtime_env)
        request.output_dir.mkdir(parents=True, exist_ok=True)

        official_result = request.output_dir / "official_episode.json"
        driver_log = request.output_dir / "driver.log"
        env = os.environ.copy()
        assets = load_external_assets()
        env.update(
            materialize_runtime_env(
                profile=profile,
                assets=assets,
                output_dir=request.output_dir,
                repo_root=request.repo_root,
            )
        )
        env.update(request.runtime_env)
        env.update(
            {
                "TASK_ID": str(request.task_id),
                "EPISODE_INDEX": str(request.episode_index),
                "SEED": str(request.seed),
                "OUTPUT_DIR": str(request.output_dir),
                "VLA_CHECKPOINT": str(profile.vla_checkpoint),
                "VLM_CHECKPOINT": str(profile.vlm_checkpoint),
                "NORM_STATS_PATH": str(profile.norm_path),
                "NORM_STATS_SHA256": profile.norm_sha256,
                "BDDL_PATH": str(profile.bddl_path),
                "OFFICIAL_SCORER_LOCK": str(request.scorer_lock_path),
                "OFFICIAL_RESULT_JSON": str(official_result),
                "REPLAN_STEPS": str(profile.replan_steps),
                "MAX_STEPS": str(profile.max_steps),
            }
        )

        with driver_log.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                _command(profile.plugin_entrypoint),
                cwd=request.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if completed.returncode != 0:
            return RolloutResult(
                return_code=completed.returncode,
                episode_log=driver_log,
                official_result_path=official_result,
                video_paths=(),
                prompt_trace=None,
                termination_reason=f"plugin_exit_{completed.returncode}",
            )
        if not official_result.is_file():
            raise RuntimeError(
                f"plugin exited successfully without official result: {official_result}"
            )

        payload = json.loads(official_result.read_text(encoding="utf-8"))
        videos = tuple(Path(item) for item in payload.get("video_paths", []))
        prompt_raw = payload.get("prompt_trace")
        return RolloutResult(
            return_code=0,
            episode_log=driver_log,
            official_result_path=official_result,
            video_paths=videos,
            prompt_trace=Path(prompt_raw) if prompt_raw else None,
            termination_reason=str(payload.get("termination_reason", "completed")),
        )
