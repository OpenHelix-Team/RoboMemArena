from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .frozen_subprocess import DIAGNOSTIC_ENV_NAMES, DIAGNOSTIC_ENV_PREFIXES, _enabled
from .assets import load_external_assets
from .profile_loader import load_profile, verify_profile_assets
from .scoring_bridge import OfficialScoringBridge
from .upstream_lock import verify_upstream_lock


OFFICIAL_COMMIT = "d9f83ac5182e25ad7f0a301a77a0b667f2392df1"
MODEL_WEIGHT_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "adapter_model.safetensors",
)
EMBEDDED_OFFICIAL_SOURCE_MARKERS = (
    "source/RoboMemArena_d9f83ac",
    'source" / "RoboMemArena_d9f83ac',
    "source' / 'RoboMemArena_d9f83ac",
)


@dataclass(frozen=True)
class ProfilePreflight:
    task_id: int
    status: str
    checked_paths: int
    checked_hashes: int
    stage_names: tuple[str, ...]
    vlm_weight_file: str
    official_runtime_root: str
    result: str = "PASS"


def _vlm_weight(path: Path) -> Path:
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(config)
    for name in MODEL_WEIGHT_NAMES:
        candidate = path / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            if candidate.name.endswith(".index.json"):
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                weight_map = payload.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise RuntimeError(f"invalid VLM weight index: {candidate}")
                shards = {path / str(value) for value in weight_map.values()}
                missing = [
                    shard
                    for shard in sorted(shards)
                    if not shard.is_file() or shard.stat().st_size == 0
                ]
                if missing:
                    raise FileNotFoundError(
                        f"VLM weight index has missing shards: {missing}"
                    )
            return candidate
    shards = sorted(path.glob("*.safetensors"))
    if shards and all(item.stat().st_size > 0 for item in shards):
        return shards[0]
    raise FileNotFoundError(f"no VLM weight file in {path}")


def _verify_official_runtime_root(path: Path) -> None:
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"official runtime checkout mismatch: expected={OFFICIAL_COMMIT} actual={actual}"
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=path,
        text=True,
    ).strip():
        raise RuntimeError(f"official runtime checkout is dirty: {path}")


def _verify_autonomy(
    profile_status: str, prompt_config: dict, runtime_env: dict
) -> None:
    if prompt_config.get("source") != "vlm":
        raise RuntimeError("prompt source must be VLM")
    if bool(prompt_config.get("oracle_prompt_injection")):
        raise RuntimeError("oracle prompt injection must be disabled")
    if profile_status == "experimental":
        return
    for name, value in runtime_env.items():
        diagnostic = (
            name.startswith(DIAGNOSTIC_ENV_PREFIXES) or name in DIAGNOSTIC_ENV_NAMES
        )
        if diagnostic and _enabled(value):
            raise RuntimeError(f"enabled diagnostic flag: {name}={value}")


def _verify_source_command(runtime_env: dict[str, str]) -> None:
    raw = runtime_env.get("SOURCE_COMMAND_JSON", "")
    command = json.loads(raw)
    if not isinstance(command, list) or len(command) < 2:
        raise RuntimeError(
            "SOURCE_COMMAND_JSON must contain an executable and entrypoint"
        )
    entrypoint = Path(str(command[-1]))
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)


def _verify_runtime_paths(runtime_env: dict[str, str]) -> None:
    for name, value in runtime_env.items():
        if not value or value == "__NONE__":
            continue
        path = Path(value)
        if path.is_absolute() and not path.exists():
            raise FileNotFoundError(f"{name}={path}")


def _verify_vla_checkpoint(path: Path) -> None:
    required = (
        path / "_CHECKPOINT_METADATA",
        path / "params",
        path / "params/_METADATA",
    )
    for item in required:
        if not item.exists():
            raise FileNotFoundError(item)


def _verify_no_embedded_official_source(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.suffix not in {".py", ".sh"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        marker = next(
            (item for item in EMBEDDED_OFFICIAL_SOURCE_MARKERS if item in text),
            None,
        )
        if marker is not None:
            raise RuntimeError(
                f"active rollout source embeds stale official checkout "
                f"marker={marker!r}: {path}"
            )


def verify_profiles(
    *,
    repo_root: Path,
    profile_dir: Path,
    task_ids: Iterable[int],
    output_root: Path | None = None,
) -> list[ProfilePreflight]:
    repo_root = repo_root.resolve()
    assets = load_external_assets()
    lock = verify_upstream_lock(repo_root, runtime_root=assets.official_runtime_root)
    if lock.commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"unexpected scorer commit: {lock.commit}")
    bridge = OfficialScoringBridge(repo_root, assets=assets)
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()

    results: list[ProfilePreflight] = []
    for task_id in task_ids:
        profile_path = profile_dir / f"task{task_id:02d}.yaml"
        profile = load_profile(profile_path, assets=assets)
        if profile.task_id != task_id:
            raise RuntimeError(
                f"profile task mismatch: filename={task_id} profile={profile.task_id}"
            )
        verification = verify_profile_assets(profile)
        _verify_autonomy(profile.status, profile.prompt_config, profile.runtime_env)
        _verify_no_embedded_official_source(profile.source_paths)
        _verify_source_command(profile.runtime_env)
        _verify_runtime_paths(profile.runtime_env)
        _verify_vla_checkpoint(profile.vla_checkpoint)
        official_runtime_root = Path(
            profile.runtime_env.get("OFFICIAL_RUNTIME_ROOT", "")
        )
        if not official_runtime_root.is_absolute():
            raise RuntimeError(f"Task{task_id} missing absolute OFFICIAL_RUNTIME_ROOT")
        _verify_official_runtime_root(official_runtime_root)
        weight = _vlm_weight(profile.vlm_checkpoint)
        stage_names = tuple(bridge.stage_names(task_id))
        if not stage_names:
            raise RuntimeError(f"Task{task_id} has no official stages")
        if task_id in {20, 21, 23, 24} and any(
            re.search("close.*microwave", name, re.IGNORECASE) for name in stage_names
        ):
            raise RuntimeError(f"Task{task_id} unexpectedly scores microwave close")
        results.append(
            ProfilePreflight(
                task_id=task_id,
                status=profile.status,
                checked_paths=verification.checked_paths,
                checked_hashes=verification.checked_hashes,
                stage_names=stage_names,
                vlm_weight_file=str(weight),
                official_runtime_root=str(official_runtime_root),
            )
        )
    if {item.task_id for item in results} & {22} and next(
        item for item in results if item.task_id == 22
    ).status != "experimental":
        raise RuntimeError("Task22 must remain explicitly experimental")
    return results


def preflight_as_dict(results: Iterable[ProfilePreflight]) -> dict:
    rows = [asdict(item) for item in results]
    return {
        "official_commit": OFFICIAL_COMMIT,
        "profiles": rows,
        "passed": len(rows),
        "failed": 0,
    }
