from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ProfileStatus = Literal["frozen-success", "best-local", "experimental"]
PluginKind = Literal["native", "frozen-subprocess"]
SeedMode = Literal["increment", "fixed"]


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeTopology:
    allocation_gpus: int
    vla_visible_index: int
    vlm_visible_index: int

    def __post_init__(self) -> None:
        if self.allocation_gpus not in {1, 2}:
            raise ProfileError("allocation_gpus must be 1 or 2")
        for name, index in (
            ("vla_visible_index", self.vla_visible_index),
            ("vlm_visible_index", self.vlm_visible_index),
        ):
            if not 0 <= index < self.allocation_gpus:
                raise ProfileError(
                    f"{name} outside allocation_gpus={self.allocation_gpus}: {index}"
                )


@dataclass(frozen=True)
class TaskProfile:
    task_id: int
    task_name: str
    status: ProfileStatus
    plugin_kind: PluginKind
    plugin_entrypoint: Path
    bddl_path: Path
    vla_checkpoint: Path
    vlm_checkpoint: Path
    norm_path: Path
    norm_sha256: str
    seed: int
    seed_mode: SeedMode
    replan_steps: int
    max_steps: int
    runtime_topology: RuntimeTopology
    runtime_env: dict[str, str] = field(default_factory=dict)
    prompt_config: dict[str, Any] = field(default_factory=dict)
    source_paths: tuple[Path, ...] = ()
    source_hashes: dict[str, str] = field(default_factory=dict)
    hf_assets: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileVerification:
    task_id: int
    checked_paths: int
    checked_hashes: int
