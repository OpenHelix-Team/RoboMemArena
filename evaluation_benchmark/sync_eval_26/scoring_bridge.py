from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .assets import ExternalAssets, load_external_assets
from .upstream_lock import verify_upstream_lock


STAGE_MODULE_RELATIVE_PATH = Path(
    "evaluation_benchmark/scripts/task2_26_reference_stage.py"
)
EVAL_COMMON_RELATIVE_PATH = Path(
    "evaluation_benchmark/openpi_minimal_runtime/eval_common.py"
)
OPENPI_CLIENT_RELATIVE_PATH = Path(
    "third_party/openpi_minimal/packages/openpi-client/src"
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_stage_module(repo_root: Path, runtime_root: Path) -> ModuleType:
    verify_upstream_lock(repo_root, runtime_root=runtime_root)
    stage_path = runtime_root / STAGE_MODULE_RELATIVE_PATH
    eval_common_path = runtime_root / EVAL_COMMON_RELATIVE_PATH
    openpi_client_path = runtime_root / OPENPI_CLIENT_RELATIVE_PATH
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    if not eval_common_path.is_file():
        raise FileNotFoundError(eval_common_path)
    if not openpi_client_path.is_dir():
        raise FileNotFoundError(openpi_client_path)

    previous_eval_common = sys.modules.get("eval_common")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    import_paths = (str(eval_common_path.parent), str(openpi_client_path))
    for import_path in reversed(import_paths):
        sys.path.insert(0, import_path)
    sys.dont_write_bytecode = True
    try:
        eval_common = _load_module(
            "_sync_eval_26_official_eval_common", eval_common_path
        )
        sys.modules["eval_common"] = eval_common
        return _load_module("_sync_eval_26_official_stage", stage_path)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        for import_path in import_paths:
            try:
                sys.path.remove(import_path)
            except ValueError:
                pass
        if previous_eval_common is None:
            sys.modules.pop("eval_common", None)
        else:
            sys.modules["eval_common"] = previous_eval_common


class OfficialScoringBridge:
    def __init__(
        self,
        repo_root: Path,
        *,
        assets: ExternalAssets | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.assets = assets or load_external_assets()
        self._stage = _load_official_stage_module(
            self.repo_root, self.assets.official_runtime_root
        )

    def stage_specs(self, task_id: int) -> list[Any]:
        return list(self._stage._task_specs(task_id))

    def stage_names(self, task_id: int) -> list[str]:
        return [spec.name for spec in self.stage_specs(task_id)]

    def stage_score_pct(self, task_id: int, stage_done: dict[str, bool]) -> float:
        return float(self._stage._stage_score_pct(task_id, stage_done))

    def stage_success(self, task_id: int, stage_done: dict[str, bool]) -> bool:
        return bool(self._stage._stage_success_from_stage_done(task_id, stage_done))

    def counting_pour_task(self, task_id: int) -> bool:
        return task_id in self._stage.COUNTING_POUR_TASKS

    def extra_pour_check(
        self, task_id: int
    ) -> Callable[[Any, dict[str, Any], int], bool] | None:
        return self._stage._extra_pour_check(task_id)
