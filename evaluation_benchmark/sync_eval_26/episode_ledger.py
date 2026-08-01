from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .profile_loader import load_profile
from .validation import EpisodeValidation, validate_episode


LEDGER_COLUMNS = (
    "task_id",
    "episode_index",
    "expected_seed",
    "selection_status",
    "candidate_count",
    "valid_candidate_count",
    "validation_reason",
    "stage_success",
    "stage_score_pct",
    "goal_success",
    "scorer_commit",
    "profile_sha256",
    "manifest_sha256",
    "official_result_sha256",
    "vla_checkpoint",
    "vlm_checkpoint",
    "norm_path",
    "source_root",
    "episode_dir",
    "manifest_path",
    "official_result_path",
    "episode_log_path",
    "video_paths_json",
)


CANDIDATE_COLUMNS = (
    "task_id",
    "episode_index",
    "root_priority",
    "eligible_for_selection",
    "selected",
    "valid",
    "validation_reason",
    "source_root",
    "episode_dir",
    "manifest_path",
    "manifest_sha256",
    "official_result_path",
    "official_result_sha256",
)


VALID_SELECTION_STATUSES = frozenset(
    {"valid_selected", "multiple_valid_selected_by_root_priority"}
)


@dataclass(frozen=True)
class LedgerCandidate:
    task_id: int
    episode_index: int
    root_priority: int
    source_root: Path
    episode_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    validation: EpisodeValidation
    eligible_for_selection: bool


@dataclass(frozen=True)
class MappedManifest:
    """One canonical campaign trial backed by an immutable source manifest.

    Fixed-seed batches are commonly sharded, so each source shard starts its
    local episode numbering at zero.  ``episode_index`` is the canonical trial
    number used by the campaign ledger; ``manifest_path`` remains the original
    source manifest and is validated in place.
    """

    task_id: int
    episode_index: int
    source_root: Path
    manifest_path: Path


Validator = Callable[..., EpisodeValidation]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_text(value: object) -> str:
    return str(value) if isinstance(value, (str, Path)) else ""


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _candidate_paths(
    output_roots: Sequence[Path],
    *,
    task_ids: Sequence[int],
    episodes: int,
) -> Iterable[tuple[int, Path, Path]]:
    """Find only expected manifest paths; never recursively walk video trees."""
    for root_priority, root in enumerate(output_roots):
        root = root.resolve()
        if not root.is_dir():
            continue
        for task_id in task_ids:
            for episode_index in range(episodes):
                manifest_path = (
                    root
                    / f"task{task_id:02d}"
                    / f"ep{episode_index:03d}"
                    / "run_manifest.json"
                )
                if manifest_path.is_file():
                    yield root_priority, root, manifest_path


def collect_candidates(
    *,
    repo_root: Path,
    profile_dir: Path,
    output_roots: Sequence[Path],
    task_ids: Sequence[int],
    episodes: int,
    eligible_task_ids_by_root: Sequence[frozenset[int]] | None = None,
    validator: Validator = validate_episode,
) -> dict[tuple[int, int], list[LedgerCandidate]]:
    if (
        eligible_task_ids_by_root is not None
        and len(eligible_task_ids_by_root) != len(output_roots)
    ):
        raise ValueError("eligible_task_ids_by_root must match output_roots")
    allowed = set(task_ids)
    candidates: dict[tuple[int, int], list[LedgerCandidate]] = {}
    for root_priority, source_root, manifest_path in _candidate_paths(
        output_roots,
        task_ids=task_ids,
        episodes=episodes,
    ):
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            continue
        try:
            task_id = int(manifest["task_id"])
            episode_index = int(manifest["episode_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if task_id not in allowed or episode_index < 0:
            continue
        eligible = (
            True
            if eligible_task_ids_by_root is None
            else task_id in eligible_task_ids_by_root[root_priority]
        )
        validation = validator(
            repo_root=repo_root,
            profile_path=profile_dir / f"task{task_id:02d}.yaml",
            episode_dir=manifest_path.parent,
        )
        candidate = LedgerCandidate(
            task_id=task_id,
            episode_index=episode_index,
            root_priority=root_priority,
            source_root=source_root,
            episode_dir=manifest_path.parent,
            manifest_path=manifest_path,
            manifest=manifest,
            validation=validation,
            eligible_for_selection=eligible,
        )
        candidates.setdefault((task_id, episode_index), []).append(candidate)
    for rows in candidates.values():
        rows.sort(key=lambda row: (row.root_priority, str(row.manifest_path)))
    return candidates


def collect_mapped_candidates(
    *,
    repo_root: Path,
    profile_dir: Path,
    mapped_manifests: Sequence[MappedManifest],
    validator: Validator = validate_episode,
) -> dict[tuple[int, int], list[LedgerCandidate]]:
    """Validate source manifests while assigning deterministic global trials."""
    candidates: dict[tuple[int, int], list[LedgerCandidate]] = {}
    for root_priority, mapping in enumerate(mapped_manifests):
        manifest_path = mapping.manifest_path.resolve()
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            continue
        try:
            manifest_task_id = int(manifest["task_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if manifest_task_id != mapping.task_id:
            continue
        validation = validator(
            repo_root=repo_root,
            profile_path=profile_dir / f"task{mapping.task_id:02d}.yaml",
            episode_dir=manifest_path.parent,
        )
        candidate = LedgerCandidate(
            task_id=mapping.task_id,
            episode_index=mapping.episode_index,
            root_priority=root_priority,
            source_root=mapping.source_root.resolve(),
            episode_dir=manifest_path.parent,
            manifest_path=manifest_path,
            manifest=manifest,
            validation=validation,
            eligible_for_selection=True,
        )
        candidates.setdefault((mapping.task_id, mapping.episode_index), []).append(
            candidate
        )
    for rows in candidates.values():
        rows.sort(key=lambda row: (row.root_priority, str(row.manifest_path)))
    return candidates


def _expected_seed(profile_dir: Path, task_id: int, episode_index: int) -> int | str:
    try:
        profile = load_profile(profile_dir / f"task{task_id:02d}.yaml")
    except Exception as exc:
        return f"profile_load_error:{type(exc).__name__}"
    if profile.seed_mode == "fixed":
        return profile.seed
    return profile.seed + episode_index


def _official_goal_success(candidate: LedgerCandidate) -> str:
    path = _path_text(candidate.manifest.get("rollout", {}).get("official_result_path"))
    if not path:
        return ""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    value = payload.get("goal_success")
    if value is None:
        return ""
    return str(bool(value)).lower()


def _maybe_hash(path_text: str) -> str:
    path = Path(path_text)
    return sha256(path) if path.is_file() else ""


def _row_from_candidate(
    *,
    candidate: LedgerCandidate,
    selection_status: str,
    candidate_count: int,
    valid_candidate_count: int,
    expected_seed: int | str,
) -> dict[str, str | int | float]:
    manifest = candidate.manifest
    rollout = manifest.get("rollout", {})
    if not isinstance(rollout, dict):
        rollout = {}
    official_path = _path_text(rollout.get("official_result_path"))
    episode_log = _path_text(rollout.get("episode_log"))
    videos = rollout.get("video_paths", [])
    if not isinstance(videos, list):
        videos = []
    record = candidate.validation.record
    return {
        "task_id": candidate.task_id,
        "episode_index": candidate.episode_index,
        "expected_seed": expected_seed,
        "selection_status": selection_status,
        "candidate_count": candidate_count,
        "valid_candidate_count": valid_candidate_count,
        "validation_reason": candidate.validation.reason,
        "stage_success": "" if record is None else str(record.stage_success).lower(),
        "stage_score_pct": "" if record is None else record.stage_score_pct,
        "goal_success": _official_goal_success(candidate),
        "scorer_commit": _path_text(manifest.get("scorer_commit")),
        "profile_sha256": _path_text(manifest.get("profile_sha256")),
        "manifest_sha256": sha256(candidate.manifest_path),
        "official_result_sha256": _maybe_hash(official_path),
        "vla_checkpoint": _path_text(manifest.get("vla_checkpoint")),
        "vlm_checkpoint": _path_text(manifest.get("vlm_checkpoint")),
        "norm_path": _path_text(manifest.get("norm_path")),
        "source_root": str(candidate.source_root),
        "episode_dir": str(candidate.episode_dir),
        "manifest_path": str(candidate.manifest_path),
        "official_result_path": official_path,
        "episode_log_path": episode_log,
        "video_paths_json": json.dumps(videos, ensure_ascii=True),
    }


def build_ledger_rows(
    *,
    repo_root: Path,
    profile_dir: Path,
    output_roots: Sequence[Path],
    task_ids: Sequence[int],
    episodes: int,
    eligible_task_ids_by_root: Sequence[frozenset[int]] | None = None,
    validator: Validator = validate_episode,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int]]]:
    candidates = collect_candidates(
        repo_root=repo_root,
        profile_dir=profile_dir,
        output_roots=output_roots,
        task_ids=task_ids,
        episodes=episodes,
        eligible_task_ids_by_root=eligible_task_ids_by_root,
        validator=validator,
    )
    return _build_rows_from_candidates(
        candidates=candidates,
        profile_dir=profile_dir,
        task_ids=task_ids,
        episodes=episodes,
    )


def build_mapped_ledger_rows(
    *,
    repo_root: Path,
    profile_dir: Path,
    mapped_manifests: Sequence[MappedManifest],
    task_ids: Sequence[int],
    episodes: int,
    validator: Validator = validate_episode,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int]]]:
    """Build ledger rows for explicitly mapped fixed-seed source trials."""
    candidates = collect_mapped_candidates(
        repo_root=repo_root,
        profile_dir=profile_dir,
        mapped_manifests=mapped_manifests,
        validator=validator,
    )
    return _build_rows_from_candidates(
        candidates=candidates,
        profile_dir=profile_dir,
        task_ids=task_ids,
        episodes=episodes,
    )


def _build_rows_from_candidates(
    *,
    candidates: dict[tuple[int, int], list[LedgerCandidate]],
    profile_dir: Path,
    task_ids: Sequence[int],
    episodes: int,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int]]]:
    rows: list[dict[str, str | int | float]] = []
    candidate_rows: list[dict[str, str | int]] = []
    for task_id in task_ids:
        for episode_index in range(episodes):
            key = (task_id, episode_index)
            choices = candidates.get(key, [])
            valid = [
                choice
                for choice in choices
                if choice.validation.valid and choice.eligible_for_selection
            ]
            expected_seed = _expected_seed(profile_dir, task_id, episode_index)
            selected = valid[0] if valid else (choices[0] if choices else None)
            if selected is None:
                missing_row = {column: "" for column in LEDGER_COLUMNS}
                missing_row.update(
                    {
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "expected_seed": expected_seed,
                        "selection_status": "missing",
                        "candidate_count": 0,
                        "valid_candidate_count": 0,
                        "validation_reason": "missing_run_manifest",
                    }
                )
                rows.append(missing_row)
            else:
                if selected.validation.valid and not selected.eligible_for_selection:
                    status = "only_ineligible_candidates"
                else:
                    status = (
                        "valid_selected"
                        if selected.validation.valid
                        else "invalid_candidate"
                    )
                if len(valid) > 1:
                    status = "multiple_valid_selected_by_root_priority"
                rows.append(
                    _row_from_candidate(
                        candidate=selected,
                        selection_status=status,
                        candidate_count=len(choices),
                        valid_candidate_count=len(valid),
                        expected_seed=expected_seed,
                    )
                )
            for choice in choices:
                rollout = choice.manifest.get("rollout", {})
                if not isinstance(rollout, dict):
                    rollout = {}
                official_path = _path_text(rollout.get("official_result_path"))
                candidate_rows.append(
                    {
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "root_priority": choice.root_priority,
                        "eligible_for_selection": str(
                            choice.eligible_for_selection
                        ).lower(),
                        "selected": str(choice is selected).lower(),
                        "valid": str(choice.validation.valid).lower(),
                        "validation_reason": choice.validation.reason,
                        "source_root": str(choice.source_root),
                        "episode_dir": str(choice.episode_dir),
                        "manifest_path": str(choice.manifest_path),
                        "manifest_sha256": sha256(choice.manifest_path),
                        "official_result_path": official_path,
                        "official_result_sha256": _maybe_hash(official_path),
                    }
                )
    return rows, candidate_rows


def write_tsv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def merge_inventory_rows(
    *,
    inventories: Sequence[dict[str, object]],
    task_ids: Sequence[int],
    episodes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Merge independently exported root inventories without touching their filesystems."""
    keyed_rows: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = {}
    keyed_candidates: dict[tuple[int, int], list[tuple[int, dict[str, object]]]] = {}
    for inventory_index, inventory in enumerate(inventories):
        for raw_row in inventory.get("ledger_rows", []):
            if not isinstance(raw_row, dict):
                continue
            try:
                key = (int(raw_row["task_id"]), int(raw_row["episode_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            keyed_rows.setdefault(key, []).append((inventory_index, dict(raw_row)))
        for raw_row in inventory.get("candidate_rows", []):
            if not isinstance(raw_row, dict):
                continue
            try:
                key = (int(raw_row["task_id"]), int(raw_row["episode_index"]))
            except (KeyError, TypeError, ValueError):
                continue
            keyed_candidates.setdefault(key, []).append((inventory_index, dict(raw_row)))

    merged_rows: list[dict[str, object]] = []
    merged_candidates: list[dict[str, object]] = []
    for task_id in task_ids:
        for episode_index in range(episodes):
            key = (task_id, episode_index)
            source_rows = keyed_rows.get(key, [])
            nonmissing = [
                row
                for row in source_rows
                if str(row[1].get("selection_status", "")) != "missing"
            ]
            valid = [
                row
                for row in source_rows
                if str(row[1].get("selection_status", "")) in VALID_SELECTION_STATUSES
            ]
            selected_pair = valid[0] if valid else (nonmissing[0] if nonmissing else None)
            if selected_pair is None:
                if source_rows:
                    selected = dict(source_rows[0][1])
                else:
                    selected = {column: "" for column in LEDGER_COLUMNS}
                    selected.update(
                        {
                            "task_id": task_id,
                            "episode_index": episode_index,
                            "selection_status": "missing",
                            "validation_reason": "missing_inventory_row",
                        }
                    )
            else:
                selected = dict(selected_pair[1])

            candidate_count = sum(
                int(row.get("candidate_count", 0) or 0)
                for _, row in source_rows
            )
            valid_candidate_count = sum(
                int(row.get("valid_candidate_count", 0) or 0)
                for _, row in source_rows
            )
            if len(valid) > 1:
                selected["selection_status"] = "multiple_valid_selected_by_root_priority"
            elif selected_pair is None:
                selected["selection_status"] = "missing"
            selected["candidate_count"] = candidate_count
            selected["valid_candidate_count"] = valid_candidate_count
            merged_rows.append(selected)

            selected_manifest = "" if selected_pair is None else str(
                selected.get("manifest_path", "")
            )
            for _, candidate in keyed_candidates.get(key, []):
                candidate["selected"] = str(
                    bool(selected_manifest)
                    and str(candidate.get("manifest_path", "")) == selected_manifest
                ).lower()
                merged_candidates.append(candidate)
    return merged_rows, merged_candidates
