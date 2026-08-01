#!/usr/bin/env python3
"""Build a counting-task no-place VLM dataset with explicit Pour1-to-Pour2 boundaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_LABELS = {
    6: (
        "pick tomato sauce",
        "pour tomato sauce over cookies 1st",
        "pour tomato sauce over cookies 2nd",
        "place tomato sauce bowl drainer",
    ),
    7: (
        "pick tomato sauce",
        "pour tomato sauce into frypan 1st",
        "pour tomato sauce into frypan 2nd",
        "place tomato sauce bowl drainer",
    ),
}

TASK_ID = 7
TASK_NAME = "task7"
PICK, POUR_ONE, POUR_TWO, PLACE = TASK_LABELS[TASK_ID]
ALLOWED = (PICK, POUR_ONE, POUR_TWO)


def configure_task(task_id: int) -> None:
    """Set the four task-specific primitive strings for one counting task."""
    global TASK_ID, TASK_NAME, PICK, POUR_ONE, POUR_TWO, PLACE, ALLOWED
    if task_id not in TASK_LABELS:
        raise ValueError(f"unsupported counting task: {task_id}")
    TASK_ID = int(task_id)
    TASK_NAME = f"task{TASK_ID}"
    PICK, POUR_ONE, POUR_TWO, PLACE = TASK_LABELS[TASK_ID]
    ALLOWED = (PICK, POUR_ONE, POUR_TWO)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def row_label(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    label = str(meta.get("current_primitive") or "").strip()
    if label:
        return label
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        payload = json.loads(str(message.get("content") or "{}"))
        return str(payload.get("current_primitive") or "").strip()
    return ""


def row_end(row: dict[str, Any]) -> int:
    meta = row.get("metadata") or {}
    return int(meta.get("absolute_window_end", meta.get("window_end", -1)))


def set_label(row: dict[str, Any], label: str) -> None:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            message["content"] = json.dumps(
                {"current_primitive": label, "keyframe_positions": []}, ensure_ascii=False
            )
            break
    else:
        raise ValueError(f"row {row.get('qid')} has no assistant message")
    metadata = row.setdefault("metadata", {})
    metadata["current_primitive"] = label
    metadata["policy_prompt_label"] = label
    metadata["keyframe_positions"] = []
    metadata["window_keyframe_absolute_indices"] = []
    metadata["label_semantics"] = "next_low_level_policy_prompt"


def clone_as_pour_two(row: dict[str, Any], *, source_kind: str, duplicate_index: int, duplicate_factor: int) -> dict[str, Any]:
    out = copy.deepcopy(row)
    source_label = row_label(row)
    qid = str(row.get("qid") or f"{TASK_NAME}_unknown")
    out["qid"] = f"{qid}__pour2boundary_{source_kind}_dup{duplicate_index:02d}"
    set_label(out, POUR_TWO)
    metadata = out.setdefault("metadata", {})
    metadata.update(
        {
            "source_qid": qid,
            "source_current_primitive": source_label,
            "augmentation": f"{TASK_NAME}_pour2_no_place_boundary",
            "augmentation_source": f"{TASK_NAME}_pour2_boundary:{source_kind}",
            "augmentation_factor": duplicate_factor,
            "augmentation_duplicate_index": duplicate_index,
            "label_transform": f"{source_kind}_as_pour_two",
            "next_policy_source_primitive": source_label,
            "next_policy_target_primitive": POUR_TWO,
        }
    )
    return out


def boundary_candidates(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Prefer unaugmented trajectory rows so tail/head selection is semantically ordered."""
    raw = [row for row in rows if not (row.get("metadata") or {}).get("augmentation_source")]
    return raw if raw else rows


def build_rows(
    rows: list[dict[str, Any]],
    *,
    tail_windows: int,
    head_windows: int,
    tail_duplicate_factor: int,
    head_duplicate_factor: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_rows = [row for row in rows if row_label(row) in ALLOWED]
    unknown = Counter(row_label(row) for row in rows if row_label(row) not in {*ALLOWED, PLACE})
    if unknown:
        raise ValueError(f"unexpected {TASK_NAME} labels: {dict(unknown)}")

    grouped: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in base_rows:
        meta = row.get("metadata") or {}
        seed = int(meta.get("seed", -1))
        if seed < 0:
            raise ValueError(f"row without seed: {row.get('qid')}")
        grouped[seed][row_label(row)].append(row)

    augmented: list[dict[str, Any]] = []
    boundary_counts: dict[str, int] = {}
    for seed, by_label in sorted(grouped.items()):
        source_tail = sorted(boundary_candidates(by_label[POUR_ONE], POUR_ONE), key=row_end)[-tail_windows:]
        target_head = sorted(boundary_candidates(by_label[POUR_TWO], POUR_TWO), key=row_end)[:head_windows]
        if not source_tail or not target_head:
            raise ValueError(f"seed {seed} lacks Pour1/Pou2 boundary data")
        boundary_counts[str(seed)] = len(source_tail) + len(target_head)
        for row in source_tail:
            for duplicate_index in range(tail_duplicate_factor):
                augmented.append(
                    clone_as_pour_two(
                        row,
                        source_kind="pour1_tail",
                        duplicate_index=duplicate_index,
                        duplicate_factor=tail_duplicate_factor,
                    )
                )
        for row in target_head:
            for duplicate_index in range(head_duplicate_factor):
                augmented.append(
                    clone_as_pour_two(
                        row,
                        source_kind="pour2_head",
                        duplicate_index=duplicate_index,
                        duplicate_factor=head_duplicate_factor,
                    )
                )

    all_rows = base_rows + augmented
    labels = Counter(row_label(row) for row in all_rows)
    if PLACE in labels or set(labels) - set(ALLOWED):
        raise AssertionError(f"no-place invariant failed: {dict(labels)}")
    summary = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "primitive_labels": list(TASK_LABELS[TASK_ID]),
        "input_rows": len(rows),
        "base_rows": len(base_rows),
        "dropped_place_rows": sum(1 for row in rows if row_label(row) == PLACE),
        "augmented_rows": len(augmented),
        "output_rows": len(all_rows),
        "labels": dict(labels),
        "seed_count": len(grouped),
        "per_seed_boundary_source_rows": boundary_counts,
    }
    return all_rows, summary


def validate_rows(rows: list[dict[str, Any],], *, audit_size: int, seed: int) -> dict[str, Any]:
    labels = Counter(row_label(row) for row in rows)
    if PLACE in labels or set(labels) - set(ALLOWED):
        raise AssertionError(f"no-place invariant failed: {dict(labels)}")
    for row in rows:
        text = "\n".join(str(message.get("content") or "") for message in row.get("messages") or [])
        if text.count("<image>") != len(row.get("images") or []):
            raise AssertionError(f"image placeholder mismatch: {row.get('qid')}")
    sample = random.Random(seed).sample(rows, min(audit_size, len(rows)))
    missing = [str(image) for row in sample for image in row.get("images") or [] if not Path(str(image)).is_file()]
    if missing:
        raise FileNotFoundError(f"missing sampled images: {missing[:5]}")
    return {"audited_rows": len(sample), "missing_sampled_images": len(missing), "labels": dict(labels)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=sorted(TASK_LABELS), default=7)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-windows", type=int, default=16)
    parser.add_argument("--head-windows", type=int, default=12)
    parser.add_argument("--tail-duplicate-factor", type=int, default=6)
    parser.add_argument("--head-duplicate-factor", type=int, default=3)
    parser.add_argument("--audit-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    if min(args.tail_windows, args.head_windows, args.tail_duplicate_factor, args.head_duplicate_factor) <= 0:
        raise ValueError("window and duplicate parameters must be positive")
    configure_task(args.task_id)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(args.input_jsonl)
    output_rows, summary = build_rows(
        rows,
        tail_windows=args.tail_windows,
        head_windows=args.head_windows,
        tail_duplicate_factor=args.tail_duplicate_factor,
        head_duplicate_factor=args.head_duplicate_factor,
    )
    output_jsonl = args.output_dir / "swift_compiled_data.jsonl"
    write_jsonl(output_jsonl, output_rows)
    summary.update(
        {
            "input_jsonl": str(args.input_jsonl),
            "output_jsonl": str(output_jsonl),
            "tail_windows": args.tail_windows,
            "head_windows": args.head_windows,
            "tail_duplicate_factor": args.tail_duplicate_factor,
            "head_duplicate_factor": args.head_duplicate_factor,
            "prompt_text_mode": "no_label_no_order",
            "validation": validate_rows(output_rows, audit_size=args.audit_size, seed=args.seed),
            "output_sha256": sha256(output_jsonl),
        }
    )
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
