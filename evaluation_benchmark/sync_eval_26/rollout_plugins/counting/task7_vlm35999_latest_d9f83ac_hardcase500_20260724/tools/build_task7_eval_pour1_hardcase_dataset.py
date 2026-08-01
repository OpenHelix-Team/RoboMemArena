#!/usr/bin/env python3
"""Add audited Task7 rollout windows that must remain in the first-pour prompt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


TASK_ID = 7
PICK = "pick tomato sauce"
POUR_ONE = "pour tomato sauce into frypan 1st"
POUR_TWO = "pour tomato sauce into frypan 2nd"
PLACE = "place tomato sauce bowl drainer"
ALLOWED = {PICK, POUR_ONE, POUR_TWO}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def primitive(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("current_primitive") or "")


def image_placeholder_count(row: dict[str, Any]) -> int:
    return sum(str(message.get("content") or "").count("<image>") for message in row["messages"])


def set_primitive(row: dict[str, Any], label: str) -> None:
    metadata = row.setdefault("metadata", {})
    metadata["current_primitive"] = label
    metadata["policy_prompt_label"] = label
    metadata["keyframe_positions"] = []
    metadata["window_keyframe_absolute_indices"] = []
    metadata["label_semantics"] = "next_low_level_policy_prompt"
    for message in reversed(row["messages"]):
        if message.get("role") == "assistant":
            message["content"] = json.dumps(
                {"current_primitive": label, "keyframe_positions": []}, ensure_ascii=False
            )
            return
    raise ValueError(f"missing assistant message: {row.get('qid')}")


def trace_rows(trace_path: Path, *, start: int, end: int, stride: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in read_jsonl(trace_path):
        timestep = int(item["t"])
        if start <= timestep <= end and (timestep - start) % stride == 0:
            selected.append(item)
    if not selected:
        raise ValueError(f"no trace windows in [{start}, {end}] with stride={stride}")
    return selected


def copy_images(trace: dict[str, Any], *, episode_root: Path, image_root: Path) -> list[str]:
    timestep = int(trace["t"])
    source_paths = [episode_root / relative for relative in trace["image"]["recent"]]
    if len(source_paths) != 10:
        raise ValueError(f"t={timestep} has {len(source_paths)} recent images, expected 10")
    target_dir = image_root / f"seed100_t{timestep:04d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for index, source in enumerate(source_paths, 1):
        if not source.is_file():
            raise FileNotFoundError(source)
        target = target_dir / f"image_{index:02d}_{source.name}"
        shutil.copy2(source, target)
        copied.append(str(target))
    return copied


def build_dataset(
    source_rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    episode_root: Path,
    image_root: Path,
    duplicate_factor: int,
    source_run: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if duplicate_factor < 1:
        raise ValueError("duplicate_factor must be positive")
    template = next(
        (
            row
            for row in source_rows
            if primitive(row) == POUR_ONE
            and len(row.get("images") or []) == 10
            and image_placeholder_count(row) == 10
        ),
        None,
    )
    if template is None:
        raise ValueError("source dataset has no runtime-aligned 10-image first-pour template row")

    augmented: list[dict[str, Any]] = []
    for trace in traces:
        timestep = int(trace["t"])
        images = copy_images(trace, episode_root=episode_root, image_root=image_root)
        raw_prompt = str(trace.get("subtask") or "")
        for duplicate_index in range(duplicate_factor):
            row = copy.deepcopy(template)
            row["qid"] = f"task7_evalhardcase_seed100_t{timestep:04d}_pour1_dup{duplicate_index:03d}"
            row["images"] = images
            set_primitive(row, POUR_ONE)
            metadata = row["metadata"]
            metadata.update(
                {
                    "augmentation": "task7_eval_pour1_hardcase",
                    "augmentation_source": "task7_eval_rollout_pick_regression",
                    "augmentation_duplicate_index": duplicate_index,
                    "augmentation_factor": duplicate_factor,
                    "eval_source_run": source_run,
                    "eval_source_timestep": timestep,
                    "eval_vlm_raw_primitive": raw_prompt,
                    "eval_target_primitive": POUR_ONE,
                    "source_qid": template["qid"],
                }
            )
            augmented.append(row)

    rows = source_rows + augmented
    labels = Counter(primitive(row) for row in rows)
    if labels.get(PLACE, 0) or set(labels) - ALLOWED:
        raise AssertionError(f"no-place invariant failed: {dict(labels)}")
    audit = {
        "input_rows": len(source_rows),
        "hardcase_trace_windows": len(traces),
        "duplicate_factor": duplicate_factor,
        "augmented_rows": len(augmented),
        "output_rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "place_rows": labels.get(PLACE, 0),
    }
    return rows, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--start", type=int, default=455)
    parser.add_argument("--end", type=int, default=2480)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--duplicate-factor", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_root = args.output_dir / "images"
    rows, audit = build_dataset(
        read_jsonl(args.source_jsonl),
        trace_rows(args.trace_jsonl, start=args.start, end=args.end, stride=args.stride),
        episode_root=args.episode_root,
        image_root=image_root,
        duplicate_factor=args.duplicate_factor,
        source_run=args.source_run,
    )
    output_jsonl = args.output_dir / "swift_compiled_data.jsonl"
    write_jsonl(output_jsonl, rows)
    audit.update(
        {
            "source_jsonl": str(args.source_jsonl),
            "source_jsonl_sha256": sha256(args.source_jsonl),
            "trace_jsonl": str(args.trace_jsonl),
            "trace_jsonl_sha256": sha256(args.trace_jsonl),
            "output_jsonl": str(output_jsonl),
            "output_jsonl_sha256": sha256(output_jsonl),
            "image_count": len(list(image_root.rglob("*.png"))),
        }
    )
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
