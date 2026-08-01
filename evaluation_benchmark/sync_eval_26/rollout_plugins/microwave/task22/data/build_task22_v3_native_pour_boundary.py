#!/usr/bin/env python3
"""Build a Task22 VLM dataset with native pick-to-first-pour boundary copies.

The base JSONL is copied byte-for-byte.  Only canonical, already-labelled
``pour first`` rows from the beginning of the native ``pour_first`` primitive
are cloned.  This deliberately changes data frequency only; it never changes
an evaluator, a policy prompt, an image path, or a source label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original Task22 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Augmented JSONL output")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON provenance output")
    parser.add_argument("--copies", type=int, default=4, help="Extra copies per canonical boundary row")
    parser.add_argument("--task-id", type=int, default=22)
    parser.add_argument("--primitive-stem", default="pour_first")
    parser.add_argument("--current-primitive", default="pour first")
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--verify-images", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def decode_json(raw: bytes, line_number: int, source: Path) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSONL row is not an object at {source}:{line_number}")
    return value


def response_primitive(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row.get('qid', '<unknown>')} has no messages")
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if not isinstance(content, str):
        raise ValueError(f"row {row.get('qid', '<unknown>')} has no assistant JSON")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"row {row.get('qid', '<unknown>')} has invalid assistant JSON") from exc
    primitive = parsed.get("current_primitive") if isinstance(parsed, dict) else None
    if not isinstance(primitive, str):
        raise ValueError(f"row {row.get('qid', '<unknown>')} lacks current_primitive")
    return primitive


def is_canonical_boundary(row: dict[str, Any], args: argparse.Namespace) -> bool:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return False
    if metadata.get("task_id") != args.task_id:
        return False
    if metadata.get("primitive_stem") != args.primitive_stem:
        return False
    if metadata.get("current_primitive") != args.current_primitive:
        return False
    if metadata.get("window_start") != args.window_start:
        return False
    if metadata.get("repeat_index") != args.repeat_index:
        return False
    if response_primitive(row) != args.current_primitive:
        raise ValueError(f"metadata/assistant label disagreement for {row.get('qid', '<unknown>')}")
    return True


def verify_images(row: dict[str, Any]) -> None:
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"selected row {row.get('qid', '<unknown>')} has no image paths")
    missing = [image for image in images if not isinstance(image, str) or not Path(image).is_file()]
    if missing:
        raise FileNotFoundError(f"selected row {row.get('qid', '<unknown>')} has missing image paths: {missing[:3]}")


def clone_row(row: dict[str, Any], copy_index: int) -> dict[str, Any]:
    clone = copy.deepcopy(row)
    qid = clone.get("qid")
    if not isinstance(qid, str) or not qid:
        raise ValueError("selected row has no qid")
    clone["qid"] = f"{qid}__v3_native_pour_boundary_copy{copy_index:02d}"
    metadata = clone["metadata"]
    metadata["augmentation"] = {
        "kind": "native_pick_to_first_pour_boundary_copy",
        "copy_index": copy_index,
        "source_qid": qid,
        "runtime_oracle_used": False,
    }
    return clone


def write_json_line(handle: Any, row: dict[str, Any], digest: hashlib._Hash) -> None:
    encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    handle.write(encoded)
    digest.update(encoded)


def main() -> int:
    args = parse_args()
    if args.copies < 1:
        raise ValueError("--copies must be at least one")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if args.source.resolve() == args.output.resolve():
        raise ValueError("--source and --output must differ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    manifest_tmp = args.manifest.with_name(f".{args.manifest.name}.tmp.{os.getpid()}")

    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    selected: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    source_rows = 0
    last_source_ended_newline = True

    try:
        with args.source.open("rb") as source_handle, output_tmp.open("wb") as output_handle:
            for line_number, raw in enumerate(source_handle, start=1):
                if not raw.strip():
                    raise ValueError(f"blank JSONL row at {args.source}:{line_number}")
                source_digest.update(raw)
                output_handle.write(raw)
                output_digest.update(raw)
                last_source_ended_newline = raw.endswith(b"\n")
                row = decode_json(raw, line_number, args.source)
                source_rows += 1
                qid = row.get("qid")
                if not isinstance(qid, str) or not qid:
                    raise ValueError(f"missing qid at {args.source}:{line_number}")
                if qid in seen_qids:
                    raise ValueError(f"duplicate base qid {qid}")
                seen_qids.add(qid)
                if is_canonical_boundary(row, args):
                    if args.verify_images:
                        verify_images(row)
                    selected.append(row)

            if not selected:
                raise ValueError("no canonical native pour-first boundary rows matched")
            if not last_source_ended_newline:
                output_handle.write(b"\n")
                output_digest.update(b"\n")

            for row in selected:
                for copy_index in range(1, args.copies + 1):
                    clone = clone_row(row, copy_index)
                    if clone["qid"] in seen_qids:
                        raise ValueError(f"duplicate augmented qid {clone['qid']}")
                    seen_qids.add(clone["qid"])
                    write_json_line(output_handle, clone, output_digest)

        added_rows = len(selected) * args.copies
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(args.source),
            "source_sha256": source_digest.hexdigest(),
            "source_rows": source_rows,
            "output": str(args.output),
            "output_sha256": output_digest.hexdigest(),
            "output_rows": source_rows + added_rows,
            "selection": {
                "task_id": args.task_id,
                "primitive_stem": args.primitive_stem,
                "current_primitive": args.current_primitive,
                "window_start": args.window_start,
                "repeat_index": args.repeat_index,
            },
            "selected_canonical_rows": len(selected),
            "selected_qids": [row["qid"] for row in selected],
            "copies_per_selected_row": args.copies,
            "added_rows": added_rows,
            "verify_images": args.verify_images,
            "runtime_oracle_used": False,
        }
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(output_tmp, args.output)
        os.replace(manifest_tmp, args.manifest)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise

    print(json.dumps({
        "source_rows": source_rows,
        "selected_canonical_rows": len(selected),
        "added_rows": len(selected) * args.copies,
        "output_rows": source_rows + len(selected) * args.copies,
        "source_sha256": source_digest.hexdigest(),
        "output_sha256": output_digest.hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
