#!/usr/bin/env python3
"""Resolve the Task21 anchor-data path without embedding a machine-specific path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def materialize_template(value: object, data_root: Path) -> object:
    """Resolve task-local HDF placeholders in legacy or task-indexed rules."""

    token = "${TASK21_DATA_ROOT}"

    def materialize_rule(anchor: object) -> None:
        if not isinstance(anchor, dict):
            raise ValueError(f"release-anchor rule must be an object: {anchor!r}")
        raw_value = anchor.get("anchor_hdf5")
        if raw_value is None:
            return
        if not isinstance(raw_value, str) or not raw_value.startswith(f"{token}/"):
            raise ValueError(f"invalid anchor_hdf5 template: {raw_value!r}")
        anchor["anchor_hdf5"] = str(data_root / raw_value.removeprefix(f"{token}/"))

    if isinstance(value, list):
        for anchor in value:
            materialize_rule(anchor)
        return value

    if not isinstance(value, dict):
        raise ValueError("release-anchor template must be a list or object")
    payload = value.get("tasks", value)
    if not isinstance(payload, dict):
        raise ValueError("release-anchor object must contain a 'tasks' object")
    for task_key, rules in payload.items():
        if not isinstance(rules, list):
            raise ValueError(f"release-anchor rules for task {task_key!r} must be a list")
        for anchor in rules:
            materialize_rule(anchor)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    anchors = materialize_template(
        json.loads(args.template.read_text(encoding="utf-8")),
        args.data_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(anchors, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
