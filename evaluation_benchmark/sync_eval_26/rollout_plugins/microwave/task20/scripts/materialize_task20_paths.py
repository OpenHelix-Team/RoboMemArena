#!/usr/bin/env python3
"""Resolve Task20 robot-anchor HDF paths without embedding machine paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.template.read_text(encoding="utf-8"))
    token = "${TASK20_DATA_ROOT}/"
    for anchor in payload["tasks"]["20"]:
        value = anchor.get("anchor_hdf5")
        if not isinstance(value, str) or not value.startswith(token):
            raise ValueError(f"unexpected anchor_hdf5: {value!r}")
        anchor["anchor_hdf5"] = str(args.data_root / value.removeprefix(token))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
