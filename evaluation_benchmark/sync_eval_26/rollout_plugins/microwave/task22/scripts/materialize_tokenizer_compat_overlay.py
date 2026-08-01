#!/usr/bin/env python3
"""Create a private read-only checkpoint overlay for newer Transformers.

Task22's historical checkpoint stores ``extra_special_tokens`` as a list.  The
current training environment expects a mapping, while all tokens themselves are
already preserved in ``tokenizer.json``.  This tool leaves the source checkpoint
untouched, symlinks every source artifact, and replaces only the overlay's
``tokenizer_config.json`` field with an empty mapping after verifying that every
listed token is still present in ``tokenizer.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path.name}")
    return value


def validate_preserved_tokens(source: Path, tokens: list[str]) -> None:
    tokenizer_json = source / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise FileNotFoundError("source checkpoint is missing tokenizer.json")
    payload = read_json(tokenizer_json)
    added_tokens = payload.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise ValueError("tokenizer.json does not contain an added_tokens list")
    available = {
        item.get("content")
        for item in added_tokens
        if isinstance(item, dict) and isinstance(item.get("content"), str)
    }
    missing = sorted(set(tokens) - available)
    if missing:
        raise ValueError(f"tokenizer.json is missing configured special tokens: {missing}")


def normalize_rope_parameters(source: Path) -> tuple[dict[str, Any], bool]:
    config_path = source / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError("source checkpoint is missing config.json")
    model_config = read_json(config_path)
    text_config = model_config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError("model config is missing a text_config object")
    rope_parameters = text_config.get("rope_parameters")
    if rope_parameters is None:
        return model_config, False
    if not isinstance(rope_parameters, dict):
        raise ValueError("rope_parameters must be a JSON object")
    rope_theta = rope_parameters.get("rope_theta")
    if not isinstance(rope_theta, (int, float)):
        raise ValueError("rope_parameters is missing numeric rope_theta")
    rope_scaling = {key: value for key, value in rope_parameters.items() if key != "rope_theta"}
    if not rope_scaling:
        raise ValueError("rope_parameters has no rope scaling values")
    text_config["rope_theta"] = rope_theta
    text_config["rope_scaling"] = rope_scaling
    text_config.pop("rope_parameters")
    return model_config, True


def materialize(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"source checkpoint does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    source_config_path = source / "tokenizer_config.json"
    if not source_config_path.is_file():
        raise FileNotFoundError("source checkpoint is missing tokenizer_config.json")
    tokenizer_config = read_json(source_config_path)
    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list) or not all(
        isinstance(token, str) for token in extra_special_tokens
    ):
        raise ValueError("expected list[str] extra_special_tokens in historical tokenizer config")
    validate_preserved_tokens(source, extra_special_tokens)
    model_config, rope_parameters_normalized = normalize_rope_parameters(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for entry in source.iterdir():
        destination = output / entry.name
        if entry.name in {"tokenizer_config.json", "config.json"}:
            continue
        os.symlink(entry.resolve(), destination, target_is_directory=entry.is_dir())

    tokenizer_config["extra_special_tokens"] = {}
    overlay_config_path = output / "tokenizer_config.json"
    overlay_config_path.write_text(
        json.dumps(tokenizer_config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    overlay_model_config_path = output / "config.json"
    overlay_model_config_path.write_text(
        json.dumps(model_config, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "purpose": "transformers_extra_special_tokens_list_to_mapping_compatibility_overlay",
        "source_tokenizer_config_sha256": sha256(source_config_path),
        "overlay_tokenizer_config_sha256": sha256(overlay_config_path),
        "source_model_config_sha256": sha256(source / "config.json"),
        "overlay_model_config_sha256": sha256(overlay_model_config_path),
        "tokenizer_json_sha256": sha256(source / "tokenizer.json"),
        "source_extra_special_tokens_type": "list",
        "preserved_token_count": len(extra_special_tokens),
        "rope_parameters_normalized": rope_parameters_normalized,
        "source_entry_count": len(list(source.iterdir())),
        "source_checkpoint_modified": False,
    }
    (output / "compat_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize(args.source, args.output)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
