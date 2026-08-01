#!/usr/bin/env python3
"""Aggregate independently reset fixed-seed Task6 repeat workers.

The worker TSVs establish the independent episode denominator. Metrics are
always re-read from each run's original evaluator ``summary.tsv`` so empty
columns in a shell-produced TSV cannot corrupt a reported success rate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_COLUMNS = {
    "worker_id",
    "repeat_index",
    "seed",
    "exit_code",
    "stage_score_pct",
    "stage_success_rate",
    "goal_success_rate",
    "run_dir",
}


def read_rows(workers_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result_path in sorted(workers_root.glob("worker*/repeat_results.tsv")):
        with result_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or set(reader.fieldnames) != REQUIRED_COLUMNS:
                raise ValueError(f"unexpected repeat-result schema: {result_path}")
            for row in reader:
                summary_path = Path(row["run_dir"]) / "summary.tsv"
                if not summary_path.is_file():
                    raise ValueError(f"missing evaluator summary: {summary_path}")
                with summary_path.open(encoding="utf-8", newline="") as summary_handle:
                    summary_rows = list(csv.DictReader(summary_handle, delimiter="\t"))
                if not summary_rows:
                    raise ValueError(f"empty evaluator summary: {summary_path}")
                official = summary_rows[-1]
                for key in ("stage_score_pct", "stage_success_rate", "goal_success_rate"):
                    if key not in official:
                        raise ValueError(f"missing {key} in evaluator summary: {summary_path}")
                    row[key] = official[key]
                row["official_summary_path"] = str(summary_path)
                row["result_path"] = str(result_path)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--expected-seed", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.workers_root)
    indexes = [int(row["repeat_index"]) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != args.expected_episodes:
        raise SystemExit(f"expected {args.expected_episodes} episodes, found {len(rows)}")
    if sorted(indexes) != list(range(args.expected_episodes)):
        raise SystemExit(f"repeat indexes are not exactly 0..{args.expected_episodes - 1}: {indexes}")
    if any(seed != args.expected_seed for seed in seeds):
        raise SystemExit(f"unexpected seeds: {seeds}")

    stage_successes = sum(row["stage_success_rate"] == "1.0000" for row in rows)
    goal_successes = sum(row["goal_success_rate"] == "1.0000" for row in rows)
    clean_exits = sum(row["exit_code"] == "0" for row in rows)
    summary = {
        "episodes": len(rows),
        "seed": args.expected_seed,
        "clean_exits": clean_exits,
        "stage_successes": stage_successes,
        "stage_success_rate": stage_successes / len(rows),
        "goal_successes": goal_successes,
        "goal_success_rate": goal_successes / len(rows),
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (args.out_dir / "aggregate.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["repeat_index"])))

    print(
        f"Task6 seed={args.expected_seed}: {stage_successes}/{len(rows)} stage success, "
        f"{goal_successes}/{len(rows)} goal success, {clean_exits}/{len(rows)} clean exits"
    )


if __name__ == "__main__":
    main()
