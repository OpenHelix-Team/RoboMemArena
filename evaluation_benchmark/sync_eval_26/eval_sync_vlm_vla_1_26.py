from __future__ import annotations

import argparse
from pathlib import Path

from .dispatcher import dispatch_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dispatch_range(
        repo_root=args.repo_root,
        profile_path=args.profile,
        episode_indices=range(
            args.episode_start, args.episode_start + args.num_episodes
        ),
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
