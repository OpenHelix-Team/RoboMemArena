#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_benchmark.sync_eval_26.validation import validate_episode  # noqa: E402
from evaluation_benchmark.sync_eval_26.profile_loader import load_profile  # noqa: E402


@dataclass(frozen=True)
class WorkItem:
    task_id: int
    episode_index: int
    attempt: int = 1


@dataclass
class ActiveWork:
    item: WorkItem
    slot: str
    process: subprocess.Popen
    log_stream: TextIO
    claim_stream: TextIO
    port_stream: TextIO
    port: int
    started_at: float


def parse_tasks(value: str) -> list[int]:
    if value == "all":
        return list(range(1, 27))
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            result.update(range(start, end + 1))
        elif item:
            result.add(int(item))
    if not result or min(result) < 1 or max(result) > 26:
        raise ValueError("tasks must be within 1..26")
    return sorted(result)


def parse_episode_indices(value: str, *, episodes: int) -> list[int]:
    """Parse a comma/range list of zero-based episode indices."""
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            result.update(range(start, end + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0 or max(result) >= episodes:
        raise ValueError(
            f"episode indices must be within 0..{episodes - 1}: {value!r}"
        )
    return sorted(result)


def discover_slots(
    raw: str | None = None,
    allocated_raw: str | None = None,
) -> list[str]:
    if allocated_raw is None:
        allocated_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not allocated_raw.strip():
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is empty; run scheduler inside a GPU allocation"
        )
    value = raw if raw is not None else allocated_raw
    slots = [item.strip() for item in value.split(",") if item.strip()]
    if not slots:
        raise RuntimeError("no GPU slots; set --slots or CUDA_VISIBLE_DEVICES")
    if len(set(slots)) != len(slots):
        raise RuntimeError(f"duplicate GPU slots requested: {slots}")
    if raw is not None:
        allocated = {item.strip() for item in allocated_raw.split(",") if item.strip()}
        invalid = [slot for slot in slots if slot not in allocated]
        if invalid:
            raise RuntimeError(
                f"requested GPU slots outside CUDA_VISIBLE_DEVICES: {invalid}"
            )
    return slots


def discover_gpu_bundles(
    raw: str | None = None,
    allocated_raw: str | None = None,
) -> list[tuple[str, ...]]:
    if allocated_raw is None:
        allocated_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allocated = tuple(
        item.strip() for item in allocated_raw.split(",") if item.strip()
    )
    if not allocated:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is empty; run scheduler inside a GPU allocation"
        )
    if len(set(allocated)) != len(allocated):
        raise RuntimeError(f"duplicate allocated GPU devices: {allocated}")
    if raw is None:
        return [allocated]

    bundles: list[tuple[str, ...]] = []
    for raw_bundle in raw.split(";"):
        devices = tuple(item.strip() for item in raw_bundle.split(",") if item.strip())
        if not devices:
            raise RuntimeError(f"empty GPU bundle in --gpu-bundles={raw!r}")
        if len(set(devices)) != len(devices):
            raise RuntimeError(f"duplicate GPU device in bundle: {devices}")
        invalid = [device for device in devices if device not in allocated]
        if invalid:
            raise RuntimeError(
                "requested GPU devices outside CUDA_VISIBLE_DEVICES: "
                f"{invalid}"
            )
        bundles.append(devices)
    flattened = [device for bundle in bundles for device in bundle]
    if len(set(flattened)) != len(flattened):
        raise RuntimeError(f"GPU devices appear in multiple bundles: {bundles}")
    return bundles


def assert_task_capacity(
    *,
    task_ids: list[int],
    bundles: list[tuple[str, ...]],
    profile_dir: Path,
) -> None:
    for task_id in task_ids:
        profile = load_profile(profile_dir / f"task{task_id:02d}.yaml")
        if not any(
            len(bundle) == profile.runtime_topology.allocation_gpus
            for bundle in bundles
        ):
            raise RuntimeError(
                f"no compatible GPU bundle for task {task_id}: "
                f"requires {profile.runtime_topology.allocation_gpus}, "
                f"bundles={bundles}"
            )


def build_work_queue(
    task_ids: list[int], episode_indices: list[int]
) -> deque[WorkItem]:
    return deque(
        WorkItem(task_id=task_id, episode_index=episode)
        for task_id in task_ids
        for episode in episode_indices
    )


def episode_dir(output_root: Path, item: WorkItem) -> Path:
    return output_root / f"task{item.task_id:02d}" / f"ep{item.episode_index:03d}"


def _archive_invalid(output_root: Path, item: WorkItem) -> None:
    source = episode_dir(output_root, item)
    if not source.exists():
        return
    archive = (
        output_root / "_invalid" / f"task{item.task_id:02d}_ep{item.episode_index:03d}_"
        f"attempt{item.attempt}_{time.time_ns()}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    source.replace(archive)


def _port_base() -> int:
    job_id = int(os.environ.get("SLURM_JOB_ID", "0").split("_", 1)[0] or 0)
    return 20000 + (job_id % 1000) * 32


def _try_lock(path: Path) -> TextIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def _release_lock(stream: TextIO) -> None:
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _claim_episode(output_root: Path, item: WorkItem) -> TextIO | None:
    claim_path = (
        output_root
        / "_claims"
        / f"task{item.task_id:02d}_ep{item.episode_index:03d}.lock"
    )
    stream = _try_lock(claim_path)
    if stream is None:
        return None
    stream.seek(0)
    stream.truncate()
    stream.write(
        f"pid={os.getpid()} task={item.task_id} "
        f"episode={item.episode_index} attempt={item.attempt}\n"
    )
    stream.flush()
    return stream


def _claim_slot(
    slot: str,
    *,
    lock_root: Path = Path("/tmp"),
    job_id: str | None = None,
    hostname: str | None = None,
    namespace: str | None = None,
) -> TextIO | None:
    allocation = job_id or os.environ.get("SLURM_JOB_ID", "local")
    node = hostname or socket.gethostname()
    # Two overlapping Slurm steps in one holder receive disjoint physical GPUs
    # but both can expose them locally as CUDA 0,1. Keep the physical-bundle
    # lock for one step while separating those distinct Slurm steps.
    step_id = os.environ.get("SLURM_STEP_ID", "shared")
    identity = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", f"{allocation}_{step_id}_{node}_{slot}"
    )
    stream = _try_lock(lock_root / f"robomemarena_sync_eval_gpu_v2_{identity}.lock")
    if stream is None:
        return None
    stream.seek(0)
    stream.truncate()
    stream.write(
        f"pid={os.getpid()} allocation={allocation} node={node} slot={slot} "
        f"namespace={namespace or ''}\n"
    )
    stream.flush()
    return stream


def resolve_slot_lock_namespace(namespace: str | None) -> str | None:
    """Return an optional per-worker identity for shared Slurm allocations."""
    return namespace or os.environ.get("SYNC_EVAL_SLOT_LOCK_NAMESPACE") or None


def _reserve_port(preferred: int) -> tuple[int, TextIO]:
    for offset in range(2048):
        port = 20000 + ((preferred - 20000 + offset) % 40000)
        port_stream = _try_lock(
            Path("/tmp") / f"robomemarena_sync_eval_port_{port}.lock"
        )
        if port_stream is None:
            continue
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            _release_lock(port_stream)
            continue
        finally:
            probe.close()
        return port, port_stream
    raise RuntimeError(f"unable to reserve a port near {preferred}")


def _terminate_process_group(process: subprocess.Popen) -> None:
    process_group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    if not group_exists():
        process.poll()
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 20
    while group_exists() and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.1)
    if group_exists():
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 20
        while group_exists() and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.1)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _release_work_resources(work: ActiveWork) -> None:
    work.log_stream.close()
    _release_lock(work.port_stream)
    _release_lock(work.claim_stream)


def main() -> None:
    os.umask(0o002)
    repo_root_default = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--episode-indices",
        help="zero-based comma/range list; default runs 0 through --episodes-1",
    )
    parser.add_argument("--slots")
    parser.add_argument("--gpu-bundles")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--slot-lock-namespace")
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--episode-timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.profile_dir = (
        args.profile_dir
        or args.repo_root / "evaluation_benchmark" / "sync_eval_26" / "profiles"
    ).resolve()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    scheduler_logs = args.output_root / "_scheduler"
    scheduler_logs.mkdir(parents=True, exist_ok=True)

    if args.slots is not None and args.gpu_bundles is not None:
        parser.error("--slots and --gpu-bundles cannot be used together")
    task_ids = parse_tasks(args.tasks)
    try:
        episode_indices = (
            parse_episode_indices(args.episode_indices, episodes=args.episodes)
            if args.episode_indices
            else list(range(args.episodes))
        )
    except ValueError as exc:
        parser.error(str(exc))
    allocated_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.gpu_bundles is not None:
        bundles = discover_gpu_bundles(args.gpu_bundles, allocated_raw=allocated_raw)
    elif args.slots is not None:
        bundles = [(slot,) for slot in discover_slots(args.slots, allocated_raw)]
    else:
        bundles = discover_gpu_bundles(allocated_raw=allocated_raw)
    assert_task_capacity(
        task_ids=task_ids,
        bundles=bundles,
        profile_dir=args.profile_dir,
    )
    pending = build_work_queue(task_ids, episode_indices)
    filtered: deque[WorkItem] = deque()
    while pending:
        item = pending.popleft()
        validation = validate_episode(
            repo_root=args.repo_root,
            profile_path=args.profile_dir / f"task{item.task_id:02d}.yaml",
            episode_dir=episode_dir(args.output_root, item),
        )
        if validation.valid:
            continue
        else:
            filtered.append(item)
    pending = filtered

    active: dict[str, ActiveWork] = {}
    failures: list[tuple[WorkItem, str]] = []
    stopping = False

    def stop_children(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        for work in active.values():
            try:
                os.killpg(work.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    port_base = args.port_base if args.port_base is not None else _port_base()
    slot_lock_namespace = resolve_slot_lock_namespace(args.slot_lock_namespace)

    while (pending or active) and not stopping:
        for bundle_index, bundle in enumerate(bundles):
            slot = ",".join(bundle)
            if slot in active or not pending:
                continue
            # The namespace identifies a scheduler, not physical GPU ownership.
            # All overlapping schedulers in one Slurm allocation must contend on
            # the same allocation/node/bundle lock.
            slot_stream = _claim_slot(slot, namespace=slot_lock_namespace)
            if slot_stream is None:
                continue
            item = pending.popleft()
            profile_path = args.profile_dir / f"task{item.task_id:02d}.yaml"
            current = validate_episode(
                repo_root=args.repo_root,
                profile_path=profile_path,
                episode_dir=episode_dir(args.output_root, item),
            )
            if current.valid:
                _release_lock(slot_stream)
                continue
            claim_stream = _claim_episode(args.output_root, item)
            if claim_stream is None:
                _release_lock(slot_stream)
                pending.append(item)
                continue
            current = validate_episode(
                repo_root=args.repo_root,
                profile_path=profile_path,
                episode_dir=episode_dir(args.output_root, item),
            )
            if current.valid:
                _release_lock(claim_stream)
                _release_lock(slot_stream)
                continue
            _archive_invalid(args.output_root, item)
            try:
                port, port_stream = _reserve_port(port_base + bundle_index)
            except Exception:
                _release_lock(claim_stream)
                _release_lock(slot_stream)
                raise
            log_path = (
                scheduler_logs / f"task{item.task_id:02d}_ep{item.episode_index:03d}_"
                f"attempt{item.attempt}.log"
            )
            try:
                stream = log_path.open("w", encoding="utf-8")
            except Exception:
                _release_lock(port_stream)
                _release_lock(claim_stream)
                _release_lock(slot_stream)
                raise
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": slot,
                    "PORT": str(port),
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONPATH": (f"{args.repo_root}:{env.get('PYTHONPATH', '')}"),
                }
            )
            command = [
                sys.executable,
                "-m",
                "evaluation_benchmark.sync_eval_26.eval_sync_vlm_vla_1_26",
                "--repo-root",
                str(args.repo_root),
                "--profile",
                str(profile_path),
                "--episode-start",
                str(item.episode_index),
                "--num-episodes",
                "1",
                "--output-root",
                str(args.output_root),
            ]
            try:
                process = subprocess.Popen(
                    command,
                    cwd=args.repo_root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(
                        claim_stream.fileno(),
                        port_stream.fileno(),
                        slot_stream.fileno(),
                    ),
                )
            except Exception:
                stream.close()
                _release_lock(port_stream)
                _release_lock(claim_stream)
                _release_lock(slot_stream)
                raise
            # The worker inherits the lock FD. Closing only the scheduler's copy
            # prevents a stopped scheduler from pinning the bundle after its
            # worker exits, while the live worker still owns the lock.
            slot_stream.close()
            active[slot] = ActiveWork(
                item=item,
                slot=slot,
                process=process,
                log_stream=stream,
                claim_stream=claim_stream,
                port_stream=port_stream,
                port=port,
                started_at=time.monotonic(),
            )
            print(
                f"START slot={slot} task={item.task_id} "
                f"ep={item.episode_index} attempt={item.attempt} port={port}",
                flush=True,
            )

        time.sleep(args.poll_seconds)
        for slot, work in list(active.items()):
            timed_out = (
                time.monotonic() - work.started_at > args.episode_timeout_seconds
            )
            if timed_out:
                _terminate_process_group(work.process)
            return_code = work.process.poll()
            if return_code is None:
                continue
            _terminate_process_group(work.process)
            _release_work_resources(work)
            validation = validate_episode(
                repo_root=args.repo_root,
                profile_path=args.profile_dir / f"task{work.item.task_id:02d}.yaml",
                episode_dir=episode_dir(args.output_root, work.item),
            )
            if return_code == 0 and validation.valid:
                print(
                    f"DONE slot={slot} task={work.item.task_id} "
                    f"ep={work.item.episode_index}",
                    flush=True,
                )
            elif work.item.attempt < args.max_attempts:
                retry = WorkItem(
                    work.item.task_id,
                    work.item.episode_index,
                    work.item.attempt + 1,
                )
                pending.appendleft(retry)
                print(
                    f"RETRY slot={slot} task={work.item.task_id} "
                    f"ep={work.item.episode_index} rc={return_code} "
                    f"reason={'episode_timeout' if timed_out else validation.reason}",
                    flush=True,
                )
            else:
                reason = "episode_timeout" if timed_out else validation.reason
                failures.append((work.item, reason))
                print(
                    f"FAILED slot={slot} task={work.item.task_id} "
                    f"ep={work.item.episode_index} rc={return_code} "
                    f"reason={reason}",
                    flush=True,
                )
            del active[slot]

    if stopping:
        for work in active.values():
            _terminate_process_group(work.process)
            _release_work_resources(work)
        raise SystemExit(130)
    if failures:
        for item, reason in failures:
            print(
                f"UNRESOLVED task={item.task_id} ep={item.episode_index} "
                f"reason={reason}",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print(
        f"COMPLETE tasks={len(task_ids)} episodes={len(episode_indices)} "
        f"valid={len(task_ids) * len(episode_indices)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
