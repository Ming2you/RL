from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Iterable, Sequence, TypeVar

from src.models.state import ExperimentConfig


T = TypeVar("T")
R = TypeVar("R")

_PROCESS_POOLS: dict[int, ProcessPoolExecutor] = {}


def _shutdown_process_pools() -> None:
    for executor in list(_PROCESS_POOLS.values()):
        executor.shutdown(wait=False, cancel_futures=True)
    _PROCESS_POOLS.clear()


atexit.register(_shutdown_process_pools)


@dataclass(frozen=True)
class GridParallelRun:
    results: list[Any]
    backend_used: str
    requested_backend: str
    workers: int
    chunks: int
    fallback_used: bool = False
    process_pool_reuse_enabled: bool = False
    process_pool_reused_existing: bool = False

    def diagnostics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_parallel_backend_serial": float(self.backend_used == "serial"),
            f"{prefix}_parallel_backend_thread": float(self.backend_used == "thread"),
            f"{prefix}_parallel_backend_process": float(self.backend_used == "process"),
            f"{prefix}_parallel_backend_fallback": float(self.fallback_used),
            f"{prefix}_parallel_workers": float(self.workers),
            f"{prefix}_parallel_chunks": float(self.chunks),
            f"{prefix}_parallel_process_pool_reuse_enabled": float(self.process_pool_reuse_enabled),
            f"{prefix}_parallel_process_pool_reused_existing": float(self.process_pool_reused_existing),
        }


def _chunked(items: Sequence[T], chunk_size: int) -> list[list[T]]:
    return [
        list(items[index:index + chunk_size])
        for index in range(0, len(items), chunk_size)
    ]


def _process_chunk(task: tuple[Callable[[Any], list[Any]], Any]) -> list[Any]:
    fn, payload = task
    return fn(payload)


def _flatten(chunks: Iterable[list[R]]) -> list[R]:
    out: list[R] = []
    for chunk in chunks:
        out.extend(chunk)
    return out


def _get_process_pool(workers: int) -> tuple[ProcessPoolExecutor, bool]:
    executor = _PROCESS_POOLS.get(workers)
    if executor is not None:
        return executor, True
    executor = ProcessPoolExecutor(max_workers=workers)
    _PROCESS_POOLS[workers] = executor
    return executor, False


def evaluate_grid_items(
    cfg: ExperimentConfig,
    items: Sequence[T],
    serial_fn: Callable[[T], R],
    *,
    process_chunk_fn: Callable[[Any], list[R]] | None = None,
    process_payloads: Sequence[Any] | None = None,
) -> GridParallelRun:
    """Evaluate grid candidates with a runtime-selected backend.

    The serial/thread path receives individual items and can close over the
    existing controller instance.  The process path uses explicit chunk payloads
    so Windows spawn can pickle a small top-level worker function.
    """
    requested = str(getattr(cfg.mpc, "grid_parallel_backend", "thread"))
    workers = max(1, int(getattr(cfg.mpc, "grid_parallel_max_workers", 1)))
    min_items = max(1, int(getattr(cfg.mpc, "grid_parallel_min_items", 1)))
    item_count = len(items)
    if item_count == 0:
        return GridParallelRun([], "serial", requested, 1, 0)
    if requested == "serial" or workers <= 1 or item_count < min_items:
        return GridParallelRun([serial_fn(item) for item in items], "serial", requested, 1, 1)
    workers = min(workers, item_count)
    if requested == "thread":
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return GridParallelRun(list(executor.map(serial_fn, items)), "thread", requested, workers, item_count)
    if requested == "process" and process_chunk_fn is not None and process_payloads is not None:
        chunks = list(process_payloads)
        if not chunks:
            return GridParallelRun([], "process", requested, workers, 0)
        process_workers = min(workers, len(chunks))
        reuse_enabled = bool(getattr(cfg.mpc, "grid_reuse_process_pool", True))
        if reuse_enabled:
            executor, reused_existing = _get_process_pool(process_workers)
            results = _flatten(executor.map(_process_chunk, [(process_chunk_fn, payload) for payload in chunks]))
        else:
            reused_existing = False
            with ProcessPoolExecutor(max_workers=process_workers) as executor:
                results = _flatten(executor.map(_process_chunk, [(process_chunk_fn, payload) for payload in chunks]))
        return GridParallelRun(
            results,
            "process",
            requested,
            process_workers,
            len(chunks),
            process_pool_reuse_enabled=reuse_enabled,
            process_pool_reused_existing=reused_existing,
        )

    return GridParallelRun(
        [serial_fn(item) for item in items],
        "serial",
        requested,
        1,
        1,
        fallback_used=True,
    )


def build_chunk_payloads(
    items: Sequence[T],
    *,
    static: dict[str, Any],
    item_key: str = "items",
    chunk_size: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    if not items:
        return []
    workers = max(1, min(max_workers, len(items)))
    target_chunk = max(1, chunk_size)
    # Avoid creating far more Windows spawn payloads than workers can consume.
    target_chunk = max(target_chunk, int(ceil(len(items) / max(workers * 2, 1))))
    return [
        {**static, item_key: chunk}
        for chunk in _chunked(list(items), target_chunk)
    ]
