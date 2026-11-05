"""Pool ProcessPoolExecutor isolé pour la conversion."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pokecoach_trainer.conversion_worker import initialize_conversion_worker


@contextmanager
def conversion_pool(
    *,
    workers: int,
    storage_root: Path,
    vendor_root: Path,
    timeout_s: float,
) -> Iterator[ProcessPoolExecutor | None]:
    if workers <= 0:
        yield None
        return

    ctx = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=initialize_conversion_worker,
        initargs=(str(storage_root.resolve()), str(vendor_root.resolve()), float(timeout_s)),
    )
    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
