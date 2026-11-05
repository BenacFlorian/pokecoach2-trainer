#!/usr/bin/env python3
"""Benchmark synthétique catalogue collecte."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from pokecoach_trainer.archive_catalog import (
    catalog_has_replay,
    initialize_catalog,
    insert_replay_version,
)
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource, CollectionConfig


def _synthetic_ref(index: int) -> ArchiveRef:
    replay_id = f"gen9championsvgc2026regmc-bench{index}"
    content = f"hash-{index:08d}"
    return ArchiveRef(
        replay_id=replay_id,
        content_sha256=content,
        log_sha256=content,
        format_id="gen9championsvgc2026regmc",
        uploadtime=1_700_000_000 + index,
        object_relpath=f"objects/{replay_id}/{content}.json",
        conflict=False,
    )


def _bench_size(root: Path, size: int) -> dict[str, float]:
    initialize_catalog(root)
    source = ArchiveSource(provenance="benchmark", fetched_at=datetime.now(timezone.utc))
    latencies: list[float] = []
    batch = 1000
    started = time.perf_counter()
    for offset in range(0, size, batch):
        chunk = min(batch, size - offset)
        for i in range(offset, offset + chunk):
            ref = _synthetic_ref(i)
            t0 = time.perf_counter()
            insert_replay_version(
                root,
                ref=ref,
                source=source,
                provider_format=ref.format_id,
            )
            latencies.append(time.perf_counter() - t0)
    elapsed = time.perf_counter() - started
    sample = _synthetic_ref(size - 1)
    t_lookup = time.perf_counter()
    assert catalog_has_replay(root, sample.replay_id, sample.content_sha256)
    lookup_s = time.perf_counter() - t_lookup
    store = CollectionStore(root, clock=SystemClock())
    store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=10))
    p95 = (
        statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else 0.0
    )
    return {
        "inserted": float(size),
        "elapsed_s": elapsed,
        "lookup_s": lookup_s,
        "p50_insert_ms": statistics.median(latencies) * 1000,
        "p95_insert_ms": p95,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20000])
    parser.add_argument("--workers", nargs="+", type=int, default=[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/tmp/pokecoach-collection-bench"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {
        "sizes": args.sizes,
        "workers": args.workers,
        "runs": [],
        "note": "insertions catalogue synthétiques (sans logs volumineux)",
    }
    for size in args.sizes:
        run_root = args.root / f"n{size}"
        run_root.mkdir(parents=True, exist_ok=True)
        results["runs"].append({"size": size, "metrics": _bench_size(run_root, size)})
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
