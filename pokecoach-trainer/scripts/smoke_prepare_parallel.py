#!/usr/bin/env python3
"""Smoke local : prepare snapshot avec plusieurs valeurs de workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pokecoach_trainer.cli import _load_dataset_recipe, _resolve_prepare_execution
from pokecoach_trainer.dataset_service import create_dataset, get_dataset, submit_preparation
from pokecoach_trainer.jobs import get_job
from pokecoach_trainer.prepare_resources import detect_prepare_resources
from pokecoach_trainer.schemas import DatasetState, JobStatus, PrepareExecutionConfig
from pokecoach_trainer.settings import load_dotenv_explicit


def _parse_workers(values: list[str]) -> list[int | str]:
    out: list[int | str] = []
    for raw in values:
        if raw == "auto":
            out.append("auto")
        else:
            out.append(int(raw))
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv_explicit()
    parser = argparse.ArgumentParser(description="Smoke prepare parallèle")
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument(
        "--workers",
        nargs="+",
        default=["1", "2", "4", "auto"],
        help="valeurs demandées (ex. 1 2 4 auto)",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    resources = detect_prepare_resources()
    runs: list[dict] = []
    for workers in _parse_workers(args.workers):
        dataset_id = f"ds-smoke-{workers}-{int(time.time())}"
        recipe = _load_dataset_recipe(args.recipe)
        create_dataset(args.snapshot_id, recipe, dataset_id=dataset_id)
        execution = PrepareExecutionConfig(conversion_workers=workers)  # type: ignore[arg-type]
        t0 = time.perf_counter()
        job = submit_preparation(dataset_id, execution=execution)
        deadline = time.time() + 600
        while time.time() < deadline:
            record = get_job(job.job_id)
            if record.status in {JobStatus.completed, JobStatus.failed, JobStatus.cancelled}:
                break
            time.sleep(1)
        manifest = get_dataset(dataset_id)
        runs.append(
            {
                "workers_requested": workers,
                "prepare_execution": manifest.prepare_execution,
                "state": manifest.state.value,
                "duration_s": round(time.perf_counter() - t0, 2),
                "resources": resources.__dict__,
            }
        )

    report = {"runs": runs}
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
