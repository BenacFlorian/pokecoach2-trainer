#!/usr/bin/env python3
"""Benchmark lecture bornée du pipeline d'entraînement BC en flux."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_PACKAGE_ROOT / "vendor" / "vgc-bench")

# ruff: noqa: E402
from pokecoach_trainer.bc_evaluation import evaluate_nll_batches
from pokecoach_trainer.dataset_service import get_dataset
from pokecoach_trainer.schemas import DatasetState
from pokecoach_trainer.settings import get_settings, storage_path
from pokecoach_trainer.training import load_base_model
from pokecoach_trainer.training_data import (
    SAMPLER_VERSION,
    ShuffledTransitionBatches,
    build_split_source,
    iter_source_trajectories,
    iter_transition_batches,
)
from pokecoach_trainer.upstream import vgc_bench_workdir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark flux BC / évaluation par batches")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--shuffle-buffer-mib", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def validate_benchmark_args(args: argparse.Namespace) -> None:
    if args.max_files < 1:
        raise ValueError("--max-files doit être >= 1")
    root = get_settings().storage_root.resolve()
    out = args.output.resolve()
    dataset_root = (root / "datasets" / args.dataset_id).resolve()
    parent_root = (root / "base_models" / args.base_model_id).resolve()
    for protected in (dataset_root, parent_root):
        if protected in out.parents or out == protected:
            raise ValueError(f"--output ne doit pas être dans {protected}")


def _subset_source(dataset_id: str, max_files: int):
    manifest = get_dataset(dataset_id)
    if manifest.state != DatasetState.ready:
        raise ValueError("dataset non ready")
    dataset_dir = storage_path("datasets", dataset_id)
    full = build_split_source(dataset_dir, manifest, "train")
    take = min(max_files, len(full.files))
    files = full.files[:take]
    expected = 0
    from pokecoach_trainer.trajectory_dataset import load_trajectory_pickle

    for path in files:
        expected += len(load_trajectory_pickle(path).acts)
    from pokecoach_trainer.training_data import TrajectorySplitSource

    return TrajectorySplitSource(split="train", files=files, expected_transitions=expected)


def run_benchmark(args: argparse.Namespace) -> dict:
    validate_benchmark_args(args)
    source = _subset_source(args.dataset_id, args.max_files)
    load_t0 = time.perf_counter()
    _ = list(iter_source_trajectories(source))
    load_s = time.perf_counter() - load_t0

    with vgc_bench_workdir():
        ppo = load_base_model(args.base_model_id, args.device)
        policy = ppo.policy

        eval_batches = iter_transition_batches(
            iter_source_trajectories(source),
            batch_size=args.evaluation_batch_size,
        )
        eval_t0 = time.perf_counter()
        batched = evaluate_nll_batches(policy, eval_batches, args.device)
        eval_s = time.perf_counter() - eval_t0

        shuffle = ShuffledTransitionBatches(
            source,
            batch_size=args.batch_size,
            shuffle_buffer_mib=args.shuffle_buffer_mib,
            seed=42,
            epoch=1,
        )
        shuffle_t0 = time.perf_counter()
        count = sum(b["acts"].shape[0] for b in shuffle)
        shuffle_s = time.perf_counter() - shuffle_t0
    if count != source.expected_transitions:
        raise ValueError("comptage transitions benchmark incohérent")

    return {
        "dataset_id": args.dataset_id,
        "base_model_id": args.base_model_id,
        "device": args.device,
        "max_files": args.max_files,
        "files_used": len(source.files),
        "transitions": source.expected_transitions,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "shuffle_buffer_mib": args.shuffle_buffer_mib,
        "sampler_version": SAMPLER_VERSION,
        "policy_eval_calls": math.ceil(source.expected_transitions / args.evaluation_batch_size),
        "batched_nll": batched.nll,
        "durations_s": {
            "load_all_files": load_s,
            "batched_eval": eval_s,
            "shuffle_pass": shuffle_s,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_benchmark_args(args)
        report = run_benchmark(args)
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
