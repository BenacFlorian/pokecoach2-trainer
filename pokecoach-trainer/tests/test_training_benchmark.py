"""Tests CLI du benchmark pipeline d'entraînement."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_training_pipeline import build_parser, validate_benchmark_args  # noqa: E402


def test_benchmark_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-id",
            "ds-x",
            "--base-model-id",
            "parent-x",
            "--output",
            "/tmp/bench-out.json",
        ],
    )
    assert args.max_files == 64
    assert args.batch_size == 32
    assert args.evaluation_batch_size == 256
    assert args.shuffle_buffer_mib == 256
    assert args.device == "cpu"


def test_benchmark_rejects_invalid_max_files(tmp_path, monkeypatch):
    from training_fixtures import install_storage_root

    install_storage_root(tmp_path, monkeypatch)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-id",
            "ds-x",
            "--base-model-id",
            "m",
            "--max-files",
            "0",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    with pytest.raises(ValueError, match="max-files"):
        validate_benchmark_args(args)


def test_benchmark_rejects_output_inside_dataset(tmp_path, monkeypatch):
    from pokecoach_trainer.settings import storage_path
    from training_fixtures import install_storage_root

    install_storage_root(tmp_path, monkeypatch)
    dataset_dir = storage_path("datasets", "ds-bench")
    dataset_dir.mkdir(parents=True)
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-id",
            "ds-bench",
            "--base-model-id",
            "m",
            "--output",
            str(dataset_dir / "leak.json"),
        ],
    )
    with pytest.raises(ValueError, match="output"):
        validate_benchmark_args(args)
