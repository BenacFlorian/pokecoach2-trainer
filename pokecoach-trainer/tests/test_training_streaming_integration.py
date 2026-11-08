"""Intégration train_bc sur petit dataset en mode flux."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.training import train_bc
from pokecoach_trainer.trajectory_dataset import load_trajectory_pickle
from training_fixtures import (
    experiment_config,
    install_storage_root,
    save_parent_checkpoint,
    write_ready_dataset_with_index,
)


def test_train_bc_streaming_small_dataset(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-stream")
    write_ready_dataset_with_index(
        "ds-stream",
        model,
        train_steps=[33],
        val_steps=[5],
        test_steps=[7],
        write_index=True,
    )
    config = experiment_config(
        "ds-stream",
        "parent-stream",
        model,
        batch_size=32,
        num_epochs=2,
        device="cpu",
        learning_rate=1e-3,
        evaluation_batch_size=4,
        shuffle_buffer_mib=64,
    )

    opened: list[str] = []

    def track_load(path):
        opened.append(Path(path).as_posix())
        return load_trajectory_pickle(path)

    monkeypatch.setattr(
        "pokecoach_trainer.training_data.load_trajectory_pickle",
        track_load,
    )

    with (
        patch("pokecoach_trainer.training.load_split_trajectories") as legacy_load,
        patch("imitation.data.rollout.flatten_trajectories") as flatten,
        patch.object(
            __import__("pokecoach_trainer.bc_adapter", fromlist=["CompleteBatchBC"]).CompleteBatchBC,
            "remember_trajectories",
            side_effect=AssertionError("remember_trajectories interdit"),
        ),
    ):
        legacy_load.side_effect = AssertionError("load_split_trajectories interdit")
        flatten.side_effect = AssertionError("flatten_trajectories interdit")
        result = train_bc(config, tmp_path / "run-stream")

    assert result.observed_transitions_per_epoch == [33, 33]
    assert result.observed_optimizer_steps == [2, 2]
    assert result.training_execution["sampler_version"] == "file_buffer_shuffle/1"
    assert Path(result.final_checkpoint).is_file()
    assert (tmp_path / "run-stream" / "result.json").is_file()
    metrics = (tmp_path / "run-stream" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics) == 2
    row0 = json.loads(metrics[0])
    assert row0["training_batch_size"] == 32
    assert row0["evaluation_batch_size"] == 4
    assert row0["shuffle_buffer_mib"] == 64

    test_marker = f"datasets/ds-stream/test/"
    assert len(opened) >= 6
    assert all(test_marker not in p for p in opened[:6])
