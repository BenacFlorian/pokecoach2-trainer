from pathlib import Path
from unittest.mock import patch

import pytest

from pokecoach_trainer.base_model_service import load_base_model
from pokecoach_trainer.errors import InvalidConfiguration
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.storage import sha256_file, storage_path
from pokecoach_trainer.training import actor_weight_snapshot, train_bc
from training_fixtures import (
    experiment_config,
    install_storage_root,
    save_parent_checkpoint,
    write_ready_dataset,
)


def test_train_bc_starts_from_registered_parent(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-ft")
    write_ready_dataset("ds-ft", model, n_train=6, n_val=2, n_test=2, steps=3)
    parent_path = storage_path("base_models", "parent-ft", "model.zip")
    parent_hash_before = sha256_file(parent_path)

    parent_ppo = load_base_model("parent-ft", "cpu")
    parent_snapshot = actor_weight_snapshot(parent_ppo.policy)
    del parent_ppo

    config = experiment_config(
        "ds-ft",
        "parent-ft",
        model,
        batch_size=8,
        num_epochs=1,
        device="cpu",
        learning_rate=1e-3,
    )

    with patch("pokecoach_trainer.training.build_ppo_container") as mock_build:
        result = train_bc(config, tmp_path / "run-ft")
        mock_build.assert_not_called()

    loaded_after = load_base_model("parent-ft", "cpu")
    start_snapshot = actor_weight_snapshot(loaded_after.policy)
    for key in parent_snapshot:
        assert parent_snapshot[key].equal(start_snapshot[key])
    assert sha256_file(parent_path) == parent_hash_before
    assert result.effective_learning_rate == pytest.approx(1e-3)
    assert result.weight_delta_norm > 0
    assert result.actor_delta_norm and result.actor_delta_norm > 0
    assert result.parent_base_model_id == "parent-ft"


def test_train_bc_requires_initialization(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    write_ready_dataset("ds-no-init", model)
    config = experiment_config("ds-no-init", "missing", model)
    config.initialization = None
    with pytest.raises(InvalidConfiguration):
        train_bc(config, tmp_path / "run-x")
