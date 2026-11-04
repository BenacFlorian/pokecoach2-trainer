import pickle
from pathlib import Path

import numpy as np
import torch
from imitation.data.types import Trajectory

from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.training import (
    actor_weight_snapshot,
    build_ppo_container,
    evaluate_validation,
    train_bc,
)
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation
from training_fixtures import (
    experiment_config,
    install_storage_root,
    save_parent_checkpoint,
    write_ready_dataset,
)


def test_validation_preserves_training_state_with_max_norm(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    write_ready_dataset("ds1", model, n_train=3, n_val=2, steps=2)
    device = "cpu"
    ppo = build_ppo_container(model, device)
    policy = ppo.policy
    demos = __import__(
        "pokecoach_trainer.training", fromlist=["load_split_trajectories"]
    ).load_split_trajectories("ds1", "validation")
    pi = policy.pi_features_extractor
    with torch.no_grad():
        pi.ability_embed.weight[1].fill_(100.0)
    filled = actor_weight_snapshot(policy)
    try:
        evaluate_validation(policy, demos, device)
    except ValueError:
        pass
    after = actor_weight_snapshot(policy)
    for key in filled:
        assert torch.equal(filled[key], after[key])


def test_train_bc_smoke(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-smoke")
    write_ready_dataset("ds-smoke", model, n_train=8, n_val=2, n_test=2, steps=3)
    config = experiment_config(
        "ds-smoke",
        "parent-smoke",
        model,
        batch_size=8,
        num_epochs=1,
        device="cpu",
        learning_rate=1e-3,
    )
    result = train_bc(config, tmp_path / "run1")
    assert result.weight_delta_norm > 0
    assert Path(result.final_checkpoint).is_file()
