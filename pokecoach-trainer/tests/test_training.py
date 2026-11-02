import pickle
from pathlib import Path

import numpy as np
import torch
from imitation.data.types import Trajectory

from pokecoach_trainer.schemas import ExperimentConfig, ModelConfig, TrainingConfig
from pokecoach_trainer.training import (
    actor_weight_snapshot,
    build_ppo_container,
    evaluate_validation,
    train_bc,
)
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation


def _write_dataset(root: Path, dataset_id: str, n_train: int = 5, n_val: int = 2):
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    for split, n in (("train", n_train), ("validation", n_val)):
        d = root / "datasets" / dataset_id / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            steps = 2 + (i % 2)
            obs_rows = []
            for _ in range(steps):
                sample = ppo.observation_space.sample()
                obs_rows.append(np.zeros_like(sample["observation"], dtype=np.float32))
            sample = ppo.observation_space.sample()
            terminal = np.zeros_like(sample["observation"], dtype=np.float32)
            obs = np.stack(obs_rows + [terminal], axis=0)
            acts_arr = np.zeros((steps, 2), dtype=np.int64)
            traj = trajectory_to_imitation(
                Trajectory(obs=obs, acts=acts_arr, infos=None, terminal=True)
            )
            with (d / f"t{i}.pkl").open("wb") as f:
                pickle.dump(traj, f)


def test_validation_preserves_training_state_with_max_norm(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache

    clear_settings_cache()
    ds = "ds1"
    _write_dataset(tmp_path, ds, n_train=3, n_val=2)
    device = "cpu"
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, device)
    policy = ppo.policy
    demos = __import__(
        "pokecoach_trainer.training", fromlist=["load_split_trajectories"]
    ).load_split_trajectories(ds, "validation")
    actor_weight_snapshot(policy)
    pi = policy.pi_features_extractor
    with torch.no_grad():
        pi.ability_embed.weight[1].fill_(100.0)
    filled = actor_weight_snapshot(policy)
    evaluate_validation(policy, demos, device)
    after = actor_weight_snapshot(policy)
    for key in filled:
        assert torch.equal(filled[key], after[key])


def test_train_bc_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache

    clear_settings_cache()
    ds = "ds-smoke"
    _write_dataset(tmp_path, ds, n_train=4, n_val=2)
    config = ExperimentConfig(
        name="smoke",
        seed=42,
        dataset_id=ds,
        model=ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32),
        training=TrainingConfig(batch_size=8, num_epochs=1, device="cpu"),
    )
    result = train_bc(config, tmp_path / "run1")
    assert result.weight_delta_norm > 0
    assert Path(result.final_checkpoint).is_file()
