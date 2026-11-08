import numpy as np
import torch

from pokecoach_trainer.bc_adapter import duplicate_policy, evaluate_nll
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.training import (
    actor_weight_snapshot,
    build_ppo_container,
    load_split_trajectories,
)
from training_fixtures import install_storage_root, save_parent_checkpoint, write_ready_dataset


def test_evaluation_isolation_preserves_training_state(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-eval")
    write_ready_dataset("ds-eval", model, n_train=2, n_val=2, steps=2)
    device = "cpu"
    ppo = build_ppo_container(model, device)
    policy = ppo.policy
    demos = load_split_trajectories("ds-eval", "validation")
    pi = policy.pi_features_extractor
    with torch.no_grad():
        pi.ability_embed.weight[1].fill_(100.0)
    filled = actor_weight_snapshot(policy)
    result = evaluate_nll(policy, demos, device)
    assert result.transitions > 0
    assert np.isfinite(result.nll)
    after = actor_weight_snapshot(policy)
    for key in filled:
        assert torch.equal(filled[key], after[key])


def test_duplicate_policy_placed_on_device():
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    clone = duplicate_policy(ppo.policy, "cpu")
    before = actor_weight_snapshot(ppo.policy)
    after = actor_weight_snapshot(clone)
    for key in before:
        assert torch.equal(before[key], after[key])
