import pytest

from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.training import build_ppo_container, count_policy_params, resolve_device


def test_model_configs_param_count():
    device = resolve_device("cpu")
    hist = ModelConfig(d_model=256, num_layers=3, num_heads=4, embed_len=32)
    tiny = ModelConfig(d_model=128, num_layers=2, num_heads=4, embed_len=32)
    ppo_hist = build_ppo_container(hist, device)
    ppo_tiny = build_ppo_container(tiny, device)
    assert count_policy_params(ppo_tiny.policy) < count_policy_params(ppo_hist.policy)


def test_invalid_model_config():
    with pytest.raises(ValueError):
        ModelConfig(d_model=128, num_heads=3)
