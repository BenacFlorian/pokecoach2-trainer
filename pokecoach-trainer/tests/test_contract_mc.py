import pytest

from pokecoach_trainer.compatibility import build_contract, contracts_compatible
from pokecoach_trainer.schemas import ModelConfig


def test_build_contract_requires_model_config():
    with pytest.raises(TypeError):
        build_contract(None)  # type: ignore[arg-type]


def test_build_contract_stable_fingerprint():
    model = ModelConfig(d_model=256, num_layers=3, num_heads=4, embed_len=32)
    first = build_contract(model)
    second = build_contract(model)
    assert first.fingerprint == second.fingerprint
    assert first.act_len == second.act_len


def test_build_contract_architecture_affects_fingerprint():
    tiny = ModelConfig(d_model=128, num_layers=2, num_heads=4, embed_len=32)
    hist = ModelConfig(d_model=256, num_layers=3, num_heads=4, embed_len=32)
    assert build_contract(tiny).fingerprint != build_contract(hist).fingerprint


def test_contracts_compatible_rejects_permuted_architecture():
    """Dimensions proches mais architecture différente ⇒ empreintes distinctes."""

    a = build_contract(ModelConfig(d_model=256, num_layers=3, num_heads=4, embed_len=32))
    b = build_contract(ModelConfig(d_model=256, num_layers=3, num_heads=4, embed_len=16))
    assert not contracts_compatible(a, b)
