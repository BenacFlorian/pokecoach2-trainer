import json

import pytest

from pokecoach_trainer.base_model_service import (
    describe_policy,
    load_base_model,
    read_base_model_manifest,
    register_base_model,
)
from pokecoach_trainer.errors import ContractMismatch, InvalidConfiguration
from pokecoach_trainer.schemas import BaseModelSource, ModelConfig
from pokecoach_trainer.storage import resolve_inbox_relative, sha256_file, storage_path, write_json_atomic
from training_fixtures import install_storage_root, save_parent_checkpoint


def test_register_base_model_copies_inbox_without_mutating_source(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    source = save_parent_checkpoint(tmp_path, model, inbox_relpath="src.zip", base_model_id="mb-1")
    inbox = resolve_inbox_relative("src.zip")
    before = sha256_file(inbox)
    manifest = read_base_model_manifest("mb-1")
    assert manifest.checkpoint_sha256 == before
    assert manifest.architecture == model
    assert sha256_file(inbox) == before


def test_load_base_model_verifies_hash_and_architecture(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="mb-2")
    ppo = load_base_model("mb-2", "cpu")
    loaded = describe_policy(ppo.policy)
    assert loaded.model_dump() == model.model_dump()


def test_register_rejects_wrong_expected_hash(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="ignored", inbox_relpath="bad.zip")
    source = BaseModelSource(
        base_model_id="mb-bad",
        label="x",
        expected_sha256="0" * 64,
    )
    with pytest.raises(InvalidConfiguration):
        register_base_model("bad.zip", source)


def test_duplicate_register_rejected(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="mb-dup", inbox_relpath="dup.zip")
    source = BaseModelSource(
        base_model_id="mb-dup",
        label="x",
        expected_sha256=read_base_model_manifest("mb-dup").checkpoint_sha256,
    )
    with pytest.raises(InvalidConfiguration):
        register_base_model("dup.zip", source)


def test_tampered_architecture_in_manifest_rejected_on_load(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    hist = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, hist, base_model_id="mb-permute")
    manifest_path = storage_path("base_models", "mb-permute", "manifest.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["architecture"]["d_model"] = 128
    write_json_atomic(manifest_path, data)
    with pytest.raises(ContractMismatch):
        load_base_model("mb-permute", "cpu")
