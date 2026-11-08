"""Tests TrainingConfig streaming / évaluation (tâche 1 du plan 2026-11-08)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pokecoach_trainer.cli import _load_experiment
from pokecoach_trainer.schemas import TrainingConfig
from pokecoach_trainer.settings import PACKAGE_ROOT


def test_training_config_shuffle_and_eval_defaults():
    cfg = TrainingConfig()
    assert cfg.shuffle_buffer_mib == 256
    assert cfg.evaluation_batch_size == 256


@pytest.mark.parametrize(
    "field,value",
    [
        ("shuffle_buffer_mib", True),
        ("shuffle_buffer_mib", 0),
        ("shuffle_buffer_mib", -1),
        ("shuffle_buffer_mib", 1.5),
        ("shuffle_buffer_mib", "256"),
        ("evaluation_batch_size", True),
        ("evaluation_batch_size", 0),
        ("evaluation_batch_size", -4),
        ("evaluation_batch_size", 3.0),
        ("evaluation_batch_size", "128"),
    ],
)
def test_training_config_rejects_invalid_streaming_fields(field: str, value: object):
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({field: value})


def test_shuffle_buffer_mib_bounds():
    TrainingConfig(shuffle_buffer_mib=1)
    TrainingConfig(shuffle_buffer_mib=4096)
    with pytest.raises(ValidationError):
        TrainingConfig(shuffle_buffer_mib=4097)


def test_training_config_json_roundtrip():
    raw = TrainingConfig(
        shuffle_buffer_mib=512,
        evaluation_batch_size=128,
        batch_size=16,
        num_epochs=2,
    ).model_dump(mode="json")
    payload = json.dumps(raw)
    restored = TrainingConfig.model_validate(json.loads(payload))
    assert restored.shuffle_buffer_mib == 512
    assert restored.evaluation_batch_size == 128


def test_load_finetune_mb_mc_prod_yaml():
    path = PACKAGE_ROOT / "configs" / "finetune-mb-mc-prod.yaml"
    config = _load_experiment(path)
    assert config.dataset_id == "ds-mb-mc-collectable-80k-v1"
    assert config.initialization is not None
    assert config.initialization.base_model_id == "vgcbench-mb-reference"
    assert config.training.batch_size == 32
    assert config.training.num_epochs == 3
    assert config.training.learning_rate == pytest.approx(0.0001)
    assert config.training.shuffle_buffer_mib == 256
    assert config.training.evaluation_batch_size == 256
