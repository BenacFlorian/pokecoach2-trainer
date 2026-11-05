"""Contrat PrepareExecutionConfig et propagation job."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pokecoach_trainer.schemas import (
    DatasetJobConfig,
    DatasetRecipe,
    PrepareConfig,
    PrepareExecutionConfig,
)


def test_prepare_defaults_to_auto():
    cfg = PrepareExecutionConfig()
    assert cfg.conversion_workers == "auto"
    assert cfg.conversion_timeout_s == 120.0


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, 1.5, "4"],
)
def test_workers_reject_bool_zero_negative_float(value):
    with pytest.raises(ValidationError):
        PrepareExecutionConfig(conversion_workers=value)  # type: ignore[arg-type]


def test_workers_accepts_positive_int():
    cfg = PrepareExecutionConfig(conversion_workers=4)
    assert cfg.conversion_workers == 4


def test_old_job_without_execution_loads():
    raw = {
        "kind": "prepare",
        "dataset_id": "ds-old",
        "dataset_source": {"source": "snapshot", "snapshot_id": "snap-1"},
        "recipe": DatasetRecipe(target_replays=1).model_dump(mode="json"),
    }
    cfg = DatasetJobConfig.model_validate(raw)
    assert cfg.execution.conversion_workers == "auto"
    assert cfg.execution.conversion_timeout_s == 120.0


def test_prepare_config_carries_execution_defaults():
    legacy = PrepareConfig(seed=7)
    assert legacy.conversion_workers == "auto"
    assert legacy.execution().conversion_timeout_s == 120.0


def test_recipe_sha256_unchanged_by_execution():
    from pokecoach_trainer.dataset_service import _recipe_sha256

    recipe = DatasetRecipe(target_replays=10, seed=99)
    base = _recipe_sha256(recipe)
    assert base == _recipe_sha256(recipe)


def test_timeout_must_be_positive_finite():
    with pytest.raises(ValidationError):
        PrepareExecutionConfig(conversion_timeout_s=0)
    with pytest.raises(ValidationError):
        PrepareExecutionConfig(conversion_timeout_s=float("nan"))
