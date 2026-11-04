
import pytest
from pydantic import ValidationError

from pokecoach_trainer.formats import resolve_format
from pokecoach_trainer.schemas import (
    CollectionConfig,
    DatasetRecipe,
    ExperimentConfig,
    InitializationConfig,
    TrainingConfig,
    adapt_legacy_manifest,
)


def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.method == "bc"
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.evaluation_battles_per_epoch == 0


def test_training_config_rejects_nan_lr():
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=float("nan"))


def test_training_config_rejects_unimplemented_method():
    with pytest.raises(ValueError):
        TrainingConfig.model_validate({"method": "ppo", "batch_size": 8, "num_epochs": 1})


def test_dataset_recipe_spec_defaults():
    recipe = DatasetRecipe()
    assert recipe.variant == "natural"
    assert recipe.regulation == "M-C"
    assert recipe.target_replays == 300
    assert recipe.min_elo == 1100
    assert recipe.splits.train == pytest.approx(0.80)


def test_dataset_recipe_rejects_diverse_variant():
    with pytest.raises(ValueError):
        DatasetRecipe.model_validate({"variant": "diverse"})


def test_dataset_recipe_rejects_team_cap():
    with pytest.raises(ValueError):
        DatasetRecipe(max_replays_per_team=2)


def test_dataset_recipe_alias_contradiction():
    with pytest.raises(ValueError):
        DatasetRecipe(min_elo=1200, min_rating=1100)


def test_collection_config_bounds():
    cfg = CollectionConfig(format_ids=["mc"])
    assert cfg.max_new_replays == 5000
    assert cfg.max_attempts == 5


def test_initialization_requires_base_model():
    init = InitializationConfig(base_model_id="mb-parent")
    assert init.architecture == "inherit"


def test_experiment_backward_compat_model_field():
    exp = ExperimentConfig(name="x", dataset_id="ds1")
    assert exp.model.d_model == 128
    assert exp.training.learning_rate == pytest.approx(1e-4)


def test_initialization_architecture_conflict():
    with pytest.raises(ValidationError):
        InitializationConfig.model_validate(
            {"base_model_id": "m1", "architecture": "tiny"}
        )


def test_mc_format_registry_champions_regmc_not_regf():
    entry = resolve_format("mc")
    assert entry.regulation == "M-C"
    assert entry.engine_format_id == "gen9championsvgc2026regmc"
    assert "gen9championsvgc2026regmc" in entry.source_format_ids
    assert "gen9championsvgc2026regmcbo3" in entry.source_format_ids
    assert not any("regf" in sid.lower() for sid in entry.source_format_ids)


def test_unknown_format():
    from pokecoach_trainer.errors import NotFound

    with pytest.raises(NotFound):
        resolve_format("unknown-format-xyz")


def test_legacy_manifest_adapter():
    data = adapt_legacy_manifest({"dataset_id": "d1", "state": "imported"})
    assert data["schema_version"] == 1
