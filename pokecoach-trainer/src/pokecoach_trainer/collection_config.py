"""Chargement des configurations de collecte depuis YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from pokecoach_trainer.errors import InvalidConfiguration
from pokecoach_trainer.replay_policy import filters_from_recipe
from pokecoach_trainer.schemas import CollectionConfig, DatasetRecipe, ReplayFilterConfig
from pokecoach_trainer.settings import PACKAGE_ROOT


def resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = PACKAGE_ROOT / path
    if candidate.is_file() or candidate.is_dir():
        return candidate
    return path


def load_dataset_recipe_file(path: Path) -> DatasetRecipe:
    path = resolve_config_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("dataset", data)
    return DatasetRecipe.model_validate(block)


def load_collection_config(path: Path) -> CollectionConfig:
    path = resolve_config_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = dict(data.get("collection", data))
    teams_file = block.pop("teams_file", None)
    dataset_recipe_file = block.pop("dataset_recipe_file", None)
    dataset_filters_inline = block.pop("dataset_filters", None)
    if teams_file is not None and block.get("team_filter") is not None:
        raise InvalidConfiguration("teams_file et team_filter sont exclusifs")
    if dataset_recipe_file is not None and dataset_filters_inline is not None:
        raise InvalidConfiguration("dataset_recipe_file et dataset_filters sont exclusifs")
    if teams_file is not None:
        from pokecoach_trainer.collection_filters import load_team_filter

        team_path = (path.parent / str(teams_file)).resolve()
        block["team_filter"] = load_team_filter(team_path).model_dump(mode="json")
    if dataset_recipe_file is not None:
        recipe_path = (path.parent / str(dataset_recipe_file)).resolve()
        recipe = load_dataset_recipe_file(recipe_path)
        block["dataset_filters"] = filters_from_recipe(recipe).model_dump(mode="json")
    elif dataset_filters_inline is not None:
        block["dataset_filters"] = ReplayFilterConfig.model_validate(
            dataset_filters_inline
        ).model_dump(mode="json")
    return CollectionConfig.model_validate(block)
