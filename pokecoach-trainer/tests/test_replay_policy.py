"""Tests du contrat ReplayFilterConfig / replay_policy."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pokecoach_trainer.collection_config import load_collection_config
from pokecoach_trainer.collection_filters import filter_fingerprint
from pokecoach_trainer.replay_policy import (
    FILTER_POLICY_VERSION,
    evaluate_replay_policy,
    filters_from_recipe,
    replay_filter_fingerprint,
)
from pokecoach_trainer.schemas import CollectionConfig, DatasetRecipe, ReplayFilterConfig


def test_legacy_collection_yaml_has_no_dataset_filters(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("collection:\n  format_ids: [mc]\n", encoding="utf-8")
    cfg = load_collection_config(cfg_path)
    assert cfg.dataset_filters is None


def test_dataset_recipe_file_loads_twelve_fields(tmp_path):
    recipe = tmp_path / "dataset-mc-collectable.yaml"
    recipe.write_text(
        """
dataset:
  variant: natural
  regulation: M-C
  target_replays: 100000
  seed: 42
  min_elo: 1100
  max_elo: null
  unknown_elo: exclude_perspective
  only_winners: false
  require_ots: true
  require_complete_teams: true
  include_bo1: true
  include_bo3: true
  include_forfeits: true
  min_completed_turns_forfeit: 2
  exclude_inactivity_losses: true
  deduplicate: true
  max_replays_per_team: null
  splits:
    train: 0.80
    validation: 0.10
    test: 0.10
""",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "collection.yaml"
    cfg_path.write_text(
        f"collection:\n  format_ids: [mc]\n  dataset_recipe_file: {recipe.name}\n",
        encoding="utf-8",
    )
    cfg = load_collection_config(cfg_path)
    assert cfg.dataset_filters is not None
    assert cfg.dataset_filters.min_elo == 1100
    assert cfg.dataset_filters.min_completed_turns_forfeit == 2
    assert cfg.dataset_filters.exclude_inactivity_losses is True


def test_inline_and_file_mutually_exclusive(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        """
collection:
  format_ids: [mc]
  dataset_recipe_file: x.yaml
  dataset_filters:
    min_elo: 1100
""",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        load_collection_config(cfg_path)


def test_fingerprint_changes_with_each_filter_field():
    base = ReplayFilterConfig()
    base_fp = replay_filter_fingerprint(base)
    for field, value in (
        ("min_elo", 1200),
        ("max_elo", 1500),
        ("only_winners", True),
        ("require_ots", False),
        ("require_complete_teams", False),
        ("include_bo1", False),
        ("include_bo3", False),
        ("include_forfeits", False),
        ("min_completed_turns_forfeit", 1),
        ("exclude_inactivity_losses", False),
        ("deduplicate", False),
    ):
        assert replay_filter_fingerprint(base.model_copy(update={field: value})) != base_fp


def test_seed_target_do_not_change_business_fingerprint():
    recipe_a = DatasetRecipe(seed=1, target_replays=10)
    recipe_b = DatasetRecipe(seed=99, target_replays=999)
    assert replay_filter_fingerprint(filters_from_recipe(recipe_a)) == replay_filter_fingerprint(
        filters_from_recipe(recipe_b)
    )


def test_legacy_collection_fingerprint_unchanged_without_profile():
    cfg = CollectionConfig(format_ids=["mc"])
    fp = filter_fingerprint(cfg)
    assert len(fp) == 64


def test_policy_version_in_fingerprint():
    fp = replay_filter_fingerprint(ReplayFilterConfig())
    assert FILTER_POLICY_VERSION
    assert fp != replay_filter_fingerprint(ReplayFilterConfig(min_elo=1101))


def test_invalid_min_elo_max():
    with pytest.raises(ValidationError):
        ReplayFilterConfig(min_elo=1500, max_elo=1400)


def test_evaluate_replay_policy_requires_perspective(tmp_path, monkeypatch):
    from pokecoach_trainer.archive_service import archive_replay
    from pokecoach_trainer.replay_metadata import extract_replay_metadata
    from pokecoach_trainer.schemas import ArchiveSource
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
    from replay_log_fixtures import build_log

    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-pol"
    log = build_log(p1_rating=1300, p2_rating=900)
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    meta = extract_replay_metadata(ref)
    filters = ReplayFilterConfig()
    result = evaluate_replay_policy(meta, filters)
    assert result.accepted
    assert result.accepted_sides == ("p1",)


def test_evaluate_replay_policy_unknown_elo_only_average(tmp_path, monkeypatch):
    from pokecoach_trainer.archive_service import archive_replay
    from pokecoach_trainer.replay_metadata import extract_replay_metadata
    from pokecoach_trainer.schemas import ArchiveSource
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
    from replay_log_fixtures import build_log

    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    replay_id = "gen9championsvgc2026regmc-avg"
    log = build_log(p1_rating=None, p2_rating=None)
    ref = archive_replay(
        replay_id,
        {
            "id": replay_id,
            "uploadtime": 1,
            "log": log,
            "format": "gen9championsvgc2026regmc",
            "rating": 1500,
        },
        source,
    )
    meta = extract_replay_metadata(ref)
    result = evaluate_replay_policy(meta, ReplayFilterConfig())
    assert not result.accepted
    assert result.reason_code == "no_eligible_perspective"
