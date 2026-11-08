from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pokecoach_trainer.collection_service import (
    collection_request_fingerprint,
    normalize_collection_config,
)
from pokecoach_trainer.schemas import CollectionConfig, TeamFilter


def test_legacy_config_keeps_global_rate():
    cfg = CollectionConfig.model_validate(
        {"format_ids": ["mc"], "request_interval_s": 1.0}
    )
    assert cfg.requests_per_s == pytest.approx(1.0)
    assert cfg.discovery_workers == 1
    assert cfg.download_workers == 1


def test_interval_and_rate_rejected_even_when_equal():
    with pytest.raises(ValidationError):
        CollectionConfig.model_validate(
            {
                "format_ids": ["mc"],
                "request_interval_s": 1.0,
                "requests_per_s": 1.0,
            }
        )


def test_dates_are_normalized_not_relabelled():
    cfg = CollectionConfig.model_validate(
        {
            "format_ids": ["mc"],
            "upload_after": "2026-11-04T10:00:00+02:00",
        }
    )
    assert cfg.upload_after == datetime(2026, 11, 4, 8, 0, tzinfo=timezone.utc)
    naive = CollectionConfig.model_validate(
        {"format_ids": ["mc"], "upload_before": "2026-11-04T12:00:00"}
    )
    assert naive.upload_before.tzinfo == timezone.utc
    with pytest.raises(ValidationError):
        CollectionConfig.model_validate(
            {
                "format_ids": ["mc"],
                "upload_after": "2026-11-05T00:00:00Z",
                "upload_before": "2026-11-04T00:00:00Z",
            }
        )
    with pytest.raises(ValidationError):
        CollectionConfig.model_validate(
            {
                "format_ids": ["mc"],
                "upload_after": "2026-11-04T12:00:00Z",
                "upload_before": "2026-11-04T12:00:00Z",
            }
        )
    with pytest.raises(ValidationError):
        CollectionConfig(format_ids=["mc"], requests_per_s=float("nan"))
    with pytest.raises(ValidationError):
        CollectionConfig(format_ids=["mc"], max_selected_replays=0)
    with pytest.raises(ValidationError):
        CollectionConfig(format_ids=["mc"], discovery_workers=-1)


def test_basic_config_has_no_optional_filters():
    cfg = CollectionConfig(format_ids=["mc"])
    assert cfg.upload_after is None
    assert cfg.upload_before is None
    assert cfg.team_filter is None
    assert cfg.mode == "backfill"
    normalized = normalize_collection_config(
        cfg, now=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    )
    assert normalized.effective_upload_before == datetime(
        2026, 11, 4, 12, 0, tzinfo=timezone.utc
    )


def test_optional_filters_are_independent():
    team = TeamFilter(teams=[["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]])
    only_dates = CollectionConfig(
        format_ids=["mc"],
        upload_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert only_dates.team_filter is None
    only_team = CollectionConfig(format_ids=["mc"], team_filter=team)
    assert only_team.upload_after is None
    both = CollectionConfig(
        format_ids=["mc"],
        upload_before=datetime(2026, 6, 1, tzinfo=timezone.utc),
        team_filter=team,
    )
    assert both.upload_after is None and both.team_filter is not None
    single_before = CollectionConfig(
        format_ids=["mc"],
        upload_before=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert single_before.upload_after is None


def test_fingerprint_ignores_effective_upper_bound():
    cfg = CollectionConfig(format_ids=["mc"])
    fp1 = collection_request_fingerprint(cfg)
    later = normalize_collection_config(
        cfg, now=datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)
    )
    later2 = normalize_collection_config(
        cfg, now=datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
    )
    assert fp1 == collection_request_fingerprint(later)
    assert later.effective_upload_before != later2.effective_upload_before


def test_collectable_yaml_from_other_cwd(tmp_path, monkeypatch):
    from pokecoach_trainer.collection_config import load_collection_config
    from pokecoach_trainer.settings import PACKAGE_ROOT

    monkeypatch.chdir(tmp_path)
    cfg = load_collection_config(PACKAGE_ROOT / "configs" / "collection-mc-100k-valid.yaml")
    assert cfg.dataset_filters is not None
    assert cfg.dataset_filters.min_elo == 1100
    assert cfg.dataset_filters.min_completed_turns_forfeit == 2
    assert cfg.dataset_filters.exclude_inactivity_losses is True
    assert cfg.dataset_filters.require_ots is True
    assert cfg.dataset_filters.require_complete_teams is True


def test_max_new_replays_alias():
    cfg = CollectionConfig.model_validate({"format_ids": ["mc"], "max_new_replays": 42})
    assert cfg.max_selected_replays == 42
    with pytest.raises(ValidationError):
        CollectionConfig.model_validate(
            {
                "format_ids": ["mc"],
                "max_new_replays": 1,
                "max_selected_replays": 2,
            }
        )
