"""CLI collecte parallèle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pokecoach_trainer.cli import main
from pokecoach_trainer.schemas import (
    CollectionConfig,
    CollectionJobConfig,
    JobKind,
    JobRecord,
    JobStatus,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


@patch("pokecoach_trainer.cli.submit_collection")
def test_cli_start_loads_config_from_invocation_directory(mock_submit, cli_env, monkeypatch):
    cwd = cli_env / "run"
    cwd.mkdir()
    cfg = cwd / "collection-mc.yaml"
    cfg.write_text("collection:\n  format_ids: [mc]\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    mock_submit.return_value = JobRecord(
        job_id="job-cli-1",
        kind=JobKind.collection,
        status=JobStatus.running,
        config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
    )
    rc = main(["collection", "start", "--config", "collection-mc.yaml"])
    assert rc == 0
    mock_submit.assert_called_once()
    assert mock_submit.call_args[0][0].format_ids == ["mc"]


@patch("pokecoach_trainer.collection_service.get_collection")
def test_cli_status_and_controls_share_service(mock_get, cli_env):
    from pokecoach_trainer.collection_types import CampaignView

    view = CampaignView(
        job=JobRecord(
            job_id="job-x",
            kind=JobKind.collection,
            status=JobStatus.running,
            config=CollectionJobConfig(config=CollectionConfig(format_ids=["mc"])),
        ),
        phase="running",
        attempt_id="attempt-1",
        effective_config=CollectionConfig(format_ids=["mc"]),
        counters={},
        stop_reason=None,
        coverage_incomplete=False,
    )
    mock_get.return_value = view
    rc = main(["collection", "status", "--job-id", "job-x"])
    assert rc == 0
    mock_get.assert_called_once_with("job-x")


def test_old_collection_yaml_remains_valid(cli_env):
    cfg = cli_env / "legacy.yaml"
    cfg.write_text(
        "collection:\n  format_ids: [mc]\n  request_interval_s: 1\n",
        encoding="utf-8",
    )
    from pokecoach_trainer.cli import _load_collection_config

    loaded = _load_collection_config(cfg)
    assert loaded.format_ids == ["mc"]
    assert loaded.discovery_workers == 1
    assert loaded.download_workers == 1
