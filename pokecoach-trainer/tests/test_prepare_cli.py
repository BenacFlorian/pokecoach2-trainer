"""CLI dataset prepare — options d'exécution."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from pokecoach_trainer.cli import main
from pokecoach_trainer.schemas import PrepareExecutionConfig
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


def test_cli_overrides_yaml_only_when_explicit(cli_env, tmp_path):
    cfg_path = tmp_path / "prepare.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"prepare": {"conversion_workers": 2, "conversion_timeout_s": 90}}),
        encoding="utf-8",
    )
    captured: list[PrepareExecutionConfig] = []

    def _submit(dataset_id: str, execution: PrepareExecutionConfig | None = None):
        assert execution is not None
        captured.append(execution)
        from pokecoach_trainer.schemas import (
            DatasetJobConfig,
            JobKind,
            JobRecord,
            JobStatus,
            SnapshotDatasetSource,
        )

        return JobRecord(
            job_id="job-x",
            kind=JobKind.prepare,
            status=JobStatus.queued,
            config=DatasetJobConfig(
                dataset_id=dataset_id,
                dataset_source=SnapshotDatasetSource(snapshot_id="s"),
                execution=execution,
            ),
        )

    with patch("pokecoach_trainer.cli.submit_preparation", side_effect=_submit):
        rc = main(
            [
                "dataset",
                "prepare",
                "--dataset-id",
                "ds-1",
                "--config",
                str(cfg_path),
            ]
        )
    assert rc == 0
    assert captured[0].conversion_workers == 2

    captured.clear()
    with patch("pokecoach_trainer.cli.submit_preparation", side_effect=_submit):
        rc = main(
            [
                "dataset",
                "prepare",
                "--dataset-id",
                "ds-1",
                "--config",
                str(cfg_path),
                "--conversion-workers",
                "4",
            ]
        )
    assert rc == 0
    assert captured[0].conversion_workers == 4


def test_cli_conversion_workers_auto(cli_env):
    captured: list[PrepareExecutionConfig] = []

    def _submit(dataset_id: str, execution: PrepareExecutionConfig | None = None):
        assert execution is not None
        captured.append(execution)
        from pokecoach_trainer.schemas import (
            DatasetJobConfig,
            JobKind,
            JobRecord,
            JobStatus,
            SnapshotDatasetSource,
        )

        return JobRecord(
            job_id="j",
            kind=JobKind.prepare,
            status=JobStatus.queued,
            config=DatasetJobConfig(
                dataset_id="ds-a",
                dataset_source=SnapshotDatasetSource(snapshot_id="s"),
                execution=execution,
            ),
        )

    with patch("pokecoach_trainer.cli.submit_preparation", side_effect=_submit):
        main(
            [
                "dataset",
                "prepare",
                "--dataset-id",
                "ds-a",
                "--conversion-workers",
                "auto",
            ]
        )
    assert captured[0].conversion_workers == "auto"
