import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pokecoach_trainer.artifact_service import export_model, verify_reload
from pokecoach_trainer.schemas import ModelConfig, ReloadReport, TrainingConfig
from pokecoach_trainer.settings import clear_settings_cache, storage_path
from pokecoach_trainer.training import train_bc
from training_fixtures import (
    experiment_config,
    install_storage_root,
    save_parent_checkpoint,
    write_ready_dataset,
)


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    yield tmp_path


def _train_export_run(storage_env: Path, *, dataset_id: str, run_name: str):
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(storage_env, model, base_model_id="parent-export")
    write_ready_dataset(dataset_id, model, n_train=4, n_val=2, n_test=1)
    config = experiment_config(
        dataset_id,
        "parent-export",
        model,
        batch_size=8,
        num_epochs=1,
        device="cpu",
    )
    config.name = run_name
    result = train_bc(config, storage_env / f"run-{run_name}")
    return config, result


def test_export_publishes_after_reload(storage_env):
    config, result = _train_export_run(storage_env, dataset_id="ds-export", run_name="export-smoke")
    manifest = export_model(result, config, model_id="mdl-export")

    out_dir = storage_path("models", "mdl-export")
    assert (out_dir / "published.json").is_file()
    assert (out_dir / "mdl-export.zip").is_file()
    assert not storage_path("staging", "mdl-export").exists()

    pkg_manifest = json.loads((out_dir / "package" / "manifest.json").read_text())
    assert pkg_manifest["trainer_reload"] == "passed"
    assert pkg_manifest["pokecoach_compatibility"] == "not_tested"
    assert manifest.architecture.num_layers == 1
    assert manifest.architecture.d_model == 64


def test_export_failed_reload_leaves_no_publication(storage_env):
    config, result = _train_export_run(storage_env, dataset_id="ds-fail", run_name="fail-export")
    bad_report = ReloadReport(
        ok=False,
        checkpoint_sha256="",
        param_count=0,
        architecture=config.model,
        message="reload simulé en échec",
    )
    with patch("pokecoach_trainer.artifact_service.verify_reload", return_value=bad_report):
        with pytest.raises(RuntimeError, match="reload"):
            export_model(result, config, model_id="mdl-fail")
    assert not storage_path("models", "mdl-fail").exists()
    assert not storage_path("staging", "mdl-fail").exists()


def test_verify_reload_rejects_non_json_stdout(storage_env, tmp_path):
    ckpt = tmp_path / "model.zip"
    manifest = tmp_path / "manifest.json"
    ckpt.write_bytes(b"x")
    manifest.write_text("{}", encoding="utf-8")
    with patch("pokecoach_trainer.artifact_service.subprocess.run") as mock_run:
        mock_run.return_value = type(
            "R",
            (),
            {"returncode": 0, "stdout": "pas du json", "stderr": ""},
        )()
        report = verify_reload(ckpt, manifest)
    assert report.ok is False
    assert "non JSON" in report.message
