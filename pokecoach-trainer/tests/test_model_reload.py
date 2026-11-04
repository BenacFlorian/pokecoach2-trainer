import json
from pathlib import Path

from pokecoach_trainer.artifact_service import architecture_from_checkpoint, export_model
from pokecoach_trainer.reload_check import verify_reload as verify_reload_inprocess
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.storage import sha256_file
from pokecoach_trainer.training import train_bc
from training_fixtures import (
    experiment_config,
    install_storage_root,
    save_parent_checkpoint,
    write_ready_dataset,
)


def test_reload_check_subprocess_and_inprocess(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-reload")
    ds = "ds-reload"
    write_ready_dataset(ds, model, n_train=4, n_val=2, n_test=0)
    config = experiment_config(
        ds,
        "parent-reload",
        model,
        batch_size=8,
        num_epochs=1,
        device="cpu",
    )
    config.name = "reload"
    result = train_bc(config, tmp_path / "run-reload")
    ckpt = Path(result.final_checkpoint)
    arch = architecture_from_checkpoint(ckpt)
    assert arch.d_model == 64

    manifest_path = tmp_path / "manifest.json"
    digest = sha256_file(ckpt)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_id": "reload-test",
                "architecture": arch.model_dump(),
                "checkpoint_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    report = verify_reload_inprocess(ckpt, manifest_path)
    assert report.ok is True
    assert report.param_count > 0

    from pokecoach_trainer.artifact_service import verify_reload as verify_reload_subprocess

    report_sub = verify_reload_subprocess(ckpt, manifest_path)
    assert report_sub.ok is True


def test_reload_check_rejects_bad_hash(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-bad")
    ds = "ds-bad"
    write_ready_dataset(ds, model)
    config = experiment_config(ds, "parent-bad", model, batch_size=8, num_epochs=1, device="cpu")
    result = train_bc(config, tmp_path / "run-bad")
    ckpt = Path(result.final_checkpoint)
    arch = architecture_from_checkpoint(ckpt)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model_id": "bad",
                "architecture": arch.model_dump(),
                "checkpoint_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )
    report = verify_reload_inprocess(ckpt, manifest_path)
    assert report.ok is False


def test_extracted_zip_reload(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-zip")
    ds = "ds-zip"
    write_ready_dataset(ds, model)
    config = experiment_config(ds, "parent-zip", model, batch_size=8, num_epochs=1, device="cpu")
    result = train_bc(config, tmp_path / "run-zip")
    export_model(result, config, model_id="mdl-zip")
    zip_path = tmp_path / "models" / "mdl-zip" / "mdl-zip.zip"
    assert zip_path.is_file()
