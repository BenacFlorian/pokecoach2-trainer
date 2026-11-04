from fastapi.testclient import TestClient

from pokecoach_trainer.artifact_service import export_model
from pokecoach_trainer.main import app
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.settings import ensure_storage_layout
from pokecoach_trainer.training import train_bc
from training_fixtures import (
    experiment_config,
    install_storage_root,
    save_parent_checkpoint,
    write_ready_dataset,
)


def test_models_api_list_get_download(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    ensure_storage_layout()
    client = TestClient(app)

    assert client.get("/models").json() == []

    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-api")
    ds = "ds-api"
    write_ready_dataset(ds, model)
    config = experiment_config(
        ds,
        "parent-api",
        model,
        batch_size=8,
        num_epochs=1,
        device="cpu",
    )
    config.name = "api-model"
    result = train_bc(config, tmp_path / "run-api")
    export_model(result, config, model_id="mdl-api")

    listed = client.get("/models")
    assert listed.status_code == 200
    body = listed.json()
    assert len(body) == 1
    assert body[0]["model_id"] == "mdl-api"
    assert body[0]["trainer_reload"] == "passed"

    detail = client.get("/models/mdl-api")
    assert detail.status_code == 200
    assert detail.json()["pokecoach_compatibility"] == "not_tested"

    missing = client.get("/models/inconnu")
    assert missing.status_code == 404

    download = client.get("/models/mdl-api/download")
    assert download.status_code == 200
    assert len(download.content) > 100


def test_unpublished_model_not_listed(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    ensure_storage_layout()
    client = TestClient(app)
    assert client.get("/models").json() == []
