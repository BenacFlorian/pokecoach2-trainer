from unittest.mock import patch

from pokecoach_trainer.main import app
from pokecoach_trainer.schemas import ExperimentConfig, InitializationConfig, JobRecord, JobKind, JobStatus, TrainingConfig
from pokecoach_trainer.schemas import ModelConfig
from fastapi.testclient import TestClient

from training_fixtures import install_storage_root, save_parent_checkpoint, write_ready_dataset


def test_training_job_preflight_and_submit(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-api")
    write_ready_dataset("ds-api", model)
    config = ExperimentConfig(
        name="api-run",
        seed=42,
        dataset_id="ds-api",
        model=model,
        initialization=InitializationConfig(base_model_id="parent-api"),
        training=TrainingConfig(batch_size=8, num_epochs=1, device="cpu"),
    )
    client = TestClient(app)
    with patch("pokecoach_trainer.api.training.submit_training") as mock_submit:
        mock_submit.return_value = JobRecord(
            job_id="job-train-1",
            kind=JobKind.training,
            status=JobStatus.queued,
            config=config,
        )
        resp = client.post("/training/jobs", json=config.model_dump(mode="json"))
    assert resp.status_code == 200
    mock_submit.assert_called_once()
    assert resp.json()["job_id"] == "job-train-1"


def test_training_job_rejects_unready_dataset(tmp_path, monkeypatch):
    install_storage_root(tmp_path, monkeypatch)
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    save_parent_checkpoint(tmp_path, model, base_model_id="parent-api2")
    config = ExperimentConfig(
        name="api-run",
        seed=42,
        dataset_id="missing-dataset",
        model=model,
        initialization=InitializationConfig(base_model_id="parent-api2"),
        training=TrainingConfig(batch_size=8, num_epochs=1, device="cpu"),
    )
    client = TestClient(app)
    resp = client.post("/training/jobs", json=config.model_dump(mode="json"))
    assert resp.status_code == 422
