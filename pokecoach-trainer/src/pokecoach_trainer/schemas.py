"""Schémas Pydantic partagés (config, manifestes, jobs)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    """Hyperparamètres de capacité de la policy (espace obs/actions inchangé)."""

    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    embed_len: int = 32
    choose_on_teampreview: bool = True

    @field_validator("d_model")
    @classmethod
    def _d_model_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("d_model doit être > 0")
        return v

    @field_validator("num_layers", "num_heads", "embed_len")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_layers, num_heads et embed_len doivent être >= 1")
        return v

    @model_validator(mode="after")
    def _heads_divide_d_model(self) -> ModelConfig:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model doit être divisible par num_heads")
        return self

    def policy_kwargs(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "embed_len": self.embed_len,
            "choose_on_teampreview": self.choose_on_teampreview,
        }


class TrainingConfig(BaseModel):
    batch_size: int = 32
    num_epochs: int = 3
    device: str = "auto"
    learning_rate: float | None = None


class PrepareConfig(BaseModel):
    seed: int = 42
    validation_ratio: float = 0.10
    limit: int | None = None
    min_rating: int | None = None
    only_winner: bool = False
    strict: bool = False
    force: bool = False


class DatasetState(str, Enum):
    imported = "imported"
    preparing = "preparing"
    ready = "ready"
    failed = "failed"


class DatasetManifest(BaseModel):
    dataset_id: str
    state: DatasetState
    source_relpath: str | None = None
    source_sha256: str | None = None
    format: str | None = None
    seed: int = 42
    validation_ratio: float = 0.10
    replay_ids_train: list[str] = Field(default_factory=list)
    replay_ids_val: list[str] = Field(default_factory=list)
    trajectories_train: int = 0
    trajectories_val: int = 0
    transitions_train: int = 0
    transitions_val: int = 0
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    dataset_id: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingResult(BaseModel):
    run_id: str
    checkpoints: list[str]
    final_checkpoint: str
    metrics_path: str
    param_count: int
    device: str
    weight_delta_norm: float
    train_losses: list[float]
    val_losses: list[float]


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    interrupted = "interrupted"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    config: ExperimentConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    run_dir: str | None = None
    worker_token: str | None = None
    pid: int | None = None


class ModelManifest(BaseModel):
    model_id: str
    architecture: ModelConfig
    dataset_id: str
    experiment_name: str
    contract_fingerprint: str
    checkpoint_sha256: str
    trainer_version: str
    vgc_bench_commit: str
    published_at: datetime = Field(default_factory=datetime.utcnow)


class ReloadReport(BaseModel):
    ok: bool
    checkpoint_sha256: str
    param_count: int
    architecture: ModelConfig
    max_output_delta: float | None = None
    message: str = ""
