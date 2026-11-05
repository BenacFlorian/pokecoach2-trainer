"""Schémas Pydantic partagés (config, manifestes, jobs)."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

_SHOWDOWN_SPECIES_ID = re.compile(r"^[a-z0-9]+$")


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
    model_config = ConfigDict(extra="forbid")

    method: Literal["bc"] = "bc"
    batch_size: int = 32
    num_epochs: int = 3
    device: str = "auto"
    learning_rate: float = 1e-4
    evaluation_battles_per_epoch: int = 0

    @field_validator("method")
    @classmethod
    def _method_bc_only(cls, v: str) -> str:
        if v != "bc":
            raise ValueError("seule la méthode bc est implémentée")
        return v

    @field_validator("batch_size", "num_epochs", "evaluation_battles_per_epoch")
    @classmethod
    def _non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("valeur entière invalide")
        return v

    @field_validator("batch_size", "num_epochs")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size et num_epochs doivent être >= 1")
        return v

    @field_validator("learning_rate")
    @classmethod
    def _learning_rate_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("learning_rate doit être fini et > 0")
        return v


def _validate_conversion_workers(value: object) -> Literal["auto"] | int:
    if isinstance(value, bool):
        raise ValueError("conversion_workers ne accepte pas les booléens")
    if value == "auto":
        return "auto"
    if isinstance(value, str):
        raise ValueError("conversion_workers doit être 'auto' ou un entier >= 1")
    if isinstance(value, int) and value >= 1:
        return value
    raise ValueError("conversion_workers doit être 'auto' ou un entier >= 1")


def _validate_conversion_timeout_s(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("conversion_timeout_s doit être fini et > 0")
    return value


class PrepareExecutionConfig(BaseModel):
    """Paramètres d'exécution prepare (parallélisme, timeouts) — hors recette dataset."""

    model_config = ConfigDict(extra="forbid")

    conversion_workers: Literal["auto"] | StrictInt = "auto"
    conversion_timeout_s: float = 120.0

    @field_validator("conversion_workers", mode="before")
    @classmethod
    def _coerce_workers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("conversion_workers ne accepte pas les booléens")
        if isinstance(value, str) and value.isdigit():
            raise ValueError(
                "conversion_workers: chaîne numérique interdite dans YAML/JSON ; utiliser un entier"
            )
        return value

    @field_validator("conversion_workers")
    @classmethod
    def _workers(cls, value: object) -> Literal["auto"] | int:
        return _validate_conversion_workers(value)

    @field_validator("conversion_timeout_s")
    @classmethod
    def _timeout(cls, value: float) -> float:
        return _validate_conversion_timeout_s(value)


class PrepareConfig(BaseModel):
    seed: int = 42
    validation_ratio: float = 0.10
    limit: int | None = None
    min_rating: int | None = None
    only_winner: bool = False
    strict: bool = False
    force: bool = False
    conversion_workers: Literal["auto"] | StrictInt = "auto"
    conversion_timeout_s: float = 120.0

    @field_validator("conversion_workers", mode="before")
    @classmethod
    def _coerce_workers(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("conversion_workers ne accepte pas les booléens")
        if isinstance(value, str) and value.isdigit():
            raise ValueError(
                "conversion_workers: chaîne numérique interdite dans YAML/JSON ; utiliser un entier"
            )
        return value

    @field_validator("conversion_workers")
    @classmethod
    def _workers(cls, value: object) -> Literal["auto"] | int:
        return _validate_conversion_workers(value)

    @field_validator("conversion_timeout_s")
    @classmethod
    def _timeout(cls, value: float) -> float:
        return _validate_conversion_timeout_s(value)

    def execution(self) -> PrepareExecutionConfig:
        return PrepareExecutionConfig(
            conversion_workers=self.conversion_workers,
            conversion_timeout_s=self.conversion_timeout_s,
        )


class SplitRatios(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: float = 0.80
    validation: float = 0.10
    test: float = 0.10

    @model_validator(mode="after")
    def _sum_one(self) -> SplitRatios:
        total = self.train + self.validation + self.test
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("les ratios de split doivent sommer à 1")
        if min(self.train, self.validation, self.test) <= 0:
            raise ValueError("chaque ratio de split doit être > 0")
        return self


class DatasetRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variant: Literal["natural"] = "natural"
    regulation: str = "M-C"
    target_replays: int = 300
    seed: int = 42
    min_elo: int | None = 1100
    max_elo: int | None = None
    min_rating: int | None = None
    unknown_elo: Literal["exclude_perspective"] = "exclude_perspective"
    only_winners: bool = False
    only_winner: bool | None = None
    require_ots: bool = True
    require_complete_teams: bool = True
    include_bo1: bool = True
    include_bo3: bool = True
    include_forfeits: bool = True
    min_completed_turns_forfeit: int = 3
    exclude_inactivity_losses: bool = True
    deduplicate: bool = True
    max_replays_per_team: int | None = None
    splits: SplitRatios = Field(default_factory=SplitRatios)

    @field_validator("target_replays")
    @classmethod
    def _target_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("target_replays doit être >= 1")
        return v

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("max_replays_per_team") is not None:
            raise ValueError("max_replays_per_team non nul refusé en V1")
        merged = dict(data)
        min_elo = merged.get("min_elo")
        min_rating = merged.get("min_rating")
        if min_rating is not None:
            if min_elo is not None and min_rating != min_elo:
                raise ValueError("min_rating et min_elo contradictoires")
            merged["min_elo"] = min_rating
        only_winners = merged.get("only_winners")
        only_winner = merged.get("only_winner")
        if only_winner is not None:
            if only_winners is not None and only_winner != only_winners:
                raise ValueError("only_winner et only_winners contradictoires")
            merged["only_winners"] = only_winner
        min_elo = merged.get("min_elo")
        max_elo = merged.get("max_elo")
        if min_elo is not None and max_elo is not None and min_elo > max_elo:
            raise ValueError("min_elo doit être <= max_elo")
        return merged


class TeamFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teams: list[list[str]]

    @field_validator("teams")
    @classmethod
    def _teams_shape(cls, teams: list[list[str]]) -> list[list[str]]:
        if not teams:
            raise ValueError("liste d'équipes vide")
        normalized: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for team in teams:
            if len(team) != 6:
                raise ValueError("chaque composition doit contenir exactement six identifiants")
            ids: list[str] = []
            for raw in team:
                sid = str(raw).strip().lower()
                if not sid or not _SHOWDOWN_SPECIES_ID.fullmatch(sid):
                    raise ValueError(f"identifiant Showdown invalide: {raw!r}")
                ids.append(sid)
            key = tuple(sorted(ids))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(sorted(ids))
        return normalized


class CollectionBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_selected_replays: int
    max_requests: int
    max_duration_s: int

    @field_validator("max_selected_replays", "max_requests", "max_duration_s")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("budget entier strictement positif requis")
        return v


class ResumeCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_attempt: bool = False
    budgets: CollectionBudgets | None = None


class ScaleCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    discovery_workers: int
    download_workers: int


class RetryFailedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = 5
    replay_ids: list[str] | None = None


class CollectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    format_ids: list[str]
    upload_after: datetime | None = None
    upload_before: datetime | None = None
    max_selected_replays: int = Field(
        default=5000,
        validation_alias=AliasChoices("max_selected_replays", "max_new_replays"),
    )
    max_requests: int = 10000
    max_duration_s: int = 7200
    requests_per_s: float = 1.0
    request_interval_s: float | None = None
    timeout_s: float = 30.0
    max_attempts: int = 5
    discovery_workers: int = 1
    download_workers: int = 1
    window_days: int = 7
    queue_high_water: int = 10_000
    lease_duration_s: int = 120
    heartbeat_s: int = 15
    mode: Literal["backfill", "incremental"] = "backfill"
    incremental_overlap_hours: int = 48
    reinspect: bool = False
    shutdown_grace_s: int = 60
    team_filter: TeamFilter | None = None
    effective_upload_before: datetime | None = Field(default=None, exclude=True)

    @field_validator("format_ids")
    @classmethod
    def _formats_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("format_ids ne peut pas être vide")
        return v

    @field_validator("upload_after", "upload_before", mode="before")
    @classmethod
    def _normalize_dates(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            dt = v
        else:
            dt = datetime.fromisoformat(str(v))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @field_validator("max_selected_replays", "max_requests", "max_duration_s")
    @classmethod
    def _budget_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("budget entier strictement positif requis")
        return v

    @field_validator("requests_per_s")
    @classmethod
    def _rate_finite(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0 or v > 20:
            raise ValueError("requests_per_s doit être fini et dans (0, 20]")
        return v

    @field_validator("request_interval_s")
    @classmethod
    def _interval_finite(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not math.isfinite(v) or v < 1.0:
            raise ValueError("request_interval_s doit être >= 1")
        return v

    @field_validator("timeout_s")
    @classmethod
    def _timeout_range(cls, v: float) -> float:
        if not math.isfinite(v) or v < 1 or v > 120:
            raise ValueError("timeout_s doit être entre 1 et 120")
        return v

    @field_validator("max_attempts")
    @classmethod
    def _attempts_range(cls, v: int) -> int:
        if v < 1 or v > 5:
            raise ValueError("max_attempts doit être entre 1 et 5")
        return v

    @field_validator("discovery_workers")
    @classmethod
    def _discovery_pool(cls, v: int) -> int:
        if v < 1 or v > 8:
            raise ValueError("discovery_workers doit être entre 1 et 8")
        return v

    @field_validator("download_workers")
    @classmethod
    def _download_pool(cls, v: int) -> int:
        if v < 1 or v > 64:
            raise ValueError("download_workers doit être entre 1 et 64")
        return v

    @field_validator("window_days")
    @classmethod
    def _window_days(cls, v: int) -> int:
        if v < 1 or v > 30:
            raise ValueError("window_days doit être entre 1 et 30")
        return v

    @field_validator("queue_high_water")
    @classmethod
    def _queue_hw(cls, v: int) -> int:
        if v < 100 or v > 100_000:
            raise ValueError("queue_high_water doit être entre 100 et 100000")
        return v

    @field_validator("lease_duration_s")
    @classmethod
    def _lease_duration(cls, v: int) -> int:
        if v < 30 or v > 3600:
            raise ValueError("lease_duration_s doit être entre 30 et 3600")
        return v

    @field_validator("heartbeat_s")
    @classmethod
    def _heartbeat(cls, v: int) -> int:
        if v < 1:
            raise ValueError("heartbeat_s doit être >= 1")
        return v

    @field_validator("incremental_overlap_hours")
    @classmethod
    def _overlap(cls, v: int) -> int:
        if v < 1:
            raise ValueError("incremental_overlap_hours doit être >= 1")
        return v

    @field_validator("shutdown_grace_s")
    @classmethod
    def _grace(cls, v: int) -> int:
        if v < 1:
            raise ValueError("shutdown_grace_s doit être >= 1")
        return v

    @model_validator(mode="before")
    @classmethod
    def _legacy_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "max_new_replays" in payload and "max_selected_replays" in payload:
            raise ValueError("max_new_replays et max_selected_replays sont exclusifs")
        has_interval = "request_interval_s" in payload
        has_rate = "requests_per_s" in payload
        if (
            has_interval
            and has_rate
            and payload.get("request_interval_s") is not None
            and payload.get("requests_per_s") is not None
        ):
            raise ValueError("request_interval_s et requests_per_s sont exclusifs")
        if has_interval and not has_rate:
            interval = float(payload["request_interval_s"])
            payload["requests_per_s"] = 1.0 / interval
        return payload

    @model_validator(mode="after")
    def _date_order_and_lease(self) -> CollectionConfig:
        if self.upload_after and self.upload_before:
            if self.upload_after >= self.upload_before:
                raise ValueError("upload_after doit précéder upload_before")
        if self.heartbeat_s * 3 > self.lease_duration_s:
            raise ValueError("heartbeat_s * 3 doit être <= lease_duration_s")
        return self

    @property
    def legacy_request_interval_s(self) -> float:
        return 1.0 / self.requests_per_s


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_ids: list[str]
    upload_after: datetime | None = None
    upload_before: datetime | None = None
    replay_version_selection: dict[str, str] | None = None


class ArchiveSource(BaseModel):
    """Provenance d'un objet archivé (hors contenu fournisseur)."""

    model_config = ConfigDict(extra="forbid")

    provenance: str
    fetched_at: datetime
    collector_revision: str = "pokecoach-trainer/0.1.0"


class ArchiveRef(BaseModel):
    """Référence publique vers un objet brut publié."""

    replay_id: str
    content_sha256: str
    log_sha256: str
    format_id: str
    uploadtime: int | None = None
    object_relpath: str
    conflict: bool = False


class ArchiveSnapshot(BaseModel):
    """Snapshot immuable d'une sélection de l'archive."""

    schema_version: int = 1
    snapshot_id: str
    snapshot_sha256: str
    metadata_version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    format_ids: list[str]
    upload_after: datetime | None = None
    upload_before: datetime | None = None
    refs: list[ArchiveRef]
    bo3_links_status: Literal["unknown", "partial", "confirmed"] = "unknown"
    replay_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SearchPage(BaseModel):
    """Page de résultats de recherche Showdown."""

    replay_ids: list[str]
    next_cursor: dict[str, Any] | None = None
    page_uploadtimes: list[int] = Field(default_factory=list)


class CollectionCursor(BaseModel):
    """Curseur persisté par format source Showdown."""

    format_id: str
    before: int
    phase: Literal["historical", "newer"] = "historical"
    newest_bound: int | None = None


class CollectionResult(BaseModel):
    """Résultat d'un job de collecte."""

    new_replays: int = 0
    selected_replays: int = 0
    reused_replays: int = 0
    skipped_existing: int = 0
    failed_replays: int = 0
    total_requests: int = 0
    stop_reason: str
    budget_limit: str | None = None
    coverage_incomplete: bool = False
    errors: list[dict[str, Any]] = Field(default_factory=list)
    effective_config: CollectionConfig


class InitializationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["finetune"] = "finetune"
    base_model_id: str
    architecture: Literal["inherit"] = "inherit"


class BaseModelSource(BaseModel):
    """Provenance exigée avant enregistrement d'un parent SB3."""

    model_config = ConfigDict(extra="forbid")

    base_model_id: str
    label: str
    expected_sha256: str
    audit_reference: str | None = None
    notes: str | None = None


class BaseModelManifest(BaseModel):
    """Manifeste immuable d'un modèle de base enregistré localement."""

    schema_version: int = 1
    base_model_id: str
    source: BaseModelSource
    checkpoint_sha256: str
    checkpoint_relpath: str = "model.zip"
    architecture: ModelConfig
    contract_fingerprint: str
    param_count: int
    policy_class: str
    load_status: str = "verified"
    registered_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationResult(BaseModel):
    """NLL moyenne sur des démonstrations (copie isolée de la policy)."""

    nll: float
    transitions: int


class FilterStage(str, Enum):
    game = "game"
    perspective = "perspective"
    selection = "selection"
    conversion = "conversion"


class FilterDecision(BaseModel):
    """Décision d'admissibilité traçable (n'altère pas l'archive)."""

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    stage: FilterStage
    reason_code: str
    replay_id: str
    player_side: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)


class ReplayMetadata(BaseModel):
    """Faits observés sur un replay archivé (valeurs inconnues explicites)."""

    model_config = ConfigDict(extra="forbid")

    replay_id: str
    raw_sha256: str
    log_sha256: str
    format_id: str
    parser_version: str = "replay_metadata/1"
    p1_username: str | None = None
    p2_username: str | None = None
    p1_rating: int | None = None
    p2_rating: int | None = None
    average_rating: int | None = None
    winner: str | None = None
    termination: Literal[
        "normal_win",
        "forfeit",
        "inactivity",
        "tie",
        "unknown",
    ] = "unknown"
    completed_turns: int | None = None
    ots_p1: bool = False
    ots_p2: bool = False
    team_complete_p1: bool = False
    team_complete_p2: bool = False
    match_type: Literal["bo1", "bo3", "unknown"] = "unknown"
    series_id: str | None = None
    game_number: int | None = None
    linked_replay_ids: list[str] = Field(default_factory=list)
    series_link_source: str | None = None
    series_link_status: Literal["confirmed", "unknown", "conflict"] = "unknown"


class ConversionStatus(str, Enum):
    converted = "converted"
    rejected = "rejected"
    error = "error"


class ConversionResult(BaseModel):
    """Résultat d'une conversion replay/perspective."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    replay_id: str
    raw_sha256: str
    player_side: Literal["p1", "p2"]
    series_id: str | None = None
    status: ConversionStatus
    reason_code: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    trajectory: Any | None = None
    converter_revision: str = "conversion/1"
    encoding_contract: str | None = None


class ReplayGroup(BaseModel):
    """Groupe indivisible pour sélection et splits."""

    group_id: str
    replay_ids: list[str]
    alias_ids: list[str] = Field(default_factory=list)
    series_id: str | None = None


class SplitAssignments(BaseModel):
    """Affectation train/validation/test par groupes."""

    algorithm: str = "group_greedy_v1"
    split_id: str
    train: list[str] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    group_to_split: dict[str, Literal["train", "validation", "test"]] = Field(
        default_factory=dict
    )
    replay_counts: dict[str, int] = Field(default_factory=dict)
    trainable: bool = True
    reason_code: str | None = None
    series_leakage_guarantee: bool = True
    ungrouped_bo3_count: int = 0


class DatasetState(str, Enum):
    imported = "imported"
    created = "created"
    preparing = "preparing"
    ready = "ready"
    insufficient_data = "insufficient_data"
    failed = "failed"
    interrupted = "interrupted"


class DatasetManifest(BaseModel):
    schema_version: int = 2
    dataset_id: str
    state: DatasetState
    source_relpath: str | None = None
    source_sha256: str | None = None
    snapshot_id: str | None = None
    snapshot_sha256: str | None = None
    recipe: DatasetRecipe | None = None
    recipe_sha256: str | None = None
    parent_dataset_id: str | None = None
    prepare_job_id: str | None = None
    format: str | None = None
    seed: int = 42
    validation_ratio: float = 0.10
    replay_ids_train: list[str] = Field(default_factory=list)
    replay_ids_val: list[str] = Field(default_factory=list)
    replay_ids_test: list[str] = Field(default_factory=list)
    trajectories_train: int = 0
    trajectories_val: int = 0
    trajectories_test: int = 0
    transitions_train: int = 0
    transitions_val: int = 0
    transitions_test: int = 0
    selected_replays: int = 0
    target_replays: int | None = None
    overshoot_replays: int = 0
    split_id: str | None = None
    series_leakage_guarantee: bool = True
    ungrouped_bo3_count: int = 0
    encoding_contract_fingerprint: str | None = None
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    prepare_execution: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CreateDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str | None = None
    snapshot_id: str
    recipe: DatasetRecipe
    parent_dataset_id: str | None = None


class LegacyImportDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str | None = None
    inbox_relpath: str
    recipe: DatasetRecipe
    parent_dataset_id: str | None = None


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    dataset_id: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    initialization: InitializationConfig | None = None
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
    architecture: ModelConfig | None = None
    parent_base_model_id: str | None = None
    parent_checkpoint_sha256: str | None = None
    dataset_id: str | None = None
    dataset_manifest_sha256: str | None = None
    split_id: str | None = None
    effective_learning_rate: float | None = None
    observed_optimizer_steps: list[int] = Field(default_factory=list)
    observed_transitions_per_epoch: list[int] = Field(default_factory=list)
    actor_delta_norm: float | None = None
    test_nll: float | None = None
    test_report_path: str | None = None


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    interrupted = "interrupted"


class JobKind(str, Enum):
    collection = "collection"
    prepare = "prepare"
    training = "training"


class CollectionJobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["collection"] = "collection"
    config: CollectionConfig


class SnapshotDatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["snapshot"] = "snapshot"
    snapshot_id: str


class LegacyImportDatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["legacy_import"] = "legacy_import"
    inbox_relpath: str
    recipe: DatasetRecipe


DatasetSource = Annotated[
    SnapshotDatasetSource | LegacyImportDatasetSource,
    Field(discriminator="source"),
]


class DatasetJobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["prepare"] = "prepare"
    dataset_id: str
    dataset_source: DatasetSource
    recipe: DatasetRecipe | None = None
    execution: PrepareExecutionConfig = Field(default_factory=PrepareExecutionConfig)


class TrainingJobConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["training"] = "training"
    experiment: ExperimentConfig


JobConfig = Annotated[
    CollectionJobConfig | DatasetJobConfig | TrainingJobConfig,
    Field(discriminator="kind"),
]


class JobRecord(BaseModel):
    schema_version: int = 2
    job_id: str
    kind: JobKind = JobKind.training
    status: JobStatus
    config: ExperimentConfig | JobConfig
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    run_dir: str | None = None
    worker_token: str | None = None
    pid: int | None = None
    startup_identity: str | None = None
    progress: dict[str, Any] | None = None
    result_ref: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_job_record(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "kind" not in data and "config" in data:
            cfg = data["config"]
            if isinstance(cfg, dict) and "name" in cfg and "dataset_id" in cfg:
                data = dict(data)
                data.setdefault("kind", JobKind.training.value)
                data.setdefault("schema_version", 1)
        return data


class ModelManifest(BaseModel):
    schema_version: int = 2
    model_id: str
    architecture: ModelConfig
    dataset_id: str
    experiment_name: str
    contract_fingerprint: str
    checkpoint_sha256: str
    trainer_version: str
    vgc_bench_commit: str
    trainer_reload: str = "passed"
    reserved_test: str = "pending"
    pokecoach_compatibility: str = "not_tested"
    published_at: datetime = Field(default_factory=datetime.utcnow)


class ReloadReport(BaseModel):
    ok: bool
    checkpoint_sha256: str
    param_count: int
    architecture: ModelConfig
    max_output_delta: float | None = None
    message: str = ""


def adapt_legacy_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Adaptateur explicite pour manifestes historiques sans schema_version."""

    out = dict(data)
    out.setdefault("schema_version", 1)
    return out
