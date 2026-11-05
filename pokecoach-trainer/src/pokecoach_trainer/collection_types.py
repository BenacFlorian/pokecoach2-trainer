"""Types internes immuables pour la collecte parallèle (sans dépendance ML)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pokecoach_trainer.schemas import ArchiveRef, CollectionConfig, JobRecord


@dataclass(frozen=True, slots=True)
class SearchItem:
    replay_id: str
    uploadtime: int
    source_format: str


@dataclass(frozen=True, slots=True)
class DiscoveryPage:
    items: tuple[SearchItem, ...]
    next_before: int | None
    exhausted: bool
    blocked_reason: str | None


@dataclass(frozen=True, slots=True)
class WindowBounds:
    source_format: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TaskLease:
    kind: Literal["download", "discovery"]
    task_id: str
    campaign_id: str
    attempt_id: str
    worker_id: str
    token: str
    expires_at: float
    replay_id: str | None
    window: WindowBounds | None
    cursor_before: int | None
    attempts_used: int


@dataclass(frozen=True, slots=True)
class ClaimResult:
    lease: TaskLease | None
    wait_s: float
    reason: str | None


@dataclass(frozen=True, slots=True)
class RequestPermit:
    permit_id: str
    attempt_id: str
    task_id: str
    kind: Literal["search", "fetch"]


@dataclass(frozen=True, slots=True)
class PermitResult:
    permit: RequestPermit | None
    wait_s: float
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class PreparedReplay:
    ref: ArchiveRef
    temp_relpath: str
    envelope: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PublicationResult:
    ref: ArchiveRef
    new_replay: bool
    new_version: bool


@dataclass(frozen=True, slots=True)
class WorkerIdentity:
    worker_id: str
    pid: int
    boot_id: str
    process_start_ticks: int
    token: str


@dataclass(frozen=True, slots=True)
class TeamSummary:
    p1: tuple[str, ...] | None
    p2: tuple[str, ...] | None
    parser_version: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    selected: bool
    reason: Literal[
        "selected",
        "team_mismatch",
        "team_unknown",
        "outside_period",
        "date_unknown",
    ]
    filter_sha256: str


@dataclass(frozen=True, slots=True)
class CampaignView:
    job: JobRecord
    phase: str
    attempt_id: str
    effective_config: CollectionConfig
    counters: dict[str, int]
    stop_reason: str | None
    coverage_incomplete: bool


class Clock(Protocol):
    def wall(self) -> float: ...

    def monotonic(self) -> float: ...

    def boot_id(self) -> str: ...
