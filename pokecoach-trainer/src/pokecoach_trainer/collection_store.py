"""Persistance SQLite des campagnes, tâches et réservations."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Sequence

from pokecoach_trainer.archive_catalog import catalog_connection, initialize_catalog
from pokecoach_trainer.collection_filters import evaluate_selection, filter_fingerprint
from pokecoach_trainer.formats import resolve_format
from pokecoach_trainer.collection_service import (
    collection_request_fingerprint,
    normalize_collection_config,
)
from pokecoach_trainer.collection_types import (
    CampaignView,
    ClaimResult,
    Clock,
    DiscoveryPage,
    SearchItem,
    SelectionDecision,
    TaskLease,
    TeamSummary,
    WindowBounds,
)
from pokecoach_trainer.errors import Conflict, NotFound, StaleLease
from pokecoach_trainer.jobs import save_job
from pokecoach_trainer.schemas import (
    CollectionBudgets,
    CollectionConfig,
    CollectionJobConfig,
    JobKind,
    JobRecord,
    JobStatus,
    RetryFailedRequest,
)
from pokecoach_trainer.storage import new_opaque_id

_ACTIVE_PHASES = frozenset({"queued", "running", "pausing", "paused", "stopping"})


class SystemClock:
    def __init__(self) -> None:
        self._boot_id = self._read_boot_id()

    @staticmethod
    def _read_boot_id() -> str:
        try:
            return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
        except OSError:
            return "unknown-boot"

    def wall(self) -> float:
        return datetime.now(timezone.utc).timestamp()

    def monotonic(self) -> float:
        import time

        return time.monotonic()

    def boot_id(self) -> str:
        return self._boot_id


def _phase_to_status(phase: str) -> JobStatus:
    if phase in _ACTIVE_PHASES:
        return JobStatus.running
    if phase in {"completed", "budget_reached", "stopped"}:
        return JobStatus.completed
    if phase == "failed":
        return JobStatus.failed
    if phase == "interrupted":
        return JobStatus.interrupted
    return JobStatus.queued


class CollectionStore:
    def __init__(self, root: Path, *, clock: Clock | None = None) -> None:
        self._root = root
        self._clock = clock or SystemClock()
        initialize_catalog(root)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with catalog_connection(self._root) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _campaign_view(self, conn: sqlite3.Connection, campaign_id: str) -> CampaignView:
        row = conn.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"campagne {campaign_id} introuvable")
        attempt = conn.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (row["attempt_id"],)
        ).fetchone()
        effective = CollectionConfig.model_validate_json(row["effective_config_json"])
        job = JobRecord(
            job_id=row["campaign_id"],
            kind=JobKind.collection,
            status=_phase_to_status(row["phase"]),
            config=CollectionJobConfig(config=effective),
            created_at=datetime.fromtimestamp(row["created_at"], tz=timezone.utc),
        )
        counters = {
            "selected_in_attempt": int(attempt["selected_in_attempt"]) if attempt else 0,
            "requests_consumed": int(attempt["requests_consumed"]) if attempt else 0,
            "reserved_slots": int(attempt["reserved_slots"]) if attempt else 0,
            "new_published": int(attempt["new_published"]) if attempt else 0,
        }
        return CampaignView(
            job=job,
            phase=row["phase"],
            attempt_id=row["attempt_id"],
            effective_config=effective,
            counters=counters,
            stop_reason=row["stop_reason"],
            coverage_incomplete=False,
        )

    def _persist_job(self, view: CampaignView) -> None:
        save_job(view.job)

    def start_campaign(self, config: CollectionConfig) -> CampaignView:
        now = _utc_now()
        normalized = normalize_collection_config(config, now=now)
        fingerprint = collection_request_fingerprint(config)
        request_json = config.model_dump(mode="json", exclude={"effective_upload_before"})
        effective_json = normalized.model_dump(mode="json")
        with self._tx() as conn:
            active = conn.execute(
                """
                SELECT campaign_id FROM campaigns
                WHERE active_slot = 1 AND phase IN (
                    'queued','running','pausing','paused','stopping'
                )
                """
            ).fetchone()
            if active:
                existing = conn.execute(
                    "SELECT request_fingerprint FROM campaigns WHERE campaign_id = ?",
                    (active["campaign_id"],),
                ).fetchone()
                if existing and existing["request_fingerprint"] == fingerprint:
                    view = self._campaign_view(conn, active["campaign_id"])
                    self._persist_job(view)
                    return view
                raise Conflict("une campagne active existe déjà avec une autre configuration")
            campaign_id = new_opaque_id("job-")
            attempt_id = new_opaque_id("attempt-")
            ts = int(now.timestamp())
            conn.execute(
                """
                INSERT INTO campaigns(
                    campaign_id, request_fingerprint, request_config_json,
                    effective_config_json, phase, active_slot, attempt_id,
                    discovery_workers, download_workers, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'running', 1, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    fingerprint,
                    json.dumps(request_json, sort_keys=True),
                    json.dumps(effective_json, sort_keys=True),
                    attempt_id,
                    normalized.discovery_workers,
                    normalized.download_workers,
                    ts,
                    ts,
                ),
            )
            deadline = self._clock.wall() + normalized.max_duration_s
            conn.execute(
                """
                INSERT INTO attempts(
                    attempt_id, campaign_id, max_selected_replays, max_requests,
                    max_duration_s, started_at, deadline_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    campaign_id,
                    normalized.max_selected_replays,
                    normalized.max_requests,
                    normalized.max_duration_s,
                    self._clock.wall(),
                    deadline,
                ),
            )
            conn.execute(
                """
                INSERT INTO host_limits(
                    host_id, configured_rate, effective_rate, next_slot_mon,
                    provider_pause_until_wall, boot_id, boot_mon
                ) VALUES (1, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(host_id) DO UPDATE SET
                    configured_rate=excluded.configured_rate,
                    effective_rate=excluded.effective_rate,
                    boot_id=excluded.boot_id,
                    boot_mon=excluded.boot_mon
                """,
                (
                    normalized.requests_per_s,
                    normalized.requests_per_s,
                    self._clock.monotonic(),
                    self._clock.boot_id(),
                    self._clock.monotonic(),
                ),
            )
            self._reevaluate_known_metadata(conn, campaign_id, normalized)
            view = self._campaign_view(conn, campaign_id)
        self._persist_job(view)
        return view

    def _summary_from_row(self, row: sqlite3.Row) -> TeamSummary | None:
        if not row["p1_json"] and not row["p2_json"]:
            return None
        p1 = tuple(json.loads(row["p1_json"])) if row["p1_json"] else None
        p2 = tuple(json.loads(row["p2_json"])) if row["p2_json"] else None
        return TeamSummary(
            p1=p1,
            p2=p2,
            parser_version=row["parser_version"] or "",
            content_sha256=row["content_sha256"] or "",
        )

    def _reevaluate_known_metadata(
        self,
        conn: sqlite3.Connection,
        campaign_id: str,
        config: CollectionConfig,
    ) -> None:
        """Réévalue les métadonnées persistées pour la sélection courante (sans découverte)."""

        source_formats: list[str] = []
        for format_id in config.format_ids:
            source_formats.extend(resolve_format(format_id).source_format_ids)
        if not source_formats:
            return
        placeholders = ",".join("?" * len(source_formats))
        current_fp = filter_fingerprint(config)
        batch = 500
        offset = 0
        while True:
            rows = conn.execute(
                f"""
                SELECT rs.replay_id, rs.source_format, rs.uploadtime, rs.p1_json, rs.p2_json,
                       rs.parser_version, rs.content_sha256
                FROM replay_summaries rs
                WHERE rs.source_format IN ({placeholders})
                ORDER BY rs.replay_id
                LIMIT ? OFFSET ?
                """,
                (*source_formats, batch, offset),
            ).fetchall()
            if not rows:
                break
            for row in rows:
                prior = conn.execute(
                    """
                    SELECT decision, filter_sha256 FROM campaign_replays
                    WHERE campaign_id=? AND replay_id=?
                    """,
                    (campaign_id, row["replay_id"]),
                ).fetchone()
                if prior and prior["decision"] == "selected":
                    continue
                if prior and prior["filter_sha256"] == current_fp and prior["decision"] == "rejected":
                    continue
                summary = self._summary_from_row(row)
                uploadtime = int(row["uploadtime"]) if row["uploadtime"] is not None else None
                decision = evaluate_selection(
                    config, uploadtime=uploadtime, summary=summary
                )
                conn.execute(
                    """
                    INSERT INTO campaign_replays(
                        campaign_id, replay_id, decision, reason, filter_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(campaign_id, replay_id) DO UPDATE SET
                        decision=excluded.decision, reason=excluded.reason,
                        filter_sha256=excluded.filter_sha256
                    """,
                    (
                        campaign_id,
                        row["replay_id"],
                        "selected" if decision.selected else "rejected",
                        decision.reason,
                        decision.filter_sha256,
                    ),
                )
                if decision.selected:
                    task = conn.execute(
                        "SELECT state FROM download_tasks WHERE replay_id=?",
                        (row["replay_id"],),
                    ).fetchone()
                    if task is None:
                        conn.execute(
                            """
                            INSERT INTO download_tasks(replay_id, source_format, uploadtime, state)
                            VALUES (?, ?, ?, 'pending')
                            """,
                            (row["replay_id"], row["source_format"], row["uploadtime"]),
                        )
                    elif task["state"] in {
                        "examined_only",
                        "failed",
                        "retry_wait",
                        "unavailable",
                    }:
                        conn.execute(
                            """
                            UPDATE download_tasks SET state='pending', retry_at=NULL, last_error=NULL
                            WHERE replay_id=?
                            """,
                            (row["replay_id"],),
                        )
            offset += batch

    def get_campaign(self, campaign_id: str) -> CampaignView:
        with catalog_connection(self._root) as conn:
            return self._campaign_view(conn, campaign_id)

    def list_workers(self, campaign_id: str) -> list[dict[str, object]]:
        with catalog_connection(self._root) as conn:
            if conn.execute(
                "SELECT 1 FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone() is None:
                raise NotFound(f"campagne {campaign_id} introuvable")
            rows = conn.execute(
                """
                SELECT worker_id, role, state, pid, heartbeat_mon, attempt_id
                FROM workers WHERE campaign_id = ?
                ORDER BY worker_id
                """,
                (campaign_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def command(
        self,
        campaign_id: str,
        action: Literal["pause", "stop", "resume"],
        *,
        new_budgets: CollectionBudgets | None = None,
    ) -> CampaignView:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT phase, attempt_id FROM campaigns WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise NotFound(f"campagne {campaign_id} introuvable")
            phase = row["phase"]
            if action == "pause":
                if phase in {"paused", "stopping", "stopped", "completed", "failed"}:
                    pass
                else:
                    conn.execute(
                        "UPDATE campaigns SET phase='paused', updated_at=? WHERE campaign_id=?",
                        (int(_utc_now().timestamp()), campaign_id),
                    )
            elif action == "stop":
                conn.execute(
                    """
                    UPDATE campaigns SET phase='stopped', active_slot=NULL,
                           stop_reason='operator_stop', updated_at=?
                    WHERE campaign_id=?
                    """,
                    (int(_utc_now().timestamp()), campaign_id),
                )
            elif action == "resume":
                if phase == "running":
                    view = self._campaign_view(conn, campaign_id)
                    self._persist_job(view)
                    return view
                if phase == "budget_reached" and new_budgets is None:
                    raise Conflict("nouveaux budgets requis pour reprendre après budget")
                if new_budgets is not None:
                    attempt_id = new_opaque_id("attempt-")
                    conn.execute(
                        """
                        INSERT INTO attempts(
                            attempt_id, campaign_id, max_selected_replays, max_requests,
                            max_duration_s, started_at, deadline_at
                        )
                        SELECT ?, campaign_id, ?, ?, ?, ?, ? + ?
                        FROM campaigns WHERE campaign_id = ?
                        """,
                        (
                            attempt_id,
                            new_budgets.max_selected_replays,
                            new_budgets.max_requests,
                            new_budgets.max_duration_s,
                            self._clock.wall(),
                            self._clock.wall(),
                            new_budgets.max_duration_s,
                            campaign_id,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE campaigns SET phase='running', attempt_id=?, active_slot=1,
                               updated_at=? WHERE campaign_id=?
                        """,
                        (attempt_id, int(_utc_now().timestamp()), campaign_id),
                    )
                else:
                    conn.execute(
                        "UPDATE campaigns SET phase='running', updated_at=? WHERE campaign_id=?",
                        (int(_utc_now().timestamp()), campaign_id),
                    )
            view = self._campaign_view(conn, campaign_id)
        self._persist_job(view)
        return view

    def scale(
        self, campaign_id: str, *, discovery_workers: int, download_workers: int
    ) -> CampaignView:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE campaigns SET discovery_workers=?, download_workers=?, updated_at=?
                WHERE campaign_id=?
                """,
                (
                    discovery_workers,
                    download_workers,
                    int(_utc_now().timestamp()),
                    campaign_id,
                ),
            )
            view = self._campaign_view(conn, campaign_id)
        self._persist_job(view)
        return view

    def retry_failed(self, campaign_id: str, request: RetryFailedRequest) -> int:
        with self._tx() as conn:
            phase = conn.execute(
                "SELECT phase FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if phase is None:
                raise NotFound(f"campagne {campaign_id} introuvable")
            if phase["phase"] not in {"paused", "stopped", "failed", "budget_reached"}:
                raise Conflict("retry_failed uniquement sur campagne arrêtée ou en pause")
            if request.replay_ids:
                ids = request.replay_ids
            else:
                ids = [
                    r[0]
                    for r in conn.execute(
                        """
                        SELECT replay_id FROM download_tasks
                        WHERE state IN ('failed', 'unavailable') AND replay_id IN (
                            SELECT replay_id FROM campaign_replays WHERE campaign_id=?
                        )
                        """,
                        (campaign_id,),
                    ).fetchall()
                ]
            count = 0
            for replay_id in ids:
                cur = conn.execute(
                    """
                    UPDATE download_tasks SET state='pending', retry_at=NULL, last_error=NULL
                    WHERE replay_id=? AND state IN ('failed', 'unavailable')
                    """,
                    (replay_id,),
                )
                count += cur.rowcount
            return count

    def enqueue_replays(self, campaign_id: str, items: Sequence[SearchItem]) -> int:
        inserted = 0
        with self._tx() as conn:
            for item in items:
                cur = conn.execute(
                    """
                    INSERT INTO download_tasks(replay_id, source_format, uploadtime, state)
                    VALUES (?, ?, ?, 'pending')
                    ON CONFLICT(replay_id) DO NOTHING
                    """,
                    (item.replay_id, item.source_format, item.uploadtime),
                )
                if cur.rowcount:
                    inserted += 1
                    conn.execute(
                        """
                        INSERT INTO replay_summaries(replay_id, source_format, uploadtime)
                        VALUES (?, ?, ?)
                        ON CONFLICT(replay_id) DO UPDATE SET uploadtime=excluded.uploadtime
                        """,
                        (item.replay_id, item.source_format, item.uploadtime),
                    )
        return inserted

    def _lease_token(self) -> str:
        return secrets.token_hex(16)

    def _lease_expiry(self, config: CollectionConfig) -> tuple[float, float]:
        return (
            self._clock.monotonic() + config.lease_duration_s,
            self._clock.wall() + config.lease_duration_s,
        )

    def _lease_stale_clause(self) -> tuple[str, float, str]:
        return (
            "(owner_worker IS NULL OR lease_expires_wall <= ? OR lease_boot_id IS NULL OR lease_boot_id != ?)",
            self._clock.wall(),
            self._clock.boot_id(),
        )

    def verify_lease_for_permit(self, lease: TaskLease) -> bool:
        with catalog_connection(self._root) as conn:
            if lease.kind == "download":
                row = conn.execute(
                    """
                    SELECT lease_token, lease_boot_id, state FROM download_tasks
                    WHERE replay_id=?
                    """,
                    (lease.task_id,),
                ).fetchone()
                if row is None:
                    return False
                return (
                    row["state"] == "leased"
                    and row["lease_token"] == lease.token
                    and row["lease_boot_id"] == self._clock.boot_id()
                )
            if lease.kind == "discovery":
                row = conn.execute(
                    """
                    SELECT lease_token, lease_boot_id, state FROM windows
                    WHERE window_id=?
                    """,
                    (lease.task_id,),
                ).fetchone()
                if row is None:
                    return False
                return (
                    row["state"] in {"scanning", "pending"}
                    and row["lease_token"] == lease.token
                    and row["lease_boot_id"] == self._clock.boot_id()
                )
        return False

    def reconcile_expired_download_leases(self, campaign_id: str) -> int:
        now_wall = self._clock.wall()
        with self._tx() as conn:
            rows = conn.execute(
                """
                SELECT replay_id, reserved_attempt_id FROM download_tasks
                WHERE state='leased' AND lease_expires_wall <= ?
                """,
                (now_wall,),
            ).fetchall()
            count = 0
            for row in rows:
                conn.execute(
                    """
                    UPDATE download_tasks SET state='pending', owner_worker=NULL,
                           lease_token=NULL, lease_expires_mon=NULL, lease_expires_wall=NULL,
                           lease_boot_id=NULL, reserved_slot=0
                    WHERE replay_id=? AND state='leased'
                    """,
                    (row["replay_id"],),
                )
                if row["reserved_attempt_id"]:
                    conn.execute(
                        """
                        UPDATE attempts SET reserved_slots = MAX(0, reserved_slots - 1)
                        WHERE attempt_id=?
                        """,
                        (row["reserved_attempt_id"],),
                    )
                count += 1
            return count

    def _reconcile_reserved_slots(self, conn: sqlite3.Connection, attempt_id: str) -> None:
        """Aligne reserved_slots sur les téléchargements réellement en lease."""

        row = conn.execute(
            """
            SELECT COUNT(*) AS n FROM download_tasks
            WHERE state='leased' AND reserved_attempt_id=?
            """,
            (attempt_id,),
        ).fetchone()
        leased = int(row["n"]) if row else 0
        conn.execute(
            "UPDATE attempts SET reserved_slots=? WHERE attempt_id=?",
            (leased, attempt_id),
        )

    def claim_download(self, campaign_id: str, worker_id: str) -> ClaimResult:
        with self._tx() as conn:
            camp = conn.execute(
                "SELECT phase, attempt_id, effective_config_json FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            if camp is None or camp["phase"] not in {"running", "queued"}:
                return ClaimResult(lease=None, wait_s=0.0, reason="campaign_not_running")
            config = CollectionConfig.model_validate_json(camp["effective_config_json"])
            attempt = conn.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (camp["attempt_id"],)
            ).fetchone()
            self._reconcile_reserved_slots(conn, camp["attempt_id"])
            attempt = conn.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (camp["attempt_id"],)
            ).fetchone()
            if attempt["selected_in_attempt"] + attempt["reserved_slots"] >= attempt["max_selected_replays"]:
                return ClaimResult(lease=None, wait_s=1.0, reason="selection_full")
            row = conn.execute(
                """
                SELECT replay_id, source_format, uploadtime, durable_attempts, state
                FROM download_tasks
                WHERE (
                    (state='pending' AND (retry_at IS NULL OR retry_at <= ?))
                    OR (
                        state='leased' AND (
                            lease_expires_wall <= ? OR lease_boot_id IS NULL OR lease_boot_id != ?
                        )
                    )
                )
                ORDER BY replay_id LIMIT 1
                """,
                (self._clock.wall(), self._clock.wall(), self._clock.boot_id()),
            ).fetchone()
            if row is None:
                return ClaimResult(lease=None, wait_s=0.5, reason="queue_empty")
            token = self._lease_token()
            exp_mon, exp_wall = self._lease_expiry(config)
            boot = self._clock.boot_id()
            if row["state"] == "leased":
                prev = conn.execute(
                    "SELECT reserved_attempt_id FROM download_tasks WHERE replay_id=?",
                    (row["replay_id"],),
                ).fetchone()
                prev_attempt = prev["reserved_attempt_id"] if prev else camp["attempt_id"]
                conn.execute(
                    """
                    UPDATE attempts SET reserved_slots = MAX(0, reserved_slots - 1)
                    WHERE attempt_id=?
                    """,
                    (prev_attempt,),
                )
            updated = conn.execute(
                """
                UPDATE download_tasks SET state='leased', owner_worker=?, lease_token=?,
                       lease_expires_mon=?, lease_expires_wall=?, lease_boot_id=?,
                       reserved_slot=1, reserved_attempt_id=?
                WHERE replay_id=? AND (
                    state='pending'
                    OR (
                        state='leased' AND (
                            lease_expires_wall <= ? OR lease_boot_id IS NULL OR lease_boot_id != ?
                        )
                    )
                )
                """,
                (
                    worker_id,
                    token,
                    exp_mon,
                    exp_wall,
                    boot,
                    camp["attempt_id"],
                    row["replay_id"],
                    self._clock.wall(),
                    self._clock.boot_id(),
                ),
            )
            if updated.rowcount != 1:
                return ClaimResult(lease=None, wait_s=0.0, reason="lost_race")
            conn.execute(
                "UPDATE attempts SET reserved_slots = reserved_slots + 1 WHERE attempt_id=?",
                (camp["attempt_id"],),
            )
            lease = TaskLease(
                kind="download",
                task_id=row["replay_id"],
                campaign_id=campaign_id,
                attempt_id=camp["attempt_id"],
                worker_id=worker_id,
                token=token,
                expires_at=exp_wall,
                replay_id=row["replay_id"],
                window=None,
                cursor_before=None,
                attempts_used=int(row["durable_attempts"]),
            )
            return ClaimResult(lease=lease, wait_s=0.0, reason=None)

    def _window_key(self, bounds: WindowBounds, scan_generation: int) -> str:
        payload = f"{bounds.source_format}:{bounds.start}:{bounds.end}:{scan_generation}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def pending_download_count(self) -> int:
        with catalog_connection(self._root) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM download_tasks
                WHERE state IN ('pending', 'leased', 'retry_wait')
                """
            ).fetchone()
            return int(row["n"]) if row else 0

    def ensure_window(
        self, campaign_id: str, bounds: WindowBounds, scan_generation: int
    ) -> int:
        window_id = self._window_key(bounds, scan_generation)
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO windows(
                    window_id, source_format, start_ts, end_ts, scan_generation,
                    state, cursor_before
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?)
                ON CONFLICT(source_format, start_ts, end_ts, scan_generation) DO NOTHING
                """,
                (
                    window_id,
                    bounds.source_format,
                    bounds.start,
                    bounds.end,
                    scan_generation,
                    bounds.end,
                ),
            )
            conn.execute(
                """
                INSERT INTO campaign_windows(campaign_id, window_id)
                VALUES (?, ?)
                ON CONFLICT DO NOTHING
                """,
                (campaign_id, window_id),
            )
            return 1

    def materialize_windows(
        self,
        campaign_id: str,
        *,
        formats: list[str],
        range_start: int,
        range_end: int,
        partition_days: int,
        scan_generation: int,
        max_windows: int,
    ) -> int:
        step = partition_days * 86_400
        with catalog_connection(self._root) as conn:
            open_count = conn.execute(
                """
                SELECT COUNT(*) AS n FROM windows w
                JOIN campaign_windows cw ON cw.window_id = w.window_id
                WHERE cw.campaign_id=? AND w.state NOT IN ('complete', 'blocked')
                """,
                (campaign_id,),
            ).fetchone()
            need = max(0, max_windows - int(open_count["n"]))
        if need == 0:
            return 0
        created = 0
        fmt_idx = 0
        while created < need and formats:
            source_format = formats[fmt_idx % len(formats)]
            fmt_idx += 1
            with catalog_connection(self._root) as conn:
                row = conn.execute(
                    """
                    SELECT MIN(w.start_ts) AS oldest FROM windows w
                    JOIN campaign_windows cw ON cw.window_id = w.window_id
                    WHERE cw.campaign_id=? AND w.source_format=?
                    """,
                    (campaign_id, source_format),
                ).fetchone()
            next_end = int(row["oldest"]) if row and row["oldest"] is not None else range_end
            if next_end <= range_start:
                continue
            w_end = next_end
            w_start = max(range_start, w_end - step)
            bounds = WindowBounds(source_format=source_format, start=w_start, end=w_end)
            with catalog_connection(self._root) as conn:
                exists = conn.execute(
                    """
                    SELECT 1 FROM windows
                    WHERE source_format=? AND start_ts=? AND end_ts=? AND scan_generation=?
                    """,
                    (bounds.source_format, bounds.start, bounds.end, scan_generation),
                ).fetchone()
            if not exists:
                self.ensure_window(campaign_id, bounds, scan_generation)
                created += 1
        return created

    def claim_window(self, campaign_id: str, worker_id: str) -> ClaimResult:
        with self._tx() as conn:
            camp = conn.execute(
                "SELECT phase, attempt_id, effective_config_json FROM campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            if camp is None or camp["phase"] not in {"running", "queued"}:
                return ClaimResult(lease=None, wait_s=0.0, reason="campaign_not_running")
            config = CollectionConfig.model_validate_json(camp["effective_config_json"])
            stale_sql, stale_wall, stale_boot = self._lease_stale_clause()
            row = conn.execute(
                f"""
                SELECT w.window_id, w.source_format, w.start_ts, w.end_ts,
                       w.cursor_before, w.cursor_attempts
                FROM windows w
                JOIN campaign_windows cw ON cw.window_id = w.window_id
                WHERE cw.campaign_id=? AND w.state IN ('pending', 'scanning')
                  AND {stale_sql}
                ORDER BY w.source_format, w.start_ts DESC
                LIMIT 1
                """,
                (campaign_id, stale_wall, stale_boot),
            ).fetchone()
            if row is None:
                return ClaimResult(lease=None, wait_s=0.5, reason="no_windows")
            token = self._lease_token()
            exp_mon, exp_wall = self._lease_expiry(config)
            boot = self._clock.boot_id()
            updated = conn.execute(
                f"""
                UPDATE windows SET state='scanning', owner_worker=?, lease_token=?,
                       lease_expires_mon=?, lease_expires_wall=?, lease_boot_id=?,
                       cursor_before=COALESCE(cursor_before, end_ts)
                WHERE window_id=? AND {stale_sql}
                """,
                (
                    worker_id,
                    token,
                    exp_mon,
                    exp_wall,
                    boot,
                    row["window_id"],
                    stale_wall,
                    stale_boot,
                ),
            )
            if updated.rowcount != 1:
                return ClaimResult(lease=None, wait_s=0.0, reason="lost_race")
            bounds = WindowBounds(
                source_format=row["source_format"],
                start=int(row["start_ts"]),
                end=int(row["end_ts"]),
            )
            lease = TaskLease(
                kind="discovery",
                task_id=row["window_id"],
                campaign_id=campaign_id,
                attempt_id=camp["attempt_id"],
                worker_id=worker_id,
                token=token,
                expires_at=exp_wall,
                replay_id=None,
                window=bounds,
                cursor_before=int(row["cursor_before"] or row["end_ts"]),
                attempts_used=int(row["cursor_attempts"]),
            )
            return ClaimResult(lease=lease, wait_s=0.0, reason=None)

    def commit_discovery_page(self, lease: TaskLease, page: DiscoveryPage) -> None:
        if lease.window is None:
            raise StaleLease("lease découverte sans fenêtre")
        bounds = lease.window
        with self._tx() as conn:
            cur = conn.execute(
                """
                SELECT cursor_before, cursor_attempts, start_ts, end_ts, source_format
                FROM windows WHERE window_id=? AND lease_token=? AND owner_worker=?
                """,
                (lease.task_id, lease.token, lease.worker_id),
            ).fetchone()
            if cur is None:
                raise StaleLease("jeton fenêtre invalide")
            cursor_before = int(cur["cursor_before"] or bounds.end)
            if page.blocked_reason:
                conn.execute(
                    """
                    UPDATE windows SET state='blocked', last_signature=?, owner_worker=NULL,
                           lease_token=NULL
                    WHERE window_id=? AND lease_token=?
                    """,
                    (page.blocked_reason, lease.task_id, lease.token),
                )
                return
            if page.exhausted and not page.items:
                conn.execute(
                    """
                    INSERT INTO coverage(
                        segment_id, source_format, start_ts, end_ts, scan_generation,
                        proof_kind, proof_payload
                    ) VALUES (?, ?, ?, ?, 0, 'empty_search', ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        f"{bounds.source_format}:{bounds.start}:{bounds.end}:empty",
                        bounds.source_format,
                        bounds.start,
                        bounds.end,
                        json.dumps({"before": cursor_before}),
                    ),
                )
                conn.execute(
                    """
                    UPDATE windows SET state='complete', owner_worker=NULL, lease_token=NULL,
                           cursor_attempts=0
                    WHERE window_id=? AND lease_token=?
                    """,
                    (lease.task_id, lease.token),
                )
                return
            in_window: list[SearchItem] = []
            for item in page.items:
                if item.uploadtime is None:
                    continue
                ts = int(item.uploadtime)
                if ts < bounds.start or ts >= bounds.end:
                    continue
                in_window.append(item)
            if page.next_before is not None and page.next_before >= cursor_before:
                conn.execute(
                    """
                    UPDATE windows SET state='blocked', last_signature='timestamp_stall',
                           owner_worker=NULL, lease_token=NULL
                    WHERE window_id=? AND lease_token=?
                    """,
                    (lease.task_id, lease.token),
                )
                return
            attempt = conn.execute(
                "SELECT requests_consumed, max_requests FROM attempts WHERE attempt_id=?",
                (lease.attempt_id,),
            ).fetchone()
            budget_left = int(attempt["max_requests"]) - int(attempt["requests_consumed"])
            if not in_window and page.items and page.next_before is not None:
                new_cursor = int(page.next_before)
                if new_cursor < bounds.start:
                    if budget_left <= 0:
                        conn.execute(
                            """
                            UPDATE windows SET cursor_before=?, state='pending',
                                   owner_worker=NULL, lease_token=NULL
                            WHERE window_id=? AND lease_token=?
                            """,
                            (new_cursor, lease.task_id, lease.token),
                        )
                        return
                    conn.execute(
                        """
                        UPDATE windows SET state='complete', cursor_before=?, cursor_attempts=0,
                               owner_worker=NULL, lease_token=NULL
                        WHERE window_id=? AND lease_token=?
                        """,
                        (new_cursor, lease.task_id, lease.token),
                    )
                    conn.execute(
                        """
                        INSERT INTO coverage(
                            segment_id, source_format, start_ts, end_ts, scan_generation,
                            proof_kind, proof_payload
                        ) VALUES (?, ?, ?, ?, 0, 'scan_complete', ?)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            f"{bounds.source_format}:{bounds.start}:{bounds.end}:done",
                            bounds.source_format,
                            bounds.start,
                            bounds.end,
                            json.dumps({"cursor": new_cursor}),
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE windows SET cursor_before=?, cursor_attempts=0, state='scanning',
                               owner_worker=?, lease_token=?
                        WHERE window_id=? AND lease_token=?
                        """,
                        (
                            new_cursor,
                            lease.worker_id,
                            lease.token,
                            lease.task_id,
                            lease.token,
                        ),
                    )
                return
            if page.exhausted and not page.items and cursor_before > bounds.start:
                conn.execute(
                    """
                    UPDATE windows SET cursor_before=?, cursor_attempts=0, state='scanning',
                           owner_worker=?, lease_token=?
                    WHERE window_id=? AND lease_token=?
                    """,
                    (
                        bounds.start,
                        lease.worker_id,
                        lease.token,
                        lease.task_id,
                        lease.token,
                    ),
                )
                return
            for item in in_window:
                conn.execute(
                    """
                    INSERT INTO download_tasks(replay_id, source_format, uploadtime, state)
                    VALUES (?, ?, ?, 'pending')
                    ON CONFLICT(replay_id) DO NOTHING
                    """,
                    (item.replay_id, item.source_format, item.uploadtime),
                )
                conn.execute(
                    """
                    INSERT INTO replay_summaries(replay_id, source_format, uploadtime)
                    VALUES (?, ?, ?)
                    ON CONFLICT(replay_id) DO UPDATE SET uploadtime=excluded.uploadtime
                    """,
                    (item.replay_id, item.source_format, item.uploadtime),
                )
            new_cursor = page.next_before
            if new_cursor is not None and new_cursor < bounds.start:
                if budget_left <= 0:
                    conn.execute(
                        """
                        UPDATE windows SET cursor_before=?, state='pending',
                               owner_worker=NULL, lease_token=NULL
                        WHERE window_id=? AND lease_token=?
                        """,
                        (new_cursor, lease.task_id, lease.token),
                    )
                    return
                conn.execute(
                    """
                    UPDATE windows SET state='complete', cursor_before=?, cursor_attempts=0,
                           owner_worker=NULL, lease_token=NULL
                    WHERE window_id=? AND lease_token=?
                    """,
                    (new_cursor, lease.task_id, lease.token),
                )
                conn.execute(
                    """
                    INSERT INTO coverage(
                        segment_id, source_format, start_ts, end_ts, scan_generation,
                        proof_kind, proof_payload
                    ) VALUES (?, ?, ?, ?, 0, 'scan_complete', ?)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        f"{bounds.source_format}:{bounds.start}:{bounds.end}:done",
                        bounds.source_format,
                        bounds.start,
                        bounds.end,
                        json.dumps({"cursor": new_cursor}),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE windows SET cursor_before=?, cursor_attempts=0,
                           owner_worker=?, lease_token=?, state='scanning'
                    WHERE window_id=? AND lease_token=?
                    """,
                    (
                        new_cursor if new_cursor is not None else cursor_before,
                        lease.worker_id,
                        lease.token,
                        lease.task_id,
                        lease.token,
                    ),
                )

    def renew_lease(self, lease: TaskLease) -> TaskLease:
        with self._tx() as conn:
            camp = conn.execute(
                "SELECT effective_config_json FROM campaigns WHERE campaign_id=?",
                (lease.campaign_id,),
            ).fetchone()
            if camp is None:
                raise StaleLease("campagne absente")
            config = CollectionConfig.model_validate_json(camp["effective_config_json"])
            exp_mon, exp_wall = self._lease_expiry(config)
            boot = self._clock.boot_id()
            if lease.kind == "download":
                cur = conn.execute(
                    """
                    UPDATE download_tasks SET lease_expires_mon=?, lease_expires_wall=?, lease_boot_id=?
                    WHERE replay_id=? AND lease_token=? AND owner_worker=? AND lease_boot_id=?
                    """,
                    (
                        exp_mon,
                        exp_wall,
                        boot,
                        lease.task_id,
                        lease.token,
                        lease.worker_id,
                        boot,
                    ),
                )
            elif lease.kind == "discovery":
                cur = conn.execute(
                    """
                    UPDATE windows SET lease_expires_mon=?, lease_expires_wall=?, lease_boot_id=?
                    WHERE window_id=? AND lease_token=? AND owner_worker=? AND lease_boot_id=?
                    """,
                    (
                        exp_mon,
                        exp_wall,
                        boot,
                        lease.task_id,
                        lease.token,
                        lease.worker_id,
                        boot,
                    ),
                )
            else:
                raise StaleLease("type de lease inconnu")
            if cur.rowcount != 1:
                raise StaleLease("jeton de réservation invalide")
        return TaskLease(
            kind=lease.kind,
            task_id=lease.task_id,
            campaign_id=lease.campaign_id,
            attempt_id=lease.attempt_id,
            worker_id=lease.worker_id,
            token=lease.token,
            expires_at=exp_wall,
            replay_id=lease.replay_id,
            window=lease.window,
            cursor_before=lease.cursor_before,
            attempts_used=lease.attempts_used,
        )

    def release_lease(self, lease: TaskLease, *, retry_at: float | None = None) -> None:
        with self._tx() as conn:
            if lease.kind == "download":
                state = "retry_wait" if retry_at is not None else "pending"
                cur = conn.execute(
                    """
                    UPDATE download_tasks SET state=?, owner_worker=NULL, lease_token=NULL,
                           lease_expires_mon=NULL, lease_expires_wall=NULL, retry_at=?,
                           reserved_slot=0
                    WHERE replay_id=? AND lease_token=?
                    """,
                    (state, retry_at, lease.task_id, lease.token),
                )
                if cur.rowcount != 1:
                    raise StaleLease("jeton invalide à la libération")
                conn.execute(
                    """
                    UPDATE attempts SET reserved_slots = MAX(0, reserved_slots - 1)
                    WHERE attempt_id=?
                    """,
                    (lease.attempt_id,),
                )
            elif lease.kind == "discovery":
                cur = conn.execute(
                    """
                    UPDATE windows SET cursor_attempts = cursor_attempts + 1,
                           owner_worker=NULL, lease_token=NULL,
                           lease_expires_mon=NULL, lease_expires_wall=NULL,
                           lease_boot_id=NULL, state='pending'
                    WHERE window_id=? AND lease_token=?
                    """,
                    (lease.task_id, lease.token),
                )
                if cur.rowcount != 1:
                    raise StaleLease("jeton invalide à la libération")
                row = conn.execute(
                    "SELECT cursor_attempts FROM windows WHERE window_id=?",
                    (lease.task_id,),
                ).fetchone()
                camp = conn.execute(
                    "SELECT effective_config_json FROM campaigns WHERE campaign_id=?",
                    (lease.campaign_id,),
                ).fetchone()
                if row and camp:
                    config = CollectionConfig.model_validate_json(camp["effective_config_json"])
                    if int(row["cursor_attempts"]) >= config.max_attempts:
                        conn.execute(
                            """
                            UPDATE windows SET state='blocked', last_signature='max_page_attempts'
                            WHERE window_id=?
                            """,
                            (lease.task_id,),
                        )
            else:
                raise StaleLease("type de lease inconnu")

    def fail_lease(
        self,
        lease: TaskLease,
        *,
        permanent: bool,
        unavailable: bool,
        error: str,
    ) -> None:
        state = "unavailable" if unavailable else ("failed" if permanent else "retry_wait")
        with self._tx() as conn:
            cur = conn.execute(
                """
                UPDATE download_tasks SET state=?, last_error=?, owner_worker=NULL,
                       lease_token=NULL, durable_attempts=durable_attempts+1,
                       reserved_slot=0
                WHERE replay_id=? AND lease_token=?
                """,
                (state, error, lease.task_id, lease.token),
            )
            if cur.rowcount != 1:
                raise StaleLease("jeton invalide")
            conn.execute(
                "UPDATE attempts SET reserved_slots = MAX(0, reserved_slots - 1) WHERE attempt_id=?",
                (lease.attempt_id,),
            )

    def expired_leases(self, campaign_id: str, *, limit: int = 100) -> list[TaskLease]:
        now_mon = self._clock.monotonic()
        leases: list[TaskLease] = []
        with catalog_connection(self._root) as conn:
            rows = conn.execute(
                """
                SELECT d.replay_id, d.lease_token, d.owner_worker, d.lease_expires_wall,
                       c.attempt_id, d.durable_attempts
                FROM download_tasks d
                JOIN campaigns c ON c.campaign_id=?
                WHERE d.state='leased' AND d.lease_expires_mon <= ?
                LIMIT ?
                """,
                (campaign_id, now_mon, limit),
            ).fetchall()
            for row in rows:
                leases.append(
                    TaskLease(
                        kind="download",
                        task_id=row["replay_id"],
                        campaign_id=campaign_id,
                        attempt_id=row["attempt_id"],
                        worker_id=row["owner_worker"] or "",
                        token=row["lease_token"] or "",
                        expires_at=row["lease_expires_wall"] or 0.0,
                        replay_id=row["replay_id"],
                        window=None,
                        cursor_before=None,
                        attempts_used=int(row["durable_attempts"]),
                    )
                )
        return leases

    def record_selection(
        self,
        lease: TaskLease,
        decision: SelectionDecision,
        summary: TeamSummary | None,
    ) -> None:
        with self._tx() as conn:
            cur = conn.execute(
                "SELECT 1 FROM download_tasks WHERE replay_id=? AND lease_token=?",
                (lease.task_id, lease.token),
            ).fetchone()
            if cur is None:
                raise StaleLease("jeton invalide pour sélection")
            p1 = json.dumps(list(summary.p1)) if summary and summary.p1 else None
            p2 = json.dumps(list(summary.p2)) if summary and summary.p2 else None
            if summary:
                conn.execute(
                    """
                    INSERT INTO replay_summaries(
                        replay_id, source_format, uploadtime, p1_json, p2_json,
                        parser_version, content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(replay_id) DO UPDATE SET
                        p1_json=excluded.p1_json, p2_json=excluded.p2_json,
                        parser_version=excluded.parser_version,
                        content_sha256=excluded.content_sha256
                    """,
                    (
                        lease.replay_id,
                        "",
                        None,
                        p1,
                        p2,
                        summary.parser_version,
                        summary.content_sha256,
                    ),
                )
            conn.execute(
                """
                INSERT INTO campaign_replays(
                    campaign_id, replay_id, decision, reason, filter_sha256
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id, replay_id) DO UPDATE SET
                    decision=excluded.decision, reason=excluded.reason,
                    filter_sha256=excluded.filter_sha256
                """,
                (
                    lease.campaign_id,
                    lease.replay_id,
                    "selected" if decision.selected else "rejected",
                    decision.reason,
                    decision.filter_sha256,
                ),
            )
            if not decision.selected:
                conn.execute(
                    """
                    UPDATE download_tasks SET state='examined_only', reserved_slot=0,
                           owner_worker=NULL, lease_token=NULL, lease_expires_mon=NULL,
                           lease_expires_wall=NULL, lease_boot_id=NULL
                    WHERE replay_id=? AND lease_token=?
                    """,
                    (lease.task_id, lease.token),
                )
                conn.execute(
                    "UPDATE attempts SET reserved_slots = MAX(0, reserved_slots - 1) WHERE attempt_id=?",
                    (lease.attempt_id,),
                )
            else:
                conn.execute(
                    """
                    UPDATE attempts SET
                        selected_in_attempt = selected_in_attempt + 1,
                        reserved_slots = MAX(0, reserved_slots - 1)
                    WHERE attempt_id=?
                    """,
                    (lease.attempt_id,),
                )

    # --- API consommée par collection_limits ---

    def get_attempt_for_permit(self, attempt_id: str) -> sqlite3.Row:
        with catalog_connection(self._root) as conn:
            row = conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if row is None:
                raise NotFound("attempt introuvable")
            return row

    def reserve_request_permit(
        self,
        *,
        attempt_id: str,
        task_id: str,
        kind: Literal["search", "fetch"],
        permit_id: str,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE attempts SET requests_consumed = requests_consumed + 1 WHERE attempt_id=?
                """,
                (attempt_id,),
            )
            if kind == "fetch":
                conn.execute(
                    """
                    UPDATE download_tasks SET durable_attempts = durable_attempts + 1
                    WHERE replay_id=?
                    """,
                    (task_id,),
                )
            elif kind == "search":
                conn.execute(
                    """
                    UPDATE windows SET cursor_attempts = cursor_attempts + 1
                    WHERE window_id=?
                    """,
                    (task_id,),
                )
            conn.execute(
                """
                INSERT INTO request_permits(
                    permit_id, attempt_id, task_id, kind, state,
                    created_wall, updated_wall
                ) VALUES (?, ?, ?, ?, 'reserved', ?, ?)
                """,
                (permit_id, attempt_id, task_id, kind, self._clock.wall(), self._clock.wall()),
            )

    def sync_host_limit_boot(self) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE host_limits SET boot_id=?, boot_mon=?, next_slot_mon=?
                WHERE host_id=1
                """,
                (self._clock.boot_id(), self._clock.monotonic(), self._clock.monotonic()),
            )

    def finish_request_permit(
        self,
        permit_id: str,
        *,
        state: str,
        status_code: int | None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE request_permits SET state=?, http_code=?, updated_wall=?
                WHERE permit_id=?
                """,
                (state, status_code, self._clock.wall(), permit_id),
            )

    def host_limit_state(self) -> sqlite3.Row:
        with catalog_connection(self._root) as conn:
            row = conn.execute("SELECT * FROM host_limits WHERE host_id=1").fetchone()
            if row is None:
                raise NotFound("host_limits absent")
            return row

    def update_host_limit_slot(self, next_mon: float, effective_rate: float | None = None) -> None:
        with self._tx() as conn:
            if effective_rate is None:
                conn.execute(
                    "UPDATE host_limits SET next_slot_mon=? WHERE host_id=1", (next_mon,)
                )
            else:
                conn.execute(
                    "UPDATE host_limits SET next_slot_mon=?, effective_rate=? WHERE host_id=1",
                    (next_mon, effective_rate),
                )

    def set_provider_pause(self, until_wall: float, effective_rate: float) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE host_limits SET provider_pause_until_wall=?, effective_rate=?
                WHERE host_id=1
                """,
                (until_wall, effective_rate),
            )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
