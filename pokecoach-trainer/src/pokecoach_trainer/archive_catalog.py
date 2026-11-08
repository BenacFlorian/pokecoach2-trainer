"""Catalogue SQLite transactionnel pour archive et collecte."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pokecoach_trainer.errors import Conflict, StorageIncompatible
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource
from pokecoach_trainer.storage import read_json

CATALOG_VERSION = 2
CATALOG_FILENAME = "catalog.sqlite3"
MAINTENANCE_LOCK = ".maintenance.lock"
_OBJECTS_DIR = "objects"
_ENVELOPE_VERSION = 1
MIN_SQLITE_VERSION = (3, 44, 6)

_SCHEMA_V1 = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS replays (
    replay_id TEXT PRIMARY KEY,
    source_format TEXT NOT NULL,
    uploadtime INTEGER,
    first_published_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_versions (
    publication_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    replay_id TEXT NOT NULL REFERENCES replays(replay_id),
    content_sha256 TEXT NOT NULL,
    log_sha256 TEXT NOT NULL,
    object_relpath TEXT NOT NULL,
    provider_format TEXT NOT NULL,
    uploadtime INTEGER,
    provenance TEXT NOT NULL,
    fetched_at INTEGER NOT NULL,
    conflict INTEGER NOT NULL DEFAULT 0 CHECK (conflict IN (0, 1)),
    UNIQUE (replay_id, content_sha256)
);
CREATE INDEX IF NOT EXISTS idx_replay_versions_replay_id ON replay_versions(replay_id);
CREATE INDEX IF NOT EXISTS idx_replay_versions_format_upload
    ON replay_versions(provider_format, uploadtime, replay_id);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    request_fingerprint TEXT NOT NULL,
    request_config_json TEXT NOT NULL,
    effective_config_json TEXT NOT NULL,
    phase TEXT NOT NULL,
    active_slot INTEGER UNIQUE CHECK (active_slot IS NULL OR active_slot = 1),
    attempt_id TEXT NOT NULL,
    discovery_workers INTEGER NOT NULL,
    download_workers INTEGER NOT NULL,
    supervisor_token TEXT,
    supervisor_heartbeat REAL,
    stop_reason TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    max_selected_replays INTEGER NOT NULL,
    max_requests INTEGER NOT NULL,
    max_duration_s INTEGER NOT NULL,
    started_at REAL NOT NULL,
    deadline_at REAL NOT NULL,
    requests_consumed INTEGER NOT NULL DEFAULT 0,
    selected_in_attempt INTEGER NOT NULL DEFAULT 0,
    reserved_slots INTEGER NOT NULL DEFAULT 0,
    new_published INTEGER NOT NULL DEFAULT 0,
    finished_at REAL,
    finish_reason TEXT
);

CREATE TABLE IF NOT EXISTS windows (
    window_id TEXT PRIMARY KEY,
    source_format TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    scan_generation INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    cursor_before INTEGER,
    last_signature TEXT,
    cursor_attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL,
    owner_worker TEXT,
    lease_token TEXT,
    lease_expires_mon REAL,
    lease_expires_wall REAL,
    lease_boot_id TEXT,
    UNIQUE (source_format, start_ts, end_ts, scan_generation)
);
CREATE INDEX IF NOT EXISTS idx_windows_state ON windows(state, retry_at);

CREATE TABLE IF NOT EXISTS campaign_windows (
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    window_id TEXT NOT NULL REFERENCES windows(window_id),
    PRIMARY KEY (campaign_id, window_id)
);

CREATE TABLE IF NOT EXISTS coverage (
    segment_id TEXT PRIMARY KEY,
    source_format TEXT NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    scan_generation INTEGER NOT NULL,
    proof_kind TEXT NOT NULL,
    proof_payload TEXT
);

CREATE TABLE IF NOT EXISTS download_tasks (
    replay_id TEXT PRIMARY KEY,
    source_format TEXT NOT NULL,
    uploadtime INTEGER,
    state TEXT NOT NULL CHECK (
        state IN (
            'pending', 'leased', 'archived', 'retry_wait', 'unavailable',
            'failed', 'examined_only'
        )
    ),
    owner_worker TEXT,
    lease_token TEXT,
    lease_expires_mon REAL,
    lease_expires_wall REAL,
    lease_boot_id TEXT,
    durable_attempts INTEGER NOT NULL DEFAULT 0,
    retry_at REAL,
    last_error TEXT,
    reserved_attempt_id TEXT,
    reserved_slot INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_download_tasks_claim
    ON download_tasks(state, retry_at, replay_id);

CREATE TABLE IF NOT EXISTS replay_summaries (
    replay_id TEXT PRIMARY KEY,
    source_format TEXT NOT NULL,
    uploadtime INTEGER,
    p1_json TEXT,
    p2_json TEXT,
    parser_version TEXT,
    content_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS campaign_replays (
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    replay_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    filter_sha256 TEXT NOT NULL,
    version_content_sha256 TEXT,
    first_selected_attempt TEXT,
    first_publication_attempt TEXT,
    PRIMARY KEY (campaign_id, replay_id)
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    attempt_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('discovery', 'download', 'supervisor')),
    pid INTEGER,
    boot_id TEXT,
    start_ticks INTEGER,
    token TEXT,
    state TEXT NOT NULL,
    heartbeat_mon REAL
);

CREATE TABLE IF NOT EXISTS publication_intents (
    intent_id TEXT PRIMARY KEY,
    replay_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    attempt_id TEXT,
    lease_token TEXT,
    temp_relpath TEXT NOT NULL,
    final_relpath TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    phase TEXT NOT NULL,
    UNIQUE (replay_id, content_sha256)
);

CREATE TABLE IF NOT EXISTS request_permits (
    permit_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    task_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('search', 'fetch')),
    state TEXT NOT NULL CHECK (
        state IN ('reserved', 'sent', 'completed', 'uncertain')
    ),
    http_code INTEGER,
    created_wall REAL NOT NULL,
    updated_wall REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_permits_attempt ON request_permits(attempt_id, state);

CREATE TABLE IF NOT EXISTS host_limits (
    host_id INTEGER PRIMARY KEY CHECK (host_id = 1),
    configured_rate REAL NOT NULL,
    effective_rate REAL NOT NULL,
    next_slot_mon REAL NOT NULL,
    provider_pause_until_wall REAL,
    boot_id TEXT NOT NULL,
    boot_mon REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_buckets (
    bucket_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    role TEXT NOT NULL,
    minute_ts INTEGER NOT NULL,
    counters_json TEXT NOT NULL
);
"""

_MIGRATE_V1_TO_V2 = """
ALTER TABLE campaigns ADD COLUMN validation_parser_version TEXT;
ALTER TABLE campaigns ADD COLUMN validation_policy_version TEXT;
ALTER TABLE campaign_replays ADD COLUMN decision_json TEXT;
ALTER TABLE campaign_replays ADD COLUMN log_sha256 TEXT;

CREATE TABLE IF NOT EXISTS replay_metadata_cache (
    replay_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    log_sha256 TEXT NOT NULL,
    source_format TEXT NOT NULL,
    uploadtime INTEGER,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (replay_id, content_sha256, parser_version)
);

CREATE TABLE IF NOT EXISTS campaign_log_claims (
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    log_sha256 TEXT NOT NULL,
    replay_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    lease_token TEXT,
    state TEXT NOT NULL CHECK (state IN ('pending', 'selected')),
    PRIMARY KEY (campaign_id, log_sha256)
);
"""


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    while len(parts) < 3:
        parts.append("0")
    return int(parts[0]), int(parts[1]), int(parts[2])


def check_sqlite_version(version: str) -> None:
    if _parse_version(version) < MIN_SQLITE_VERSION:
        raise StorageIncompatible(
            "version SQLite insuffisante pour le catalogue",
            details={"minimum": ".".join(map(str, MIN_SQLITE_VERSION)), "found": version},
        )


def catalog_path(root: Path) -> Path:
    return root / "archive" / CATALOG_FILENAME


def _legacy_objects_without_catalog(root: Path) -> bool:
    cat = catalog_path(root)
    if cat.is_file():
        return False
    objects = root / "archive" / "objects"
    if not objects.is_dir():
        return False
    for path in objects.rglob("*.json"):
        if not path.name.endswith(".tmp"):
            return True
    return False


def _connect(path: Path) -> sqlite3.Connection:
    check_sqlite_version(sqlite3.sqlite_version)
    conn = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=FULL")
    journal = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(journal).lower() != "wal":
        raise StorageIncompatible(f"WAL indisponible (journal_mode={journal})")
    return conn


@contextmanager
def catalog_connection(root: Path) -> Iterator[sqlite3.Connection]:
    path = catalog_path(root)
    if not path.is_file():
        raise StorageIncompatible("catalogue absent ; initialiser d'abord")
    conn = _connect(path)
    try:
        yield conn
    finally:
        conn.close()


def initialize_catalog(root: Path) -> None:
    if _legacy_objects_without_catalog(root):
        raise StorageIncompatible(
            "objets d'archive présents sans catalogue ; stockage incompatible"
        )
    archive_dir = root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_path(root)
    lock_path = archive_dir / ".catalog.bootstrap.lock"
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if path.is_file():
            with _connect(path) as conn:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0])
                if version == CATALOG_VERSION:
                    return
                if version == 1 and CATALOG_VERSION == 2:
                    conn.executescript(_MIGRATE_V1_TO_V2)
                    conn.execute("PRAGMA user_version=2")
                    conn.commit()
                    return
                if version not in (0, CATALOG_VERSION):
                    raise StorageIncompatible(
                        f"version de schéma inconnue: {version}",
                        details={"expected": CATALOG_VERSION},
                    )
        conn = _connect(path)
        try:
            conn.executescript(_SCHEMA_V1)
            if CATALOG_VERSION >= 2:
                conn.executescript(_MIGRATE_V1_TO_V2)
            conn.execute(f"PRAGMA user_version={CATALOG_VERSION}")
            conn.commit()
        finally:
            conn.close()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def catalog_is_initialized(root: Path) -> bool:
    return catalog_path(root).is_file()


def catalog_index_rows(root: Path) -> list[dict[str, Any]]:
    """Lignes compatibles index JSONL pour filtres archive existants."""

    if not catalog_is_initialized(root):
        return []
    with catalog_connection(root) as conn:
        rows = conn.execute(
            """
            SELECT replay_id, content_sha256, log_sha256, provider_format AS format_id,
                   uploadtime, object_relpath, conflict
            FROM replay_versions
            ORDER BY replay_id, content_sha256
            """
        ).fetchall()
    return [dict(row) for row in rows]


def catalog_list_replays(
    root: Path,
    *,
    offset: int = 0,
    limit: int = 100,
    format_id: str | None = None,
    upload_after_ts: int | None = None,
    upload_before_ts: int | None = None,
) -> tuple[list[ArchiveRef], int]:
    """Liste paginée depuis le catalogue (sans charger tout l'index)."""

    if not catalog_is_initialized(root):
        return [], 0
    clauses: list[str] = []
    params: list[Any] = []
    if format_id:
        clauses.append("provider_format = ?")
        params.append(format_id)
    if upload_after_ts is not None:
        clauses.append("uploadtime >= ?")
        params.append(upload_after_ts)
    if upload_before_ts is not None:
        clauses.append("uploadtime < ?")
        params.append(upload_before_ts)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with catalog_connection(root) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM replay_versions {where}",
                params,
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT replay_id, content_sha256, log_sha256, provider_format,
                   uploadtime, object_relpath, conflict
            FROM replay_versions
            {where}
            ORDER BY replay_id, content_sha256
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    refs = [
        ArchiveRef(
            replay_id=row["replay_id"],
            content_sha256=row["content_sha256"],
            log_sha256=row["log_sha256"],
            format_id=row["provider_format"],
            uploadtime=row["uploadtime"],
            object_relpath=row["object_relpath"],
            conflict=bool(row["conflict"]),
        )
        for row in rows
    ]
    return refs, total


def catalog_has_replay(
    root: Path, replay_id: str, content_sha256: str | None = None
) -> bool:
    if not catalog_path(root).is_file():
        return False
    with catalog_connection(root) as conn:
        if content_sha256 is None:
            row = conn.execute(
                "SELECT 1 FROM replay_versions WHERE replay_id = ? LIMIT 1", (replay_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM replay_versions WHERE replay_id = ? AND content_sha256 = ?",
                (replay_id, content_sha256),
            ).fetchone()
        return row is not None


def catalog_versions(root: Path, replay_id: str) -> list[ArchiveRef]:
    with catalog_connection(root) as conn:
        rows = conn.execute(
            """
            SELECT replay_id, content_sha256, log_sha256, provider_format,
                   uploadtime, object_relpath, conflict
            FROM replay_versions WHERE replay_id = ? ORDER BY publication_seq
            """,
            (replay_id,),
        ).fetchall()
    refs: list[ArchiveRef] = []
    for row in rows:
        refs.append(
            ArchiveRef(
                replay_id=row["replay_id"],
                content_sha256=row["content_sha256"],
                log_sha256=row["log_sha256"],
                format_id=row["provider_format"],
                uploadtime=row["uploadtime"],
                object_relpath=row["object_relpath"],
                conflict=bool(row["conflict"]),
            )
        )
    return refs


def insert_replay_version(
    root: Path,
    *,
    ref: ArchiveRef,
    source: ArchiveSource,
    provider_format: str,
) -> int:
    fetched_at = int(source.fetched_at.replace(tzinfo=timezone.utc).timestamp())
    now = int(datetime.now(timezone.utc).timestamp())
    with catalog_connection(root) as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO replays(replay_id, source_format, uploadtime, first_published_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(replay_id) DO NOTHING
            """,
            (ref.replay_id, provider_format, ref.uploadtime, now),
        )
        cur = conn.execute(
            """
            INSERT INTO replay_versions(
                replay_id, content_sha256, log_sha256, object_relpath,
                provider_format, uploadtime, provenance, fetched_at, conflict
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ref.replay_id,
                ref.content_sha256,
                ref.log_sha256,
                ref.object_relpath,
                provider_format,
                ref.uploadtime,
                source.provenance,
                fetched_at,
                int(ref.conflict),
            ),
        )
        conn.execute("COMMIT")
        return int(cur.lastrowid)


@contextmanager
def catalog_maintenance_shared(root: Path) -> Iterator[None]:
    """Verrou partagé pendant collecte, import ou snapshot."""

    lock_path = root / "archive" / MAINTENANCE_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def catalog_maintenance_exclusive(root: Path) -> Iterator[None]:
    """Verrou exclusif non bloquant pour reconstruction du catalogue."""

    lock_path = root / "archive" / MAINTENANCE_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Conflict(
                "maintenance catalogue en cours",
                details={"lock": str(lock_path)},
            ) from exc
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def catalog_max_publication_seq(root: Path) -> int:
    if not catalog_is_initialized(root):
        return 0
    with catalog_connection(root) as conn:
        row = conn.execute("SELECT MAX(publication_seq) AS m FROM replay_versions").fetchone()
        return int(row["m"] or 0)


def _row_to_archive_ref(row: sqlite3.Row) -> ArchiveRef:
    return ArchiveRef(
        replay_id=row["replay_id"],
        content_sha256=row["content_sha256"],
        log_sha256=row["log_sha256"],
        format_id=row["provider_format"],
        uploadtime=row["uploadtime"],
        object_relpath=row["object_relpath"],
        conflict=bool(row["conflict"]),
    )


def catalog_list_campaign_replays(
    root: Path,
    campaign_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    max_publication_seq: int | None = None,
) -> tuple[list[ArchiveRef], int]:
    """Replays retenus (`decision=selected`) pour une campagne."""

    if not catalog_is_initialized(root):
        return [], 0
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    seq_outer = ""
    seq_inner = ""
    if max_publication_seq is not None:
        seq_outer = " AND rv.publication_seq <= ?"
        seq_inner = " AND rv2.publication_seq <= ?"
    join_filter = f"""
        FROM campaign_replays cr
        INNER JOIN replay_versions rv ON rv.replay_id = cr.replay_id
        WHERE cr.campaign_id = ? AND cr.decision = 'selected'
        {seq_outer}
        AND rv.publication_seq = (
            SELECT MAX(rv2.publication_seq) FROM replay_versions rv2
            WHERE rv2.replay_id = cr.replay_id
            {seq_inner}
              AND (cr.version_content_sha256 IS NULL
                   OR rv2.content_sha256 = cr.version_content_sha256)
        )
    """
    params: list[Any] = [campaign_id]
    if max_publication_seq is not None:
        params.extend([max_publication_seq, max_publication_seq])
    with catalog_connection(root) as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(DISTINCT cr.replay_id) AS n {join_filter}",
                params,
            ).fetchone()["n"]
        )
        rows = conn.execute(
            f"""
            SELECT rv.replay_id, rv.content_sha256, rv.log_sha256, rv.provider_format,
                   rv.uploadtime, rv.object_relpath, rv.conflict
            {join_filter}
            ORDER BY cr.replay_id, rv.content_sha256
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
    return [_row_to_archive_ref(row) for row in rows], total


def _parse_object_envelope(obj_path: Path, archive_root: Path) -> dict[str, Any] | None:
    if obj_path.name.endswith(".tmp"):
        return None
    try:
        envelope = read_json(obj_path)
    except (json.JSONDecodeError, OSError):
        return None
    if envelope.get("schema_version") != _ENVELOPE_VERSION:
        return None
    ref = envelope.get("archive_ref") or {}
    replay_id = envelope.get("replay_id")
    content_sha256 = envelope.get("content_sha256")
    if not replay_id or not content_sha256:
        return None
    rel = obj_path.relative_to(archive_root).as_posix()
    source = envelope.get("source") or {}
    return {
        "replay_id": replay_id,
        "content_sha256": content_sha256,
        "log_sha256": envelope.get("log_sha256"),
        "format_id": ref.get("format_id") or envelope.get("format_id"),
        "uploadtime": ref.get("uploadtime"),
        "object_relpath": rel,
        "conflict": bool(envelope.get("conflict")),
        "provenance": source.get("provenance") or "rebuild",
        "fetched_at": source.get("fetched_at"),
        "provider_format": str(
            ref.get("format_id") or envelope.get("format_id") or replay_id.split("-")[0]
        ),
    }


def rebuild_catalog_from_objects(root: Path) -> int:
    """Reconstruit `replays` / `replay_versions` depuis les objets (sans toucher aux campagnes)."""

    initialize_catalog(root)
    archive_root = root / "archive"
    objects_dir = archive_root / _OBJECTS_DIR
    entries: list[dict[str, Any]] = []
    if objects_dir.is_dir():
        for obj_path in sorted(objects_dir.rglob("*.json")):
            parsed = _parse_object_envelope(obj_path, archive_root)
            if parsed:
                entries.append(parsed)
    entries.sort(key=lambda e: (e["replay_id"], e["content_sha256"]))
    with catalog_maintenance_exclusive(root):
        with catalog_connection(root) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM replay_versions")
            conn.execute("DELETE FROM replays")
            now = int(datetime.now(timezone.utc).timestamp())
            for entry in entries:
                conn.execute(
                    """
                    INSERT INTO replays(replay_id, source_format, uploadtime, first_published_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(replay_id) DO NOTHING
                    """,
                    (
                        entry["replay_id"],
                        entry["provider_format"],
                        entry.get("uploadtime"),
                        now,
                    ),
                )
                fetched = entry.get("fetched_at")
                if isinstance(fetched, str):
                    try:
                        fetched_ts = int(
                            datetime.fromisoformat(fetched.replace("Z", "+00:00"))
                            .timestamp()
                        )
                    except ValueError:
                        fetched_ts = now
                elif isinstance(fetched, (int, float)):
                    fetched_ts = int(fetched)
                else:
                    fetched_ts = now
                conn.execute(
                    """
                    INSERT INTO replay_versions(
                        replay_id, content_sha256, log_sha256, object_relpath,
                        provider_format, uploadtime, provenance, fetched_at, conflict
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry["replay_id"],
                        entry["content_sha256"],
                        entry["log_sha256"],
                        entry["object_relpath"],
                        entry["provider_format"],
                        entry.get("uploadtime"),
                        entry["provenance"],
                        fetched_ts,
                        int(bool(entry.get("conflict"))),
                    ),
                )
            conn.execute("COMMIT")
    return len(entries)


def export_catalog_jsonl(root: Path, destination: Path) -> int:
    """Export explicite de l'index catalogue vers JSONL (renommage atomique)."""

    initialize_catalog(root)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with catalog_maintenance_shared(root):
        with catalog_connection(root) as conn:
            fd, tmp = tempfile.mkstemp(dir=destination.parent, suffix=".jsonl.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    offset = 0
                    batch = 1000
                    while True:
                        rows = conn.execute(
                            """
                            SELECT replay_id, content_sha256, log_sha256,
                                   provider_format AS format_id, uploadtime,
                                   object_relpath, conflict
                            FROM replay_versions
                            ORDER BY replay_id, content_sha256
                            LIMIT ? OFFSET ?
                            """,
                            (batch, offset),
                        ).fetchall()
                        if not rows:
                            break
                        for row in rows:
                            line = json.dumps(
                                dict(row),
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=False,
                            )
                            handle.write(line + "\n")
                            count += 1
                        offset += batch
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, destination)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    return count


def backup_catalog(root: Path, destination: Path) -> None:
    """Sauvegarde SQLite incluant le contenu WAL commité."""

    src_path = catalog_path(root)
    if not src_path.is_file():
        raise StorageIncompatible("catalogue absent")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with catalog_maintenance_shared(root):
        src = _connect(src_path)
        dest = sqlite3.connect(str(destination))
        try:
            src.backup(dest)
        finally:
            src.close()
            dest.close()
