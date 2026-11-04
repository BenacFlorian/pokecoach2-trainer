"""Archive brute des replays Showdown et snapshots immuables."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from pokecoach_trainer.errors import Conflict, InvalidConfiguration, NotFound
from pokecoach_trainer.formats import (
    CHAMPIONS_VGC_2026_PREFIX,
    is_champions_vgc_2026_replay_id,
    resolve_format,
)
from pokecoach_trainer.schemas import (
    ArchiveRef,
    ArchiveSnapshot,
    ArchiveSource,
    SnapshotRequest,
)
from pokecoach_trainer.settings import get_settings, storage_path
from pokecoach_trainer.storage import (
    append_jsonl_atomic,
    canonical_json_sha256,
    new_opaque_id,
    read_json,
    resolve_inbox_relative,
    sha256_text,
    validate_id,
    write_json_atomic,
)

_ENVELOPE_VERSION = 1
_INDEX_REL = "index/replays.jsonl"
_SNAPSHOTS_DIR = "snapshots"
_OBJECTS_DIR = "objects"
_METADATA_LOCK = ".metadata.lock"


def _archive_root() -> Path:
    return storage_path("archive")


def _index_path() -> Path:
    return _archive_root() / _INDEX_REL


@contextmanager
def _metadata_lock() -> Iterator[None]:
    """Verrou fichier distinct du slot de job long pour index/snapshots."""

    lock_path = _archive_root() / _METADATA_LOCK
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _provider_canonical(payload: dict[str, Any]) -> dict[str, Any]:
    """Extrait les champs fournisseur stables pour le hash de contenu."""

    keys = ("id", "uploadtime", "log", "format", "p1", "p2", "rating", "roomid")
    return {k: payload[k] for k in keys if k in payload}


def _parse_index_line(raw: str) -> dict[str, Any] | None:
    line = raw.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def load_index_entries() -> list[dict[str, Any]]:
    """Charge l'index JSONL en ignorant les lignes corrompues."""

    path = _index_path()
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_index_line(raw)
        if parsed:
            entries.append(parsed)
    return entries


def rebuild_index_from_objects() -> int:
    """
    Reconstruit l'index à partir des objets publiés (ignore les fichiers temporaires).

    Returns:
        Nombre d'entrées écrites.
    """

    root = _archive_root() / _OBJECTS_DIR
    entries: list[dict[str, Any]] = []
    if root.is_dir():
        for obj_path in sorted(root.rglob("*.json")):
            if obj_path.name.endswith(".tmp"):
                continue
            try:
                envelope = read_json(obj_path)
            except (json.JSONDecodeError, OSError):
                continue
            if envelope.get("schema_version") != _ENVELOPE_VERSION:
                continue
            rel = obj_path.relative_to(_archive_root()).as_posix()
            ref = envelope.get("archive_ref") or {}
            entries.append(
                {
                    "replay_id": envelope.get("replay_id"),
                    "content_sha256": envelope.get("content_sha256"),
                    "log_sha256": envelope.get("log_sha256"),
                    "format_id": ref.get("format_id") or envelope.get("format_id"),
                    "uploadtime": ref.get("uploadtime"),
                    "object_relpath": rel,
                    "conflict": bool(envelope.get("conflict")),
                }
            )
    entries.sort(
        key=lambda e: (e.get("replay_id") or "", e.get("content_sha256") or "")
    )
    with _metadata_lock():
        index_path = _index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_lines = [
            json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for entry in entries
        ]
        fd, tmp = tempfile.mkstemp(dir=index_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for line in tmp_lines:
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, index_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return len(entries)


def _existing_versions(replay_id: str) -> list[dict[str, Any]]:
    return [e for e in load_index_entries() if e.get("replay_id") == replay_id]


def archive_replay(
    replay_id: str, payload: dict[str, Any], source: ArchiveSource
) -> ArchiveRef:
    """
    Publie un replay brut sans filtre métier.

    Raises:
        InvalidConfiguration: ID incohérent ou payload invalide.
        Conflict: versions multiples pour le même replay_id.
    """

    provider_id = payload.get("id")
    if provider_id != replay_id:
        raise InvalidConfiguration(
            "identifiant replay incohérent avec la réponse fournisseur",
            details={"requested": replay_id, "provider": provider_id},
        )
    if not is_champions_vgc_2026_replay_id(replay_id):
        raise InvalidConfiguration(
            "replay hors périmètre Champions VGC 2026 (M-A/M-B/M-C)",
            details={"replay_id": replay_id},
        )
    log = payload.get("log")
    if not isinstance(log, str) or not log:
        raise InvalidConfiguration("log replay manquant ou invalide")

    format_id = str(payload.get("format") or "")
    uploadtime = payload.get("uploadtime")
    if uploadtime is not None:
        uploadtime = int(uploadtime)

    canonical = _provider_canonical(payload)
    content_sha256 = canonical_json_sha256(canonical)
    log_sha256 = sha256_text(log)
    rel = f"{_OBJECTS_DIR}/{replay_id}/{content_sha256}.json"
    object_path = _archive_root() / rel

    for existing in _existing_versions(replay_id):
        if existing.get("content_sha256") == content_sha256:
            return ArchiveRef(
                replay_id=replay_id,
                content_sha256=content_sha256,
                log_sha256=log_sha256,
                format_id=str(existing.get("format_id") or format_id),
                uploadtime=existing.get("uploadtime") or uploadtime,
                object_relpath=str(existing.get("object_relpath") or rel),
                conflict=bool(existing.get("conflict")),
            )

    conflict = False
    for existing in _existing_versions(replay_id):
        if existing.get("content_sha256") != content_sha256:
            conflict = True
            break

    ref = ArchiveRef(
        replay_id=replay_id,
        content_sha256=content_sha256,
        log_sha256=log_sha256,
        format_id=format_id,
        uploadtime=uploadtime,
        object_relpath=rel,
        conflict=conflict,
    )
    envelope = {
        "schema_version": _ENVELOPE_VERSION,
        "replay_id": replay_id,
        "format_id": format_id,
        "content_sha256": content_sha256,
        "log_sha256": log_sha256,
        "provider": payload,
        "source": source.model_dump(mode="json"),
        "archive_ref": ref.model_dump(mode="json"),
        "conflict": conflict,
    }
    with _metadata_lock():
        if object_path.is_file():
            return ref
        write_json_atomic(object_path, envelope)
        append_jsonl_atomic(
            _index_path(),
            {
                "replay_id": replay_id,
                "content_sha256": content_sha256,
                "log_sha256": log_sha256,
                "format_id": format_id,
                "uploadtime": uploadtime,
                "object_relpath": rel,
                "conflict": conflict,
            },
        )
    return ref


def _row_matches_format(
    row: dict[str, Any], format_filter: set[str], source_prefixes: list[str]
) -> bool:
    fmt = str(row.get("format_id") or "")
    if fmt in format_filter:
        return True
    replay_id = str(row.get("replay_id") or "")
    for prefix in source_prefixes:
        if replay_id.startswith(f"{prefix}-"):
            return True
    return False


def _index_rows_to_refs(
    rows: list[dict[str, Any]],
    *,
    format_ids: list[str] | None = None,
    source_prefixes: list[str] | None = None,
    upload_after: datetime | None = None,
    upload_before: datetime | None = None,
) -> list[ArchiveRef]:
    refs: list[ArchiveRef] = []
    allowed = set(format_ids) if format_ids else None
    prefixes = source_prefixes or []
    for row in rows:
        replay_id = row.get("replay_id")
        if not replay_id:
            continue
        if allowed is not None and not _row_matches_format(row, allowed, prefixes):
            continue
        uploadtime = row.get("uploadtime")
        if uploadtime is not None:
            ts = datetime.fromtimestamp(int(uploadtime), tz=timezone.utc)
            if upload_after and ts < upload_after.replace(tzinfo=timezone.utc):
                continue
            if upload_before and ts > upload_before.replace(tzinfo=timezone.utc):
                continue
        refs.append(
            ArchiveRef(
                replay_id=str(replay_id),
                content_sha256=str(row.get("content_sha256")),
                log_sha256=str(row.get("log_sha256")),
                format_id=str(row.get("format_id") or ""),
                uploadtime=row.get("uploadtime"),
                object_relpath=str(row.get("object_relpath")),
                conflict=bool(row.get("conflict")),
            )
        )
    refs.sort(key=lambda r: (r.replay_id, r.content_sha256))
    return refs


def _format_filter_prefixes(format_ids: list[str]) -> list[str]:
    """Préfixes Showdown pour filtrer les replays (logique ma/mb/mc ou ID brut)."""

    prefixes: list[str] = []
    for format_id in format_ids:
        try:
            prefixes.extend(resolve_format(format_id).source_format_ids)
        except (InvalidConfiguration, NotFound):
            if str(format_id).startswith(CHAMPIONS_VGC_2026_PREFIX):
                prefixes.append(str(format_id))
    return prefixes


def _expand_format_filter(format_ids: list[str]) -> set[str]:
    """Accepte IDs logiques (`mc`) et identifiants Showdown source."""

    expanded: set[str] = set()
    for format_id in format_ids:
        expanded.add(format_id)
        try:
            support = resolve_format(format_id)
            expanded.update(support.source_format_ids)
        except (InvalidConfiguration, NotFound):
            if str(format_id).startswith(CHAMPIONS_VGC_2026_PREFIX):
                expanded.add(str(format_id))
            continue
    return expanded


def _refs_from_index(
    request: SnapshotRequest,
) -> list[ArchiveRef]:
    entries = load_index_entries()
    if not entries:
        rebuild_index_from_objects()
        entries = load_index_entries()

    format_filter = _expand_format_filter(request.format_ids)
    source_prefixes = _format_filter_prefixes(request.format_ids)
    selected: dict[str, list[ArchiveRef]] = {}
    for ref in _index_rows_to_refs(
        entries,
        format_ids=list(format_filter),
        source_prefixes=source_prefixes,
        upload_after=request.upload_after,
        upload_before=request.upload_before,
    ):
        selected.setdefault(ref.replay_id, []).append(ref)

    refs: list[ArchiveRef] = []
    for replay_id, versions in sorted(selected.items()):
        versions.sort(key=lambda r: r.content_sha256)
        if len(versions) == 1:
            refs.append(versions[0])
            continue
        selection = (request.replay_version_selection or {}).get(replay_id)
        if selection is None:
            raise Conflict(
                "versions multiples sans sélection explicite",
                details={
                    "replay_id": replay_id,
                    "versions": [v.content_sha256 for v in versions],
                },
            )
        match = next((v for v in versions if v.content_sha256 == selection), None)
        if match is None:
            raise NotFound(
                "version demandée absente",
                details={"replay_id": replay_id, "content_sha256": selection},
            )
        refs.append(match)
    refs.sort(key=lambda r: (r.replay_id, r.content_sha256))
    return refs


def create_snapshot(
    request: SnapshotRequest,
    *,
    progress_line: Callable[[str], None] | None = None,
) -> ArchiveSnapshot:
    """Figé une sélection triée de l'archive."""

    if not request.format_ids:
        raise InvalidConfiguration("format_ids requis pour un snapshot")
    refs = _refs_from_index(request)
    from pokecoach_trainer.replay_metadata import (
        extract_replay_metadata,
        metadata_to_dict,
    )

    replay_metadata: dict[str, dict[str, Any]] = {}
    total = len(refs)
    if progress_line and total:
        progress_line(f"snapshot: {total} replays dans la sélection")
    for index, ref in enumerate(refs, start=1):
        try:
            replay_metadata[ref.replay_id] = metadata_to_dict(
                extract_replay_metadata(ref)
            )
        except (ValueError, OSError):
            continue
        if progress_line and (
            index == 1 or index == total or index % 100 == 0
        ):
            progress_line(f"snapshot: métadonnées {index}/{total}")
    payload = {
        "format_ids": sorted(request.format_ids),
        "upload_after": (
            request.upload_after.isoformat() if request.upload_after else None
        ),
        "upload_before": (
            request.upload_before.isoformat() if request.upload_before else None
        ),
        "refs": [r.model_dump(mode="json") for r in refs],
        "metadata_version": 2,
        "replay_metadata": replay_metadata,
    }
    snapshot_sha256 = canonical_json_sha256(payload)
    snapshot_id = new_opaque_id("snap-")
    snapshot = ArchiveSnapshot(
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        metadata_version=2,
        format_ids=list(request.format_ids),
        upload_after=request.upload_after,
        upload_before=request.upload_before,
        refs=refs,
        replay_metadata=replay_metadata,
        bo3_links_status="partial" if replay_metadata else "unknown",
    )
    with _metadata_lock():
        snap_path = _archive_root() / _SNAPSHOTS_DIR / f"{snapshot_id}.json"
        write_json_atomic(snap_path, snapshot.model_dump(mode="json"))
    return snapshot


def read_snapshot(snapshot_id: str) -> ArchiveSnapshot:
    """Charge un snapshot persisté."""

    sid = validate_id(snapshot_id)
    path = _archive_root() / _SNAPSHOTS_DIR / f"{sid}.json"
    if not path.is_file():
        raise NotFound("snapshot introuvable", details={"snapshot_id": snapshot_id})
    return ArchiveSnapshot.model_validate(read_json(path))


def _strict_json_loads(text: str) -> dict[str, Any]:
    """Décode un objet JSON en refusant les clés dupliquées contradictoires."""

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out and out[key] != value:
                raise InvalidConfiguration(
                    "clés JSON dupliquées contradictoires",
                    details={"key": key},
                )
            out[key] = value
        return out

    return json.loads(text, object_pairs_hook=pairs_hook)


def import_legacy_logs(relative_path: str) -> ArchiveSnapshot:
    """
    Importe un JSON historique `id -> [timestamp, log]` depuis l'inbox.

    Returns:
        Snapshot couvrant les replays importés.
    """

    path = resolve_inbox_relative(relative_path)
    if not path.is_file():
        raise NotFound("fichier inbox introuvable", details={"path": relative_path})
    size = path.stat().st_size
    max_bytes = get_settings().max_inbox_bytes
    if size > max_bytes:
        raise InvalidConfiguration(
            "fichier inbox trop volumineux",
            details={"size": size, "max_bytes": max_bytes},
        )
    try:
        legacy = _strict_json_loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidConfiguration("JSON legacy invalide") from exc

    if not isinstance(legacy, dict):
        raise InvalidConfiguration("JSON legacy doit être un objet")

    source = ArchiveSource(
        provenance=f"legacy_import:{relative_path}",
        fetched_at=datetime.now(timezone.utc),
    )
    format_ids: set[str] = set()
    for replay_id, value in legacy.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise InvalidConfiguration(
                "entrée legacy invalide",
                details={"replay_id": replay_id},
            )
        timestamp, log = value
        if not isinstance(log, str):
            raise InvalidConfiguration(
                "log legacy invalide", details={"replay_id": replay_id}
            )
        fmt = replay_id.split("-")[0] if "-" in replay_id else "unknown"
        format_ids.add(fmt)
        payload = {
            "id": replay_id,
            "uploadtime": int(timestamp),
            "log": log,
            "format": fmt,
        }
        archive_replay(replay_id, payload, source)

    request = SnapshotRequest(format_ids=sorted(format_ids))
    return create_snapshot(request)


def list_replays(
    *,
    offset: int = 0,
    limit: int = 100,
    format_id: str | None = None,
    upload_after: datetime | None = None,
    upload_before: datetime | None = None,
) -> tuple[list[ArchiveRef], int]:
    """Liste paginée stable par replay_id puis hash de contenu."""

    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    entries = load_index_entries()
    format_ids = [format_id] if format_id else None
    refs = _index_rows_to_refs(
        entries,
        format_ids=format_ids,
        upload_after=upload_after,
        upload_before=upload_before,
    )
    total = len(refs)
    return refs[offset : offset + limit], total


def read_archived_object(replay_id: str, content_sha256: str) -> dict[str, Any]:
    """Charge l'enveloppe complète d'un objet (usage interne/tests)."""

    path = _archive_root() / _OBJECTS_DIR / replay_id / f"{content_sha256}.json"
    if not path.is_file():
        raise NotFound("objet archive introuvable", details={"replay_id": replay_id})
    return read_json(path)


def is_replay_archived(replay_id: str, content_sha256: str | None = None) -> bool:
    """Indique si un replay (et optionnellement un hash) est déjà indexé."""

    for row in _existing_versions(replay_id):
        if content_sha256 is None or row.get("content_sha256") == content_sha256:
            return True
    return False
