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
    Reconstruit le catalogue SQLite à partir des objets publiés.

    Returns:
        Nombre d'entrées indexées.
    """

    from pokecoach_trainer.archive_catalog import initialize_catalog, rebuild_catalog_from_objects

    root = get_settings().storage_root
    initialize_catalog(root)
    return rebuild_catalog_from_objects(root)


def _existing_versions(replay_id: str) -> list[dict[str, Any]]:
    from pokecoach_trainer.archive_catalog import catalog_is_initialized, catalog_versions

    root = get_settings().storage_root
    if catalog_is_initialized(root):
        return [
            {
                "replay_id": ref.replay_id,
                "content_sha256": ref.content_sha256,
                "log_sha256": ref.log_sha256,
                "format_id": ref.format_id,
                "uploadtime": ref.uploadtime,
                "object_relpath": ref.object_relpath,
                "conflict": ref.conflict,
            }
            for ref in catalog_versions(root, replay_id)
        ]
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

    from pokecoach_trainer.collection_publication import prepare_replay, publish_replay
    from pokecoach_trainer.collection_store import CollectionStore

    root = get_settings().storage_root
    prepared = prepare_replay(root, replay_id, payload, source)
    store = CollectionStore(root)
    result = publish_replay(store, prepared, lease=None)
    if result.ref.conflict:
        for existing in _existing_versions(replay_id):
            if existing.get("content_sha256") != result.ref.content_sha256:
                return ArchiveRef(
                    replay_id=replay_id,
                    content_sha256=result.ref.content_sha256,
                    log_sha256=result.ref.log_sha256,
                    format_id=result.ref.format_id,
                    uploadtime=result.ref.uploadtime,
                    object_relpath=result.ref.object_relpath,
                    conflict=True,
                )
    return result.ref


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
    *,
    max_publication_seq: int | None = None,
) -> list[ArchiveRef]:
    from pokecoach_trainer.archive_catalog import (
        catalog_connection,
        catalog_index_rows,
        catalog_is_initialized,
    )

    root = get_settings().storage_root
    if catalog_is_initialized(root):
        entries = catalog_index_rows(root)
        if max_publication_seq is not None:
            with catalog_connection(root) as conn:
                allowed = {
                    row["content_sha256"]
                    for row in conn.execute(
                        """
                        SELECT content_sha256 FROM replay_versions
                        WHERE publication_seq <= ?
                        """,
                        (max_publication_seq,),
                    ).fetchall()
                }
            entries = [e for e in entries if e.get("content_sha256") in allowed]
    else:
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
    root = get_settings().storage_root
    from pokecoach_trainer.archive_catalog import (
        catalog_maintenance_shared,
        catalog_max_publication_seq,
        initialize_catalog,
    )

    initialize_catalog(root)
    with catalog_maintenance_shared(root):
        max_seq = catalog_max_publication_seq(root)
        refs = _refs_from_index(request, max_publication_seq=max_seq)
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


def list_campaign_replays(
    campaign_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
) -> tuple[list[ArchiveRef], int]:
    """Liste paginée des replays retenus par une campagne."""

    from pokecoach_trainer.archive_catalog import (
        catalog_connection,
        catalog_list_campaign_replays,
        catalog_maintenance_shared,
        initialize_catalog,
    )

    root = get_settings().storage_root
    initialize_catalog(root)
    with catalog_connection(root) as conn:
        if conn.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone() is None:
            raise NotFound("campagne introuvable", details={"campaign_id": campaign_id})
    with catalog_maintenance_shared(root):
        return catalog_list_campaign_replays(root, campaign_id, offset=offset, limit=limit)


def create_campaign_snapshot(campaign_id: str) -> ArchiveSnapshot:
    """Snapshot immuable limité aux replays retenus par la campagne."""

    from pokecoach_trainer.archive_catalog import (
        catalog_connection,
        catalog_list_campaign_replays,
        catalog_maintenance_shared,
        catalog_max_publication_seq,
        initialize_catalog,
    )
    from pokecoach_trainer.replay_metadata import (
        extract_replay_metadata,
        metadata_to_dict,
    )

    root = get_settings().storage_root
    initialize_catalog(root)
    with catalog_connection(root) as conn:
        row = conn.execute(
            """
            SELECT effective_config_json, validation_parser_version, validation_policy_version
            FROM campaigns WHERE campaign_id = ?
            """,
            (campaign_id,),
        ).fetchone()
        if row is None:
            raise NotFound("campagne introuvable", details={"campaign_id": campaign_id})
        effective_json = row["effective_config_json"]
        stored_parser = row["validation_parser_version"]
        stored_policy = row["validation_policy_version"]
    from pokecoach_trainer.collection_filters import filter_fingerprint
    from pokecoach_trainer.replay_metadata import REPLAY_METADATA_VERSION
    from pokecoach_trainer.replay_policy import FILTER_POLICY_VERSION
    from pokecoach_trainer.schemas import CollectionConfig

    effective = json.loads(effective_json)
    config = CollectionConfig.model_validate(effective)
    if config.dataset_filters is not None:
        if stored_parser and stored_parser != REPLAY_METADATA_VERSION:
            raise InvalidConfiguration(
                "parseur incompatible avec la campagne filtrée ; créer une nouvelle campagne"
            )
        if stored_policy and stored_policy != FILTER_POLICY_VERSION:
            raise InvalidConfiguration(
                "politique incompatible avec la campagne filtrée ; créer une nouvelle campagne"
            )
    with catalog_maintenance_shared(root):
        max_seq = catalog_max_publication_seq(root)
        refs: list = []
        offset = 0
        while True:
            page, total = catalog_list_campaign_replays(
                root,
                campaign_id,
                offset=offset,
                limit=1000,
                max_publication_seq=max_seq,
            )
            refs.extend(page)
            offset += len(page)
            if offset >= total or not page:
                break
    replay_metadata: dict[str, dict[str, Any]] = {}
    for ref in refs:
        try:
            replay_metadata[ref.replay_id] = metadata_to_dict(extract_replay_metadata(ref))
        except (ValueError, OSError, NotFound):
            if config.dataset_filters is not None:
                raise InvalidConfiguration(
                    f"métadonnées manquantes pour {ref.replay_id} dans une campagne filtrée"
                ) from None
            continue
    format_ids = sorted(effective.get("format_ids") or [])
    collection_fp = filter_fingerprint(config) if config.dataset_filters is not None else None
    dataset_filters = (
        config.dataset_filters.model_dump(mode="json") if config.dataset_filters is not None else None
    )
    payload = {
        "campaign_id": campaign_id,
        "format_ids": format_ids,
        "refs": [r.model_dump(mode="json") for r in refs],
        "metadata_version": 2,
        "replay_metadata": replay_metadata,
        "collection_filter_sha256": collection_fp,
        "collection_dataset_filters": dataset_filters,
        "collection_parser_version": stored_parser,
        "collection_policy_version": stored_policy,
    }
    snapshot_sha256 = canonical_json_sha256(payload)
    snapshot_id = new_opaque_id("snap-")
    snapshot = ArchiveSnapshot(
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        metadata_version=2,
        format_ids=format_ids,
        upload_after=None,
        upload_before=None,
        refs=refs,
        replay_metadata=replay_metadata,
        bo3_links_status="partial" if replay_metadata else "unknown",
        collection_filter_sha256=collection_fp,
        collection_dataset_filters=dataset_filters,
        collection_parser_version=stored_parser,
        collection_policy_version=stored_policy,
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
    from pokecoach_trainer.archive_catalog import catalog_is_initialized, catalog_list_replays

    root = get_settings().storage_root
    if catalog_is_initialized(root):
        from pokecoach_trainer.archive_catalog import catalog_maintenance_shared, initialize_catalog

        initialize_catalog(root)
        after_ts = (
            int(upload_after.replace(tzinfo=timezone.utc).timestamp())
            if upload_after
            else None
        )
        before_ts = (
            int(upload_before.replace(tzinfo=timezone.utc).timestamp())
            if upload_before
            else None
        )
        with catalog_maintenance_shared(root):
            return catalog_list_replays(
                root,
                offset=offset,
                limit=limit,
                format_id=format_id,
                upload_after_ts=after_ts,
                upload_before_ts=before_ts,
            )
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

    from pokecoach_trainer.archive_catalog import catalog_has_replay, initialize_catalog

    root = get_settings().storage_root
    initialize_catalog(root)
    if catalog_has_replay(root, replay_id, content_sha256):
        return True
    for row in _existing_versions(replay_id):
        if content_sha256 is None or row.get("content_sha256") == content_sha256:
            return True
    return False
