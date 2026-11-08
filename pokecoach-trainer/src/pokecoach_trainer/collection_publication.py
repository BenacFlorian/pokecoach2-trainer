"""Publication atomique des objets d'archive et reprise après panne."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

from pokecoach_trainer.archive_catalog import catalog_connection, catalog_has_replay, initialize_catalog, insert_replay_version
from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.collection_types import PreparedReplay, PublicationResult, TaskLease
from pokecoach_trainer.errors import InvalidConfiguration, StaleLease
from pokecoach_trainer.formats import is_champions_vgc_2026_replay_id
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource
from pokecoach_trainer.storage import append_jsonl_atomic, canonical_json_sha256, sha256_text, write_json_atomic

_ENVELOPE_VERSION = 1
_OBJECTS_DIR = "objects"
_PROVIDER_KEYS = ("id", "uploadtime", "log", "format", "p1", "p2", "rating", "roomid")
_PUBLICATION_HOOKS: dict[str, object] = {}


def set_publication_hook(name: str, callback: object | None) -> None:
    if callback is None:
        _PUBLICATION_HOOKS.pop(name, None)
    else:
        _PUBLICATION_HOOKS[name] = callback


def clear_publication_hooks() -> None:
    _PUBLICATION_HOOKS.clear()


def _provider_canonical(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: payload[k] for k in _PROVIDER_KEYS if k in payload}


def build_replay_envelope(
    replay_id: str, payload: dict[str, Any], source: ArchiveSource
) -> tuple[ArchiveRef, dict[str, Any]]:
    if payload.get("id") != replay_id:
        raise InvalidConfiguration("identifiant replay incohérent")
    if not is_champions_vgc_2026_replay_id(replay_id):
        raise InvalidConfiguration("replay hors périmètre Champions")
    log = payload.get("log")
    if not isinstance(log, str) or not log:
        raise InvalidConfiguration("log replay manquant")
    canonical = _provider_canonical(payload)
    content_sha256 = canonical_json_sha256(canonical)
    log_sha256 = sha256_text(log)
    rel = f"{_OBJECTS_DIR}/{replay_id}/{content_sha256}.json"
    format_id = str(payload.get("format") or "")
    uploadtime = payload.get("uploadtime")
    if uploadtime is not None:
        uploadtime = int(uploadtime)
    ref = ArchiveRef(
        replay_id=replay_id,
        content_sha256=content_sha256,
        log_sha256=log_sha256,
        format_id=format_id,
        uploadtime=uploadtime,
        object_relpath=rel,
        conflict=False,
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
        "conflict": False,
    }
    return ref, envelope


def prepare_replay(
    root: Path, replay_id: str, payload: dict[str, Any], source: ArchiveSource
) -> PreparedReplay:
    ref, envelope = build_replay_envelope(replay_id, payload, source)
    staging = root / "archive" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=staging, suffix=".json.tmp")
    os.close(fd)
    temp_path = Path(tmp)
    write_json_atomic(temp_path, envelope)
    temp_relpath = temp_path.relative_to(root / "archive").as_posix()
    return PreparedReplay(ref=ref, temp_relpath=temp_relpath, envelope=envelope)


def publish_replay(
    store: CollectionStore,
    prepared: PreparedReplay,
    *,
    lease: TaskLease | None,
) -> PublicationResult:
    root = store._root
    initialize_catalog(root)
    archive_root = root / "archive"
    final_path = archive_root / prepared.ref.object_relpath
    temp_path = archive_root / prepared.temp_relpath
    if not temp_path.is_file() and not final_path.is_file():
        raise InvalidConfiguration("fichier temporaire de publication absent")
    if catalog_has_replay(root, prepared.ref.replay_id, prepared.ref.content_sha256):
        if lease is not None:
            store.finalize_selection(lease, prepared.ref, new_version=False)
        return PublicationResult(ref=prepared.ref, new_replay=False, new_version=False)
    hook = _PUBLICATION_HOOKS.get("before_intent")
    if callable(hook):
        hook()
    intent_id = secrets.token_hex(8)
    with store._tx() as conn:
        if lease is not None:
            row = conn.execute(
                "SELECT lease_token FROM download_tasks WHERE replay_id=?",
                (lease.replay_id,),
            ).fetchone()
            if row is None or row["lease_token"] != lease.token:
                raise StaleLease("lease invalide pour publication")
        conn.execute(
            """
            INSERT INTO publication_intents(
                intent_id, replay_id, content_sha256, attempt_id, lease_token,
                temp_relpath, final_relpath, metadata_json, phase
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'prepared')
            """,
            (
                intent_id,
                prepared.ref.replay_id,
                prepared.ref.content_sha256,
                lease.attempt_id if lease else None,
                lease.token if lease else None,
                prepared.temp_relpath,
                prepared.ref.object_relpath,
                json.dumps(
                    {
                        "envelope": prepared.envelope,
                        "collection": {
                            "campaign_id": lease.campaign_id if lease else None,
                            "attempt_id": lease.attempt_id if lease else None,
                        },
                    },
                    sort_keys=True,
                ),
            ),
        )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if not final_path.exists() and temp_path.is_file():
        hook = _PUBLICATION_HOOKS.get("before_rename")
        if callable(hook):
            hook()
        try:
            os.replace(temp_path, final_path)
        except OSError:
            if lease is not None:
                with store._tx() as conn:
                    conn.execute(
                        """
                        UPDATE download_tasks SET state='pending', owner_worker=NULL,
                               lease_token=NULL, reserved_slot=0
                        WHERE replay_id=? AND lease_token=?
                        """,
                        (lease.replay_id, lease.token),
                    )
                    conn.execute(
                        "DELETE FROM publication_intents WHERE intent_id=?", (intent_id,)
                    )
            raise
    hook = _PUBLICATION_HOOKS.get("after_rename_before_catalog")
    if callable(hook):
        hook()
    new_replay = not catalog_has_replay(root, prepared.ref.replay_id)
    new_version = not catalog_has_replay(
        root, prepared.ref.replay_id, prepared.ref.content_sha256
    )
    source = ArchiveSource.model_validate(prepared.envelope["source"])
    ref = prepared.ref
    if new_version:
        conflict = catalog_has_replay(root, ref.replay_id)
        ref = ref.model_copy(update={"conflict": conflict})
        insert_replay_version(
            root,
            ref=ref,
            source=source,
            provider_format=ref.format_id,
        )
        index_path = archive_root / "index" / "replays.jsonl"
        append_jsonl_atomic(
            index_path,
            {
                "replay_id": ref.replay_id,
                "content_sha256": ref.content_sha256,
                "log_sha256": ref.log_sha256,
                "format_id": ref.format_id,
                "uploadtime": ref.uploadtime,
                "object_relpath": ref.object_relpath,
                "conflict": ref.conflict,
            },
        )
    hook = _PUBLICATION_HOOKS.get("after_catalog_before_finalize")
    if callable(hook):
        hook()
    with store._tx() as conn:
        filtered = False
        if lease is not None:
            camp = conn.execute(
                "SELECT effective_config_json FROM campaigns WHERE campaign_id=?",
                (lease.campaign_id,),
            ).fetchone()
            if camp is not None:
                from pokecoach_trainer.schemas import CollectionConfig

                cfg = CollectionConfig.model_validate_json(camp["effective_config_json"])
                filtered = cfg.dataset_filters is not None
        if lease is None or not filtered:
            conn.execute("DELETE FROM publication_intents WHERE intent_id=?", (intent_id,))
            if lease is not None:
                conn.execute(
                    """
                    UPDATE download_tasks SET state='archived', reserved_slot=0
                    WHERE replay_id=? AND lease_token=?
                    """,
                    (lease.replay_id, lease.token),
                )
        if lease is not None:
            conn.execute(
                "UPDATE attempts SET new_published = new_published + ? WHERE attempt_id=?",
                (1 if new_version else 0, lease.attempt_id),
            )
    return PublicationResult(
        ref=ref, new_replay=new_replay, new_version=new_version
    )


def recover_publications(store: CollectionStore, *, limit: int = 100) -> int:
    recovered = 0
    root = store._root
    with catalog_connection(root) as conn:
        rows = conn.execute(
            "SELECT * FROM publication_intents LIMIT ?", (limit,)
        ).fetchall()
    for row in rows:
        final_path = root / "archive" / row["final_relpath"]
        temp_path = root / "archive" / row["temp_relpath"]
        if not final_path.is_file() and temp_path.is_file():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_path, final_path)
        if final_path.is_file():
            metadata = json.loads(row["metadata_json"])
            envelope = metadata.get("envelope") if "archive_ref" not in metadata else metadata
            if envelope is None:
                envelope = metadata
            ref = ArchiveRef.model_validate(envelope["archive_ref"])
            source = ArchiveSource.model_validate(envelope["source"])
            new_version = not catalog_has_replay(root, ref.replay_id, ref.content_sha256)
            if new_version:
                insert_replay_version(
                    root, ref=ref, source=source, provider_format=ref.format_id
                )
            collection = metadata.get("collection") or {}
            campaign_id = collection.get("campaign_id")
            attempt_id = collection.get("attempt_id")
            if campaign_id:
                store.finalize_recovered_publication(
                    campaign_id=campaign_id,
                    attempt_id=attempt_id,
                    ref=ref,
                    new_version=new_version,
                )
            with store._tx() as conn:
                conn.execute(
                    "DELETE FROM publication_intents WHERE intent_id=?", (row["intent_id"],)
                )
            recovered += 1
        else:
            with store._tx() as conn:
                conn.execute(
                    "DELETE FROM publication_intents WHERE intent_id=?", (row["intent_id"],)
                )
    recovered += _recover_pending_publications(store, limit=limit)
    return recovered


def _recover_pending_publications(store: CollectionStore, *, limit: int) -> int:
    recovered = 0
    with catalog_connection(store._root) as conn:
        rows = conn.execute(
            """
            SELECT cr.campaign_id, cr.replay_id, cr.version_content_sha256, cr.log_sha256,
                   c.attempt_id, rv.object_relpath, rv.provider_format
            FROM campaign_replays cr
            JOIN campaigns c ON c.campaign_id = cr.campaign_id
            JOIN replay_versions rv
              ON rv.replay_id = cr.replay_id
             AND rv.content_sha256 = cr.version_content_sha256
            WHERE cr.decision = 'pending_publication'
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    for row in rows:
        ref = ArchiveRef(
            replay_id=row["replay_id"],
            content_sha256=row["version_content_sha256"],
            log_sha256=row["log_sha256"] or "",
            format_id=row["provider_format"] or "",
            object_relpath=row["object_relpath"],
        )
        if store.finalize_recovered_publication(
            campaign_id=row["campaign_id"],
            attempt_id=row["attempt_id"],
            ref=ref,
            new_version=False,
        ):
            recovered += 1
    return recovered
