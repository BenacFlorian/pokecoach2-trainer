#!/usr/bin/env python3
"""Snapshot figé limité à N refs (smoke prepare, sans parcourir tout le catalogue)."""

from __future__ import annotations

import argparse
import sys

from pokecoach_trainer.archive_service import (
    _SNAPSHOTS_DIR,
    _archive_root,
    _metadata_lock,
    _refs_from_index,
)
from pokecoach_trainer.replay_metadata import extract_replay_metadata, metadata_to_dict
from pokecoach_trainer.schemas import ArchiveSnapshot, SnapshotRequest
from pokecoach_trainer.settings import ensure_storage_layout, get_settings, load_dotenv_explicit
from pokecoach_trainer.storage import canonical_json_sha256, new_opaque_id, write_json_atomic


def _select_refs(
    all_refs: list,
    *,
    limit: int,
    require_both_ots: bool,
) -> tuple[list, dict, int, int]:
    """Retourne (refs retenues, replay_metadata, parcourus, erreurs parse)."""

    target = max(1, limit)
    refs: list = []
    replay_metadata: dict = {}
    scanned = 0
    errors = 0
    candidates = all_refs if require_both_ots else all_refs[:target]

    for ref in candidates:
        if len(refs) >= target:
            break
        scanned += 1
        try:
            meta = extract_replay_metadata(ref)
        except (ValueError, OSError):
            errors += 1
            continue
        if require_both_ots and not (meta.ots_p1 and meta.ots_p2):
            continue
        refs.append(ref)
        replay_metadata[ref.replay_id] = metadata_to_dict(meta)
        if require_both_ots and len(refs) % 50 == 0:
            print(
                f"  retenus {len(refs)}/{target} "
                f"(parcourus {scanned}/{len(all_refs)})",
                file=sys.stderr,
                flush=True,
            )

    return refs, replay_metadata, scanned, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crée un snapshot smoke (refs limitées)")
    parser.add_argument("--limit", type=int, default=300, help="nombre max de refs")
    parser.add_argument(
        "--format-id",
        action="append",
        dest="format_ids",
        default=["mc"],
        help="format_ids snapshot (répétable)",
    )
    parser.add_argument(
        "--require-both-ots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ne garder que les replays OTS des deux côtés (défaut: oui)",
    )
    args = parser.parse_args(argv)

    load_dotenv_explicit()
    ensure_storage_layout()
    root = get_settings().storage_root.resolve()
    print(f"storage_root={root}", file=sys.stderr, flush=True)

    request = SnapshotRequest(format_ids=list(args.format_ids))
    all_refs = _refs_from_index(request)
    if not all_refs:
        print(
            "erreur: 0 replay pour ce format — vérifier TRAINER_STORAGE_ROOT "
            f"(attendu: archive avec catalogue M-C). storage={root}",
            file=sys.stderr,
        )
        return 1

    refs, replay_metadata, scanned, errors = _select_refs(
        all_refs,
        limit=args.limit,
        require_both_ots=args.require_both_ots,
    )

    if args.require_both_ots:
        print(
            f"snapshot smoke OTS: {len(refs)}/{args.limit} retenus "
            f"après {scanned}/{len(all_refs)} parcourus "
            f"({errors} erreurs lecture)",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"snapshot smoke: {len(refs)}/{len(all_refs)} refs (sans filtre OTS)",
            file=sys.stderr,
            flush=True,
        )

    if not refs:
        print("erreur: aucune ref retenue pour ce snapshot", file=sys.stderr)
        return 1

    if args.require_both_ots and len(refs) < args.limit:
        print(
            f"avertissement: seulement {len(refs)} replays OTS "
            f"(demandé {args.limit}) — snapshot créé avec le maximum disponible",
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "format_ids": sorted(set(args.format_ids)),
        "upload_after": None,
        "upload_before": None,
        "refs": [r.model_dump(mode="json") for r in refs],
        "metadata_version": 2,
        "replay_metadata": replay_metadata,
    }
    snapshot_id = new_opaque_id("snap-")
    snapshot = ArchiveSnapshot(
        snapshot_id=snapshot_id,
        snapshot_sha256=canonical_json_sha256(payload),
        metadata_version=2,
        format_ids=list(args.format_ids),
        refs=refs,
        replay_metadata=replay_metadata,
        bo3_links_status="partial",
    )
    with _metadata_lock():
        path = _archive_root() / _SNAPSHOTS_DIR / f"{snapshot_id}.json"
        write_json_atomic(path, snapshot.model_dump(mode="json"))
    print(snapshot_id)
    print(f"path={path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
