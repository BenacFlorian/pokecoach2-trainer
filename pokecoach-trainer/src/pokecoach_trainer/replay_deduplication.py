"""Déduplication exacte par hash de log (dataset prepare)."""

from __future__ import annotations

from collections.abc import Sequence

from pokecoach_trainer.schemas import ArchiveRef


def duplicate_replay_ids(
    refs: Sequence[ArchiveRef], admissible_ids: set[str]
) -> dict[str, str]:
    """Mappe alias_id -> canonical_id (plus petit replay_id admissible par log_sha256)."""

    by_hash: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for ref in sorted(refs, key=lambda r: r.replay_id):
        if ref.replay_id not in admissible_ids:
            continue
        prior = by_hash.get(ref.log_sha256)
        if prior is None:
            by_hash[ref.log_sha256] = ref.replay_id
            continue
        if ref.replay_id != prior:
            aliases[ref.replay_id] = prior
    return aliases
