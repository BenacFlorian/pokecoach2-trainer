"""Registre des formats Showdown issus de l'audit M-C."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pokecoach_trainer.errors import InvalidConfiguration, NotFound
from pokecoach_trainer.settings import PACKAGE_ROOT

_FORMATS_PATH = PACKAGE_ROOT / "configs" / "formats.json"

CHAMPIONS_VGC_2026_PREFIX = "gen9championsvgc"


def validate_champions_vgc_2026_showdown_id(
    showdown_id: str, *, field: str = "format_id"
) -> str:
    """
    Accepte uniquement les IDs Showdown Champions VGC 2026 (M-A / M-B / M-C).

    Exclut explicitement Reg F et tout identifiant hors préfixe Champions.
    """

    sid = str(showdown_id or "").strip()
    if not sid.startswith(CHAMPIONS_VGC_2026_PREFIX):
        raise InvalidConfiguration(
            f"identifiant Showdown hors périmètre Champions VGC 2026: {showdown_id!r}",
            details={
                "field": field,
                "required_prefix": CHAMPIONS_VGC_2026_PREFIX,
            },
        )
    lowered = sid.lower()
    if "regf" in lowered or sid.startswith("gen9vgc"):
        raise InvalidConfiguration(
            f"format Reg F ou VGC non-Champions refusé: {showdown_id!r}",
            details={"field": field},
        )
    return sid


def is_champions_vgc_2026_replay_id(replay_id: str) -> bool:
    """True si l'ID replay appartient au périmètre Champions VGC 2026."""

    rid = str(replay_id or "").strip()
    if not rid.startswith(CHAMPIONS_VGC_2026_PREFIX):
        return False
    lowered = rid.lower()
    return "regf" not in lowered and not rid.startswith("gen9vgc")


@dataclass(frozen=True)
class FormatSupport:
    format_id: str
    source_format_ids: tuple[str, ...]
    engine_format_id: str
    regulation: str
    collection_supported: bool
    conversion_supported: bool
    contract_fingerprint: str | None
    audit_refs: tuple[str, ...]


def _parse_entry(format_id: str, raw: dict[str, Any]) -> FormatSupport:
    source_ids = tuple(
        validate_champions_vgc_2026_showdown_id(str(s), field="source_format_ids")
        for s in (raw.get("source_format_ids") or ())
    )
    engine_id = validate_champions_vgc_2026_showdown_id(
        str(raw.get("engine_format_id", "")),
        field="engine_format_id",
    )
    return FormatSupport(
        format_id=str(raw.get("format_id", format_id)),
        source_format_ids=source_ids,
        engine_format_id=engine_id,
        regulation=str(raw.get("regulation", "")),
        collection_supported=bool(raw.get("collection_supported")),
        conversion_supported=bool(raw.get("conversion_supported")),
        contract_fingerprint=raw.get("contract_fingerprint"),
        audit_refs=tuple(raw.get("audit_refs") or ()),
    )


@lru_cache
def _load_registry() -> dict[str, FormatSupport]:
    if not _FORMATS_PATH.is_file():
        raise InvalidConfiguration(
            f"fichier formats introuvable: {_FORMATS_PATH.name}",
            details={"path": "configs/formats.json"},
        )
    data = json.loads(_FORMATS_PATH.read_text(encoding="utf-8"))
    formats = data.get("formats") or {}
    return {key: _parse_entry(key, value) for key, value in formats.items()}


def load_formats() -> dict[str, FormatSupport]:
    """Charge tous les formats enregistrés."""

    return dict(_load_registry())


def resolve_format(
    format_id: str, *, require_conversion: bool = False
) -> FormatSupport:
    """Retourne le support d'un format ou lève une erreur typée."""

    registry = _load_registry()
    key = format_id.strip().lower()
    entry = registry.get(key)
    if entry is None:
        raise NotFound(
            f"format inconnu: {format_id!r}",
            details={"format_id": format_id, "known": sorted(registry.keys())},
        )
    if require_conversion and not entry.conversion_supported:
        raise InvalidConfiguration(
            f"conversion non supportée pour {format_id!r}",
            details={"format_id": format_id},
        )
    return entry


def clear_formats_cache() -> None:
    _load_registry.cache_clear()
