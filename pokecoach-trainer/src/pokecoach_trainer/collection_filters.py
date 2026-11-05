"""Filtrage facultatif date/équipes avant publication (pur, sans I/O réseau)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pokecoach_trainer.schemas import CollectionConfig, TeamFilter
from pokecoach_trainer.storage import canonical_json_sha256

from pokecoach_trainer.collection_types import SelectionDecision, TeamSummary

_PARSER_VERSION = "collection_filters/1"
_POKE_LINE = re.compile(r"^\|poke\|(p[12][a-f])\|([^|]*)\|")
_PREVIEW_END = re.compile(r"^\|(turn|move|switch|upkeep)\|")


def _normalize_species(raw: str) -> str | None:
    slug = raw.strip().lower().replace(" ", "").replace("-", "")
    if not slug or not re.fullmatch(r"[a-z0-9]+", slug):
        return None
    return slug


def load_team_filter(path: Path) -> TeamFilter:
    """Charge `{"teams": [[six ids], ...]}` depuis un fichier JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data.keys()) != {"teams"}:
        raise ValueError("JSON d'équipes invalide: clé teams attendue seule")
    return TeamFilter.model_validate(data)


def filter_fingerprint(config: CollectionConfig) -> str:
    """Empreinte des filtres utilisateur actifs."""

    payload = {
        "upload_after": config.upload_after.isoformat() if config.upload_after else None,
        "upload_before": config.upload_before.isoformat() if config.upload_before else None,
        "team_filter": config.team_filter.model_dump(mode="json") if config.team_filter else None,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_team_summary(payload: dict[str, Any]) -> TeamSummary:
    """Extrait les six espèces initiales par joueur depuis le log preview."""

    log = payload.get("log")
    content_sha256 = canonical_json_sha256(
        {k: payload[k] for k in ("id", "uploadtime", "log", "format") if k in payload}
    )
    if not isinstance(log, str) or not log:
        return TeamSummary(
            p1=None, p2=None, parser_version=_PARSER_VERSION, content_sha256=content_sha256
        )
    lines = log.splitlines()
    preview: list[str] = []
    for line in lines:
        if _PREVIEW_END.match(line):
            break
        preview.append(line)

    def side_team(side: str) -> tuple[str, ...] | None:
        slots = {f"{side}{letter}": None for letter in "abcdef"}
        for line in preview:
            match = _POKE_LINE.match(line)
            if not match:
                continue
            slot, body = match.group(1), match.group(2)
            if not slot.startswith(side):
                continue
            if slot not in slots:
                continue
            species = _normalize_species(body.split(",")[0])
            if species is None:
                return None
            slots[slot] = species
        if any(v is None for v in slots.values()):
            return None
        return tuple(sorted(slots.values()))

    return TeamSummary(
        p1=side_team("p1"),
        p2=side_team("p2"),
        parser_version=_PARSER_VERSION,
        content_sha256=content_sha256,
    )


def _team_matches(filter_team: list[str], known: tuple[str, ...]) -> bool:
    return tuple(sorted(filter_team)) == known


def evaluate_selection(
    config: CollectionConfig,
    *,
    uploadtime: int | None,
    summary: TeamSummary | None,
) -> SelectionDecision:
    """Applique les filtres utilisateur `[after, before)` et équipes (OU)."""

    fp = filter_fingerprint(config)
    has_date = config.upload_after is not None or config.upload_before is not None
    if has_date:
        if uploadtime is None:
            return SelectionDecision(selected=False, reason="date_unknown", filter_sha256=fp)
        if config.upload_after is not None and uploadtime < int(config.upload_after.timestamp()):
            return SelectionDecision(selected=False, reason="outside_period", filter_sha256=fp)
        if config.upload_before is not None and uploadtime >= int(config.upload_before.timestamp()):
            return SelectionDecision(selected=False, reason="outside_period", filter_sha256=fp)

    if config.team_filter is None:
        return SelectionDecision(selected=True, reason="selected", filter_sha256=fp)

    if summary is None:
        return SelectionDecision(selected=False, reason="team_unknown", filter_sha256=fp)

    for team in config.team_filter.teams:
        if summary.p1 is not None and _team_matches(team, summary.p1):
            return SelectionDecision(selected=True, reason="selected", filter_sha256=fp)
        if summary.p2 is not None and _team_matches(team, summary.p2):
            return SelectionDecision(selected=True, reason="selected", filter_sha256=fp)

    if summary.p1 is None or summary.p2 is None:
        return SelectionDecision(selected=False, reason="team_unknown", filter_sha256=fp)
    return SelectionDecision(selected=False, reason="team_mismatch", filter_sha256=fp)
