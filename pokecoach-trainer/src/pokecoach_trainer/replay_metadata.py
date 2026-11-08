"""Extraction de métadonnées depuis les objets d'archive (sans filtre métier)."""

from __future__ import annotations

import re
from typing import Any

from pokecoach_trainer.replay_log import (
    count_completed_turns,
    parse_players,
    parse_showteam_flags,
    parse_termination,
    team_complete_from_preview,
)
from pokecoach_trainer.schemas import ArchiveRef, ReplayMetadata

REPLAY_METADATA_VERSION = "replay_metadata/2"
_PARSER_VERSION = REPLAY_METADATA_VERSION
_BO3_LINK = re.compile(r"^\|linked\|([^|]+)\|")


def _parse_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_bo3(
    provider: dict[str, Any], replay_id: str
) -> tuple[str, str | None, list[str], str, str]:
    fmt = str(provider.get("format") or replay_id.split("-")[0])
    roomid = provider.get("roomid")
    series_id = None
    linked: list[str] = []
    link_source = None
    status: str = "unknown"
    if isinstance(roomid, str) and roomid:
        series_id = roomid
        link_source = "provider.roomid"
        status = "unknown"
    if "bo3" in fmt.lower():
        match_type = "bo3"
    elif "bo1" in fmt.lower() or fmt:
        match_type = "bo1"
    else:
        match_type = "unknown"
    return match_type, series_id, linked, link_source or "unknown", status


def extract_replay_metadata(
    ref: ArchiveRef, *, cached: dict[str, Any] | None = None
) -> ReplayMetadata:
    envelope = cached if cached is not None else __import__(
        "pokecoach_trainer.archive_service",
        fromlist=["read_archived_object"],
    ).read_archived_object(ref.replay_id, ref.content_sha256)
    provider = envelope.get("provider") or {}
    log = provider.get("log")
    if not isinstance(log, str):
        raise ValueError("log replay absent")

    lines = log.splitlines()
    p1_facts, p2_facts = parse_players(lines)
    ots_p1, ots_p2 = parse_showteam_flags(lines)
    linked_ids: list[str] = []
    for line in lines:
        link = _BO3_LINK.match(line)
        if link:
            linked_ids.append(link.group(1).strip())

    rating_raw = provider.get("rating")
    average_rating = _parse_int(str(rating_raw)) if rating_raw is not None else None
    term = parse_termination(lines)
    completed_turns = count_completed_turns(lines, term.index)

    bo3_fields = _parse_bo3(provider, ref.replay_id)
    match_type, series_id, linked_from_provider, link_source, link_status = bo3_fields
    linked = sorted(set(linked_ids + linked_from_provider))
    if len(linked) >= 2:
        link_status = "confirmed"
    elif linked:
        link_status = "unknown"

    return ReplayMetadata(
        replay_id=ref.replay_id,
        raw_sha256=ref.content_sha256,
        log_sha256=ref.log_sha256,
        format_id=ref.format_id,
        parser_version=_PARSER_VERSION,
        p1_username=p1_facts.username,
        p2_username=p2_facts.username,
        p1_rating=p1_facts.rating,
        p2_rating=p2_facts.rating,
        average_rating=average_rating,
        winner=term.winner,
        termination=term.kind,  # type: ignore[arg-type]
        completed_turns=completed_turns,
        ots_p1=ots_p1,
        ots_p2=ots_p2,
        team_complete_p1=team_complete_from_preview(lines, "p1"),
        team_complete_p2=team_complete_from_preview(lines, "p2"),
        match_type=match_type,  # type: ignore[arg-type]
        series_id=series_id,
        game_number=None,
        linked_replay_ids=linked,
        series_link_source=link_source,
        series_link_status=link_status,  # type: ignore[arg-type]
    )


def metadata_to_dict(meta: ReplayMetadata) -> dict[str, Any]:
    return meta.model_dump(mode="json")
