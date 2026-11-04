"""Extraction de métadonnées depuis les objets d'archive (sans filtre métier)."""

from __future__ import annotations

import re
from typing import Any

from pokecoach_trainer.schemas import ArchiveRef, ReplayMetadata

_PARSER_VERSION = "replay_metadata/1"
_PLAYER_LINE = re.compile(r"^\|player\|(p[12])\|([^|]*)\|([^|]*)\|")
_TURN_LINE = re.compile(r"^\|turn\|(\d+)\|")
_POKE_LINE = re.compile(r"^\|poke\|(p[12][a-f])\|")
_SHOWTEAM_LINE = re.compile(r"^\|showteam\|(p[12])\|(.*)$")
_WIN_LINE = re.compile(r"^\|win\|([^|\n]+)")
_TIE_LINE = re.compile(r"^\|tie\|")
_INACTIVE_LINE = re.compile(r"^\|inactive\|")
_MOVE_OR_SWITCH = re.compile(r"^\|(move|switch|upkeep)\|")
_BO3_LINK = re.compile(r"^\|linked\|([^|]+)\|")


def _parse_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _count_completed_turns(lines: list[str], end_index: int) -> int | None:
    """
    Compte les tours clos avant la fin (marqueur de résolution ou abandon).

    Un tour compte lorsqu'un |move|/|switch|/|upkeep| le clôt ; un |turn|N|
    sans action avant la fin ne compte pas.
    """

    completed: set[int] = set()
    pending: int | None = None
    for idx, line in enumerate(lines[: end_index + 1]):
        turn_match = _TURN_LINE.match(line)
        if turn_match:
            pending = int(turn_match.group(1))
            continue
        if pending is not None and _MOVE_OR_SWITCH.match(line):
            completed.add(pending)
            pending = None
    if end_index >= len(lines):
        return len(completed)
    end_line = lines[end_index]
    if pending is not None and (
        _WIN_LINE.match(end_line)
        or "forfeit" in end_line.lower()
        or _TIE_LINE.match(end_line)
        or _INACTIVE_LINE.match(end_line)
    ):
        pass
    elif pending is not None:
        return None
    return len(completed)


def _termination_index(lines: list[str]) -> tuple[int, str, str | None]:
    """Retourne (index, type, gagnant éventuel)."""

    winner: str | None = None
    for idx, line in enumerate(lines):
        win = _WIN_LINE.match(line)
        if win:
            return idx, "normal_win", win.group(1).strip()
        if _TIE_LINE.match(line):
            return idx, "tie", None
        if _INACTIVE_LINE.match(line):
            return idx, "inactivity", None
        if "forfeit" in line.lower():
            return idx, "forfeit", None
    return len(lines) - 1, "unknown", winner


def _showteam_usable(body: str) -> bool:
    """True si la feuille ouverte contient six identités distinctes."""

    if not body.strip():
        return False
    parts = [p.strip() for p in body.split(",") if p.strip()]
    return len(parts) >= 6


def _team_complete(lines: list[str], side: str) -> bool:
    slots = {f"{side}{letter}" for letter in "abcdef"}
    seen: set[str] = set()
    for line in lines:
        match = _POKE_LINE.match(line)
        if not match:
            continue
        slot = match.group(1)
        if slot in slots:
            seen.add(slot)
    return len(seen) >= 6


def _parse_bo3(
    provider: dict[str, Any], replay_id: str
) -> tuple[str, str | None, list[str], str, str]:
    """Déduit le type BO1/BO3 et les liens explicites."""

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
    """
    Charge l'objet archivé et en dérive les faits observés.

    Raises:
        FileNotFoundError: objet absent.
    """

    envelope = cached if cached is not None else __import__(
        "pokecoach_trainer.archive_service",
        fromlist=["read_archived_object"],
    ).read_archived_object(ref.replay_id, ref.content_sha256)
    provider = envelope.get("provider") or {}
    log = provider.get("log")
    if not isinstance(log, str):
        raise ValueError("log replay absent")

    lines = log.splitlines()
    p1_username = p2_username = None
    p1_rating = p2_rating = None
    ots_p1 = ots_p2 = False
    linked_ids: list[str] = []
    for line in lines:
        player = _PLAYER_LINE.match(line)
        if player:
            side, name, rating_raw = player.groups()
            rating = _parse_int(rating_raw)
            if side == "p1":
                p1_username = name or None
                p1_rating = rating
            else:
                p2_username = name or None
                p2_rating = rating
            continue
        show = _SHOWTEAM_LINE.match(line)
        if show:
            side, body = show.groups()
            usable = _showteam_usable(body)
            if side == "p1":
                ots_p1 = usable
            else:
                ots_p2 = usable
            continue
        link = _BO3_LINK.match(line)
        if link:
            linked_ids.append(link.group(1).strip())

    rating_raw = provider.get("rating")
    average_rating = _parse_int(str(rating_raw)) if rating_raw is not None else None
    end_idx, termination, winner = _termination_index(lines)
    completed_turns = _count_completed_turns(lines, end_idx)

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
        p1_username=p1_username,
        p2_username=p2_username,
        p1_rating=p1_rating,
        p2_rating=p2_rating,
        average_rating=average_rating,
        winner=winner,
        termination=termination,  # type: ignore[arg-type]
        completed_turns=completed_turns,
        ots_p1=ots_p1,
        ots_p2=ots_p2,
        team_complete_p1=_team_complete(lines, "p1"),
        team_complete_p2=_team_complete(lines, "p2"),
        match_type=match_type,  # type: ignore[arg-type]
        series_id=series_id,
        game_number=None,
        linked_replay_ids=linked,
        series_link_source=link_source,
        series_link_status=link_status,  # type: ignore[arg-type]
    )


def metadata_to_dict(meta: ReplayMetadata) -> dict[str, Any]:
    """Sérialise pour persistance dans un snapshot."""

    return meta.model_dump(mode="json")
