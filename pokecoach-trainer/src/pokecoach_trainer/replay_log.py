"""Lecture pure du protocole Showdown pour preview, OTS et fin de partie."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_PLAYER_LINE = re.compile(r"^\|player\|(p[12])\|([^|]*)\|([^|]*)\|([^|]*)\|?")
_POKE_SIDE = re.compile(r"^\|poke\|(p[12])\|")
_CLEARPOKE = re.compile(r"^\|clearpoke\|")
_SHOWTEAM = re.compile(r"^\|showteam\|(p[12])\|(.*)$")
_TURN = re.compile(r"^\|turn\|(\d+)")
_MOVE_SWITCH_UPKEEP = re.compile(r"^\|(move|switch|upkeep)\|")
_WIN = re.compile(r"^\|win\|([^|\n]+)")
_TIE = re.compile(r"^\|tie\|")
_INACTIVE = re.compile(r"^\|inactive\|")
_RAW = re.compile(r"^\|raw\|(.*)$")
_BATTLE_START = re.compile(r"^\|(turn|move|switch|upkeep)\|")


def _parse_int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class PlayerFacts:
    username: str | None
    rating: int | None


def parse_players(lines: list[str]) -> tuple[PlayerFacts, PlayerFacts]:
    """Lit name/avatar/rating ; le rating est le 5e champ, jamais l'avatar seul."""

    p1 = PlayerFacts(None, None)
    p2 = PlayerFacts(None, None)
    for line in lines:
        match = _PLAYER_LINE.match(line)
        if not match:
            continue
        side, name, _avatar, rating_raw = match.groups()
        rating = _parse_int(rating_raw or "")
        facts = PlayerFacts(name or None, rating)
        if side == "p1":
            p1 = PlayerFacts(facts.username or p1.username, facts.rating if facts.rating is not None else p1.rating)
        else:
            p2 = PlayerFacts(facts.username or p2.username, facts.rating if facts.rating is not None else p2.rating)
    return p1, p2


def preview_poke_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    """Membres preview par côté jusqu'à clearpoke ou début de combat."""

    p1: list[str] = []
    p2: list[str] = []
    for line in lines:
        if _CLEARPOKE.match(line):
            p1, p2 = [], []
            continue
        if _BATTLE_START.match(line):
            break
        side_match = _POKE_SIDE.match(line)
        if side_match:
            side = side_match.group(1)
            (p1 if side == "p1" else p2).append(line)
    return p1, p2


def team_complete_from_preview(lines: list[str], side: str) -> bool:
    p1, p2 = preview_poke_lines(lines)
    count = len(p1 if side == "p1" else p2)
    return count >= 6


def extract_preview_species(line: str) -> str | None:
    body = line.split("|", 3)[-1] if line.count("|") >= 3 else ""
    species = body.split(",", 1)[0].strip()
    if not species:
        return None
    slug = species.lower().replace(" ", "").replace("-", "")
    if not slug or not re.fullmatch(r"[a-z0-9]+", slug):
        return None
    return slug


def preview_team_tuple(lines: list[str], side: str) -> tuple[str, ...] | None:
    p1, p2 = preview_poke_lines(lines)
    raw = p1 if side == "p1" else p2
    species: list[str] = []
    for line in raw:
        sp = extract_preview_species(line)
        if sp is None:
            return None
        species.append(sp)
    if len(species) != 6:
        return None
    return tuple(sorted(species))


def _packed_species_identity(entry: str) -> str | None:
    entry = entry.strip()
    if not entry:
        return None
    name = entry.split("|", 1)[0].strip()
    if not name:
        return None
    slug = name.lower().replace(" ", "").replace("-", "")
    if not slug or not re.fullmatch(r"[a-z0-9]+", slug):
        return None
    return slug


def showteam_usable(body: str) -> bool:
    """Six entrées packed séparées par `]` avec identité d'espèce exploitable."""

    if not body.strip():
        return False
    entries = [part for part in body.split("]") if part.strip()]
    if len(entries) < 6:
        return False
    identities = 0
    for entry in entries[:6]:
        if _packed_species_identity(entry) is not None:
            identities += 1
    return identities >= 6


def parse_showteam_flags(lines: list[str]) -> tuple[bool, bool]:
    ots_p1 = ots_p2 = False
    for line in lines:
        match = _SHOWTEAM.match(line)
        if not match:
            continue
        side, body = match.groups()
        usable = showteam_usable(body)
        if side == "p1":
            ots_p1 = usable
        else:
            ots_p2 = usable
    return ots_p1, ots_p2


@dataclass(frozen=True, slots=True)
class TerminationFacts:
    index: int
    kind: Literal["normal_win", "tie", "forfeit", "inactivity", "unknown"]
    winner: str | None


def _effective_inactivity(raw_body: str) -> bool:
    lower = raw_body.lower()
    return "lost due to inactivity" in lower or "forfeited due to inactivity" in lower


def _effective_forfeit(raw_body: str) -> bool:
    lower = raw_body.lower()
    if "lost due to inactivity" in lower:
        return False
    return " forfeited." in lower or lower.endswith("forfeited.")


def parse_termination(lines: list[str]) -> TerminationFacts:
    """Priorité en fin de partie : inactivité > abandon > victoire ; ignore timer seul."""

    last_win: tuple[int, str] | None = None
    last_tie: int | None = None
    last_forfeit: int | None = None
    last_inactivity: int | None = None
    for idx, line in enumerate(lines):
        win = _WIN.match(line)
        if win:
            last_win = (idx, win.group(1).strip())
            continue
        if _TIE.match(line):
            last_tie = idx
            continue
        raw = _RAW.match(line)
        if raw:
            body = raw.group(1)
            if _effective_inactivity(body):
                last_inactivity = idx
            elif _effective_forfeit(body):
                last_forfeit = idx
            continue
        if _INACTIVE_LINE_EFFECTIVE(line):
            continue
    if last_inactivity is not None:
        return TerminationFacts(last_inactivity, "inactivity", None)
    if last_forfeit is not None:
        winner = last_win[1] if last_win else None
        return TerminationFacts(last_forfeit, "forfeit", winner)
    if last_win is not None:
        return TerminationFacts(last_win[0], "normal_win", last_win[1])
    if last_tie is not None:
        return TerminationFacts(last_tie, "tie", None)
    return TerminationFacts(len(lines) - 1, "unknown", None)


def _INACTIVE_LINE_EFFECTIVE(line: str) -> bool:
    return bool(_INACTIVE.match(line))


def count_completed_turns(lines: list[str], end_index: int) -> int | None:
    completed: set[int] = set()
    pending: int | None = None
    for idx, line in enumerate(lines[: end_index + 1]):
        turn_match = _TURN.match(line)
        if turn_match:
            if pending is not None:
                completed.add(pending)
            pending = int(turn_match.group(1))
            continue
        if pending is not None and line.startswith("|upkeep|"):
            completed.add(pending)
            pending = None
    return len(completed)
