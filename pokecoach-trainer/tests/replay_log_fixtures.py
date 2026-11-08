"""Constructeurs de logs Showdown minimaux pour les tests dataset."""


def _packed_showteam(prefix: str = "Mon") -> str:
    entries = []
    for i in range(1, 7):
        entries.append(
            f"{prefix}{i}||Ability|Hardy|Move1,Move2,Move3,Move4||M|||50|"
        )
    return "]".join(entries)


def build_log(
    *,
    p1_name: str = "Alice",
    p2_name: str = "Bob",
    p1_rating: int | None = 1300,
    p2_rating: int | None = 1000,
    turns: int = 3,
    winner: str | None = "Alice",
    termination: str = "win",
    include_ots: bool = True,
    include_teams: bool = True,
    linked: list[str] | None = None,
    fmt: str = "gen9championsvgc2026regmc",
) -> str:
    """Assemble un log avec tours clos, OTS et fin explicite (protocole avatar/rating)."""

    p1_rating_s = "" if p1_rating is None else str(p1_rating)
    p2_rating_s = "" if p2_rating is None else str(p2_rating)
    lines = [
        f"|player|p1|{p1_name}|av1|{p1_rating_s}|",
        f"|player|p2|{p2_name}|av2|{p2_rating_s}|",
    ]
    if include_ots:
        lines.append(f"|showteam|p1|{_packed_showteam('A')}")
        lines.append(f"|showteam|p2|{_packed_showteam('B')}")
    if include_teams:
        for side, prefix in (("p1", "A"), ("p2", "B")):
            for i in range(1, 7):
                lines.append(f"|poke|{side}|{prefix}{i}, L50|")
    for turn in range(1, turns + 1):
        lines.append(f"|turn|{turn}|")
        lines.append("|move|p1a| Tackle |p2a")
        lines.append("|move|p2a| Tackle |p1a")
        lines.append("|upkeep|")
    if linked:
        for other in linked:
            lines.append(f"|linked|{other}|")
    if termination == "win" and winner:
        lines.append(f"|win|{winner}")
    elif termination == "forfeit":
        lines.append(f"|raw|{p2_name} forfeited.")
    elif termination == "inactivity":
        lines.append(f"|raw|{p2_name} lost due to inactivity.")
    elif termination == "timer_warning":
        lines.append("|inactive|Time's up!")
        if winner:
            lines.append(f"|win|{winner}")
    elif termination == "tie":
        lines.append("|tie|")
    return "\n".join(lines) + "\n"
