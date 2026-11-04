"""Constructeurs de logs Showdown minimaux pour les tests dataset."""


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
    """Assemble un log avec tours clos, OTS et fin explicite."""

    p1_rating_s = "" if p1_rating is None else str(p1_rating)
    p2_rating_s = "" if p2_rating is None else str(p2_rating)
    lines = [
        f"|player|p1|{p1_name}|{p1_rating_s}|",
        f"|player|p2|{p2_name}|{p2_rating_s}|",
    ]
    if include_ots:
        team = ",".join(f"M{i}" for i in range(1, 7))
        lines.append(f"|showteam|p1|{team}")
        lines.append(f"|showteam|p2|{team}")
    if include_teams:
        for side in ("p1", "p2"):
            for slot in "abcdef":
                lines.append(f"|poke|{side}{slot}| Mon{slot} |")
    for turn in range(1, turns + 1):
        lines.append(f"|turn|{turn}|")
        lines.append(f"|move|p1a| Tackle |p2a")
        lines.append(f"|move|p2a| Tackle |p1a")
    if linked:
        for other in linked:
            lines.append(f"|linked|{other}|")
    if termination == "win" and winner:
        lines.append(f"|win|{winner}")
    elif termination == "forfeit":
        lines.append(f"|raw|{p2_name} forfeited.")
    elif termination == "inactivity":
        lines.append("|inactive|Time's up!")
    elif termination == "tie":
        lines.append("|tie|")
    return "\n".join(lines) + "\n"
