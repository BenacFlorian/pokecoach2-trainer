from datetime import datetime, timezone
from pathlib import Path

import pytest

from pokecoach_trainer.collection_filters import (
    evaluate_selection,
    extract_team_summary,
    load_team_filter,
)
from pokecoach_trainer.schemas import CollectionConfig, TeamFilter

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "collection"
TEAM_A = ["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]


def _log_pokes(p1: list[str], p2: list[str]) -> str:
    lines = ["|player|p1|A|av1|1300|", "|player|p2|B|av2|1300|"]
    for side, roster in (("p1", p1), ("p2", p2)):
        for species in roster:
            lines.append(f"|poke|{side}|{species}, L50|")
    lines.append("|turn|1|")
    return "\n".join(lines) + "\n"


def test_no_team_filter_accepts_unknown_composition():
    cfg = CollectionConfig(format_ids=["mc"])
    summary = extract_team_summary({"id": "x", "log": "|turn|1|\n", "format": "f"})
    decision = evaluate_selection(cfg, uploadtime=1_700_000_000, summary=summary)
    assert decision.selected is True


def test_match_p1_or_p2_ignores_order_and_sets():
    team = TeamFilter(teams=[TEAM_A])
    cfg = CollectionConfig(format_ids=["mc"], team_filter=team)
    log = _log_pokes(list(reversed(TEAM_A)), ["x", "y", "z", "a", "b", "c"])
    summary = extract_team_summary({"id": "x", "log": log, "format": "f"})
    assert summary.p1 == tuple(sorted(TEAM_A))
    decision = evaluate_selection(cfg, uploadtime=1, summary=summary)
    assert decision.selected is True


def test_unknown_other_side_does_not_cancel_known_match():
    team = TeamFilter(teams=[TEAM_A])
    cfg = CollectionConfig(format_ids=["mc"], team_filter=team)
    log = "|player|p1|A|av||\n|poke|p1|pikachu, L50|\n|poke|p1b|eevee|\n|turn|1|\n"
    summary = extract_team_summary({"id": "x", "log": log, "format": "f"})
    assert summary.p1 is None
    decision = evaluate_selection(cfg, uploadtime=1, summary=summary)
    assert decision.reason == "team_unknown"


def test_four_revealed_pokemon_are_not_a_team():
    log = "\n".join(
        [
            "|player|p1|A|av||",
            "|poke|p1|pikachu, L50|",
            "|poke|p1|eevee, L50|",
            "|poke|p1|bulbasaur, L50|",
            "|poke|p1|charmander, L50|",
            "|turn|1|",
        ]
    )
    summary = extract_team_summary({"id": "x", "log": log + "\n", "format": "f"})
    assert summary.p1 is None


def test_distinct_starting_forms_do_not_match():
    team = TeamFilter(teams=[["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mew"]])
    cfg = CollectionConfig(format_ids=["mc"], team_filter=team)
    other = ["pikachu", "eevee", "bulbasaur", "charmander", "squirtle", "mewtwo"]
    unrelated = ["garchomp", "landorus", "incineroar", "rillaboom", "amoonguss", "fluttermane"]
    log = _log_pokes(other, unrelated)
    summary = extract_team_summary({"id": "x", "log": log, "format": "f"})
    decision = evaluate_selection(cfg, uploadtime=1, summary=summary)
    assert decision.reason == "team_mismatch"


def test_team_and_dates_are_combined():
    team = TeamFilter(teams=[TEAM_A])
    cfg = CollectionConfig(
        format_ids=["mc"],
        upload_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        upload_before=datetime(2026, 2, 1, tzinfo=timezone.utc),
        team_filter=team,
    )
    log = _log_pokes(TEAM_A, TEAM_A)
    summary = extract_team_summary({"id": "x", "log": log, "format": "f"})
    in_range = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp())
    assert evaluate_selection(cfg, uploadtime=in_range, summary=summary).selected
    assert (
        evaluate_selection(cfg, uploadtime=1_600_000_000, summary=summary).reason
        == "outside_period"
    )


def test_absent_date_filter_does_not_reject_missing_date():
    cfg = CollectionConfig(format_ids=["mc"])
    decision = evaluate_selection(cfg, uploadtime=None, summary=None)
    assert decision.selected is True


def test_empty_or_invalid_json_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"teams": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_team_filter(bad)
    bad.write_text('{"teams": [["a"]], "extra": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_team_filter(bad)
    assert load_team_filter(FIXTURES / "teams.json").teams
