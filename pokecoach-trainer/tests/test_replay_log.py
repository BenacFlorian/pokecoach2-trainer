from pokecoach_trainer.replay_log import (
    count_completed_turns,
    parse_players,
    parse_showteam_flags,
    parse_termination,
    team_complete_from_preview,
)
from pokecoach_trainer.replay_metadata import extract_replay_metadata
from pokecoach_trainer.schemas import ArchiveRef
from replay_log_fixtures import build_log, _packed_showteam


def _lines(log: str) -> list[str]:
    return log.splitlines()


def test_numeric_avatar_is_not_elo():
    log = "|player|p1|Alice|1500||\n|player|p2|Bob|1|1100|\n"
    p1, p2 = parse_players(_lines(log))
    assert p1.rating is None
    assert p2.rating == 1100


def test_timer_warning_is_not_inactivity_loss():
    log = build_log(termination="timer_warning", winner="Alice")
    term = parse_termination(_lines(log))
    assert term.kind == "normal_win"


def test_packed_showteam_has_six_members():
    body = _packed_showteam("A")
    log = f"|showteam|p1|{body}\n|showteam|p2|{body}\n"
    ots_p1, ots_p2 = parse_showteam_flags(_lines(log))
    assert ots_p1 and ots_p2


def test_forfeit_after_two_closed_turns():
    log = build_log(turns=2, termination="forfeit")
    term = parse_termination(_lines(log))
    assert term.kind == "forfeit"
    assert count_completed_turns(_lines(log), term.index) == 2


def test_partial_second_turn_does_not_reach_two():
    lines = [
        "|turn|1|",
        "|move|p1a|Tackle|p2a",
        "|upkeep|",
        "|turn|2|",
        "|move|p1a|Tackle|p2a",
        "|raw|Bob forfeited.",
    ]
    term = parse_termination(lines)
    assert count_completed_turns(lines, term.index) == 1


def test_chat_forfeit_does_not_end_game():
    lines = [
        "|turn|1|",
        "|move|p1a|Tackle|p2a",
        "|upkeep|",
        "|c|Alice|I will forfeit later|",
        "|win|Alice",
    ]
    term = parse_termination(lines)
    assert term.kind == "normal_win"


def test_preview_six_poke_p1_is_complete():
    log = build_log()
    assert team_complete_from_preview(_lines(log), "p1")
    five = "\n".join(f"|poke|p1|Mon{i}, L50|" for i in range(5)) + "\n|turn|1|\n"
    assert not team_complete_from_preview(_lines(five), "p1")


def test_extract_from_envelope_without_disk():
    log = build_log(p1_rating=1100, p2_rating=1200)
    ref = ArchiveRef(
        replay_id="gen9championsvgc2026regmc-mem",
        content_sha256="a" * 64,
        log_sha256="b" * 64,
        format_id="gen9championsvgc2026regmc",
        object_relpath="objects/x.json",
    )
    envelope = {
        "provider": {
            "id": ref.replay_id,
            "log": log,
            "format": ref.format_id,
            "uploadtime": 1,
        }
    }
    meta = extract_replay_metadata(ref, cached=envelope)
    assert meta.p1_rating == 1100
    assert meta.ots_p1 and meta.ots_p2
    assert meta.team_complete_p1 and meta.team_complete_p2
