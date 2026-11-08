from datetime import datetime, timezone

import pytest

from pokecoach_trainer.archive_service import archive_replay
from pokecoach_trainer.dataset_filters import decide_game, decide_perspective
from pokecoach_trainer.replay_metadata import extract_replay_metadata
from pokecoach_trainer.schemas import ArchiveSource, DatasetRecipe
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout
from replay_log_fixtures import build_log


@pytest.fixture
def archive_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield


def _meta(replay_id: str, log: str):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    ref = archive_replay(
        replay_id,
        {"id": replay_id, "uploadtime": 1, "log": log, "format": "gen9championsvgc2026regmc"},
        source,
    )
    return extract_replay_metadata(ref)


def test_perspective_elo_filter_reference_case(archive_env):
    recipe = DatasetRecipe()
    log = build_log(p1_rating=1300, p2_rating=1000)
    meta = _meta("gen9championsvgc2026regmc-f1", log)
    p1 = decide_perspective(meta, "p1", recipe)
    p2 = decide_perspective(meta, "p2", recipe)
    assert p1.accepted
    assert not p2.accepted
    assert p2.reason_code == "elo_below_min"


def test_average_rating_without_individual_excluded(archive_env):
    recipe = DatasetRecipe()
    log = build_log(p1_rating=None, p2_rating=None)
    meta = _meta("gen9championsvgc2026regmc-f2", log)
    meta = meta.model_copy(update={"average_rating": 1200})
    decision = decide_perspective(meta, "p1", recipe)
    assert not decision.accepted
    assert decision.reason_code == "elo_unknown_no_individual"


def test_loser_accepted_when_only_winners_false(archive_env):
    recipe = DatasetRecipe(only_winners=False)
    log = build_log(p1_rating=1300, p2_rating=1300, winner="Alice")
    meta = _meta("gen9championsvgc2026regmc-f3", log)
    loser = decide_perspective(meta, "p2", recipe)
    assert loser.accepted


def test_forfeit_two_turns_rejected(archive_env):
    recipe = DatasetRecipe()
    log = build_log(turns=2, termination="forfeit")
    meta = _meta("gen9championsvgc2026regmc-f4", log)
    game = decide_game(meta, recipe)
    assert not game.accepted
    assert game.reason_code == "forfeit_turns_insufficient"


def test_inactivity_excluded(archive_env):
    recipe = DatasetRecipe()
    log = build_log(termination="inactivity")
    meta = _meta("gen9championsvgc2026regmc-f5", log)
    game = decide_game(meta, recipe)
    assert game.reason_code == "inactivity_loss"


def test_timer_warning_with_win_is_not_inactivity(archive_env):
    recipe = DatasetRecipe()
    log = build_log(termination="timer_warning", winner="Alice")
    meta = _meta("gen9championsvgc2026regmc-timer", log)
    game = decide_game(meta, recipe)
    assert game.accepted
