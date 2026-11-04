import pytest

from pokecoach_trainer.errors import InvalidConfiguration
from pokecoach_trainer.formats import (
    clear_formats_cache,
    is_champions_vgc_2026_replay_id,
    resolve_format,
    validate_champions_vgc_2026_showdown_id,
)


def test_validate_rejects_regf_showdown_id():
    with pytest.raises(InvalidConfiguration):
        validate_champions_vgc_2026_showdown_id("gen9vgc2026regfbo3")


def test_validate_accepts_champions_mc():
    assert (
        validate_champions_vgc_2026_showdown_id("gen9championsvgc2026regmc")
        == "gen9championsvgc2026regmc"
    )


def test_is_champions_replay_id():
    assert is_champions_vgc_2026_replay_id("gen9championsvgc2026regmc-123")
    assert not is_champions_vgc_2026_replay_id("gen9vgc2026regfbo3-123")


def test_registry_ma_mb_mc_champions_only():
    clear_formats_cache()
    for key in ("ma", "mb", "mc"):
        entry = resolve_format(key)
        assert entry.engine_format_id.startswith("gen9championsvgc2026")
        for sid in entry.source_format_ids:
            assert sid.startswith("gen9championsvgc2026")
            assert "regf" not in sid.lower()
