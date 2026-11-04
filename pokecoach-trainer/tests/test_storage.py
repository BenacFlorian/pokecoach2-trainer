import pytest

from pokecoach_trainer.settings import ensure_storage_layout, storage_path
from pokecoach_trainer.storage import (
    read_json,
    resolve_inbox_relative,
    validate_id,
    write_json_atomic,
)


def test_validate_id():
    assert validate_id("abc-123") == "abc-123"
    with pytest.raises(ValueError):
        validate_id("../x")


def test_inbox_path_traversal():
    ensure_storage_layout()
    inbox = storage_path("inbox")
    (inbox / "ok.json").write_text("{}", encoding="utf-8")
    p = resolve_inbox_relative("ok.json")
    assert p.is_file()
    with pytest.raises(ValueError):
        resolve_inbox_relative("../.env")


def test_atomic_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "pokecoach_trainer.storage.get_settings",
        lambda: type("S", (), {"storage_root": tmp_path})(),
    )
    target = tmp_path / "a" / "b.json"
    write_json_atomic(target, {"x": 1})
    assert read_json(target)["x"] == 1
    temps = list(tmp_path.glob("**/*.tmp"))
    assert temps == []


def test_storage_layout_includes_archive(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache

    clear_settings_cache()
    ensure_storage_layout()
    for name in ("archive", "base_models", "staging"):
        assert storage_path(name).is_dir()


def test_archive_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache
    from pokecoach_trainer.storage import resolve_archive_relative

    clear_settings_cache()
    ensure_storage_layout()
    archive = storage_path("archive")
    (archive / "obj.json").write_text("{}", encoding="utf-8")
    assert resolve_archive_relative("obj.json").is_file()
    with pytest.raises(ValueError):
        resolve_archive_relative("../secrets")
