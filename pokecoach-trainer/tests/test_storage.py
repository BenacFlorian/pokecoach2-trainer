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
