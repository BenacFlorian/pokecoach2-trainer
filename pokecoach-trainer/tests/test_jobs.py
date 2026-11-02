import pytest

from pokecoach_trainer.jobs import acquire_training_lock, release_training_lock


def test_lock_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

    clear_settings_cache()
    ensure_storage_layout()
    acquire_training_lock("job-a")
    with pytest.raises(RuntimeError):
        acquire_training_lock("job-b")
    release_training_lock("job-a")
    acquire_training_lock("job-c")
