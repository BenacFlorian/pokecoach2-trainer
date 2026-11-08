from datetime import datetime, timezone

from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.replay_metadata import REPLAY_METADATA_VERSION, extract_replay_metadata
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSource, CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache
from replay_log_fixtures import build_log


def test_cache_distinguishes_content_and_parser(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path)
    store.start_campaign(CollectionConfig(format_ids=["mc"]))
    ref = ArchiveRef(
        replay_id="gen9championsvgc2026regmc-cache",
        content_sha256="a" * 64,
        log_sha256="b" * 64,
        format_id="gen9championsvgc2026regmc",
        object_relpath="objects/x.json",
        uploadtime=1,
    )
    envelope = {
        "provider": {
            "id": ref.replay_id,
            "log": build_log(),
            "format": ref.format_id,
            "uploadtime": 1,
        }
    }
    meta = extract_replay_metadata(ref, cached=envelope)
    store.cache_metadata(meta, source_format=ref.format_id, uploadtime=1)
    assert store.get_cached_metadata(ref) is not None
    other = ref.model_copy(update={"content_sha256": "c" * 64})
    assert store.get_cached_metadata(other) is None
    other_parser = meta.model_copy(update={"parser_version": "replay_metadata/1"})
    store.cache_metadata(other_parser, source_format=ref.format_id, uploadtime=1)
    current = store.get_cached_metadata(ref)
    assert current is not None
    assert current.parser_version == REPLAY_METADATA_VERSION


def test_invalid_cache_json_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path)
    store.start_campaign(CollectionConfig(format_ids=["mc"]))
    ref = ArchiveRef(
        replay_id="gen9championsvgc2026regmc-badcache",
        content_sha256="a" * 64,
        log_sha256="b" * 64,
        format_id="gen9championsvgc2026regmc",
        object_relpath="objects/x.json",
    )
    with store._tx() as conn:
        conn.execute(
            """
            INSERT INTO replay_metadata_cache(
                replay_id, content_sha256, parser_version, log_sha256,
                source_format, uploadtime, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ref.replay_id,
                ref.content_sha256,
                REPLAY_METADATA_VERSION,
                ref.log_sha256,
                ref.format_id,
                1,
                "{not-json",
            ),
        )
    assert store.get_cached_metadata(ref) is None
