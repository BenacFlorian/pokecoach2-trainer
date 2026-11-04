from datetime import datetime, timezone

import httpx
import pytest

from pokecoach_trainer.archive_service import archive_replay, is_replay_archived
from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_service import _load_progress, _save_progress, collect_replays
from pokecoach_trainer.schemas import ArchiveSource, CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

SOURCE_FORMAT = "gen9championsvgc2026regmc"
REPLAY_ID = f"{SOURCE_FORMAT}-resume1"


@pytest.fixture
def collection_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


def test_resume_skips_archived(collection_env):
    source = ArchiveSource(provenance="test", fetched_at=datetime.now(timezone.utc))
    archive_replay(
        REPLAY_ID,
        {
            "id": REPLAY_ID,
            "uploadtime": 1,
            "log": "|turn|1\n",
            "format": SOURCE_FORMAT,
        },
        source,
    )
    assert is_replay_archived(REPLAY_ID)
    _save_progress(
        "job-resume",
        {"pending": [REPLAY_ID], "cursors": {}, "seen_pages": []},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "search.json" in str(request.url):
            return httpx.Response(200, json=[])
        raise AssertionError("aucune requête HTTP attendue pour un replay déjà archivé")

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    config = CollectionConfig(
        format_ids=["mc"],
        max_new_replays=5,
        max_requests=5,
        max_duration_s=30,
    )
    result = collect_replays(
        config,
        "job-resume",
        client=client,
        sleep=lambda _: None,
        clock=lambda: 0.0,
    )
    assert result.new_replays == 0
    assert result.skipped_existing == 1
    progress = _load_progress("job-resume")
    assert progress["pending"] == []
    client.close()


def test_resume_after_discover_before_fetch(collection_env):
    _save_progress(
        "job-pending",
        {"pending": [REPLAY_ID], "cursors": {}, "seen_pages": []},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "search.json" in str(request.url):
            return httpx.Response(200, json=[])
        if request.url.path.endswith(".json"):
            return httpx.Response(
                200,
                json={
                    "id": REPLAY_ID,
                    "uploadtime": 1,
                    "log": "|turn|1\n",
                    "format": SOURCE_FORMAT,
                },
            )
        return httpx.Response(404)

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    config = CollectionConfig(
        format_ids=["mc"],
        max_new_replays=5,
        max_requests=5,
        max_duration_s=30,
    )
    result = collect_replays(
        config,
        "job-pending",
        client=client,
        sleep=lambda _: None,
        clock=lambda: 0.0,
    )
    assert result.new_replays == 1
    client.close()
