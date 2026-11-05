from datetime import datetime, timezone

import httpx
import pytest

from pokecoach_trainer.archive_service import archive_replay, is_replay_archived
from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_store import CollectionStore, SystemClock
from pokecoach_trainer.collection_types import SearchItem
from pokecoach_trainer.collection_workers import run_collection_worker
from pokecoach_trainer.schemas import ArchiveSource, CollectionConfig
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

SOURCE_FORMAT = "gen9championsvgc2026regmc"
REPLAY_ID = f"{SOURCE_FORMAT}-resume1"


class FakeClock(SystemClock):
    def __init__(self) -> None:
        self._wall = 1_700_000_000.0
        self._mono = 10.0
        self._boot = "boot"

    def wall(self) -> float:
        return self._wall

    def monotonic(self) -> float:
        return self._mono

    def boot_id(self) -> str:
        return self._boot


def tick(clock: FakeClock):
    def _sleep(seconds: float) -> None:
        clock._mono += max(0.0, seconds)

    return _sleep


def mc_config(**kwargs):
    base = {"format_ids": ["mc"], "requests_per_s": 20.0, "max_requests": 500}
    base.update(kwargs)
    return CollectionConfig(**base)


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
    clock = FakeClock()
    store = CollectionStore(collection_env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=REPLAY_ID, uploadtime=1, source_format=SOURCE_FORMAT)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("aucune requête HTTP attendue pour un replay déjà archivé")

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    run_collection_worker(
        collection_env,
        view.job.job_id,
        "dl-resume",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    client.close()


def test_resume_after_discover_before_fetch(collection_env):
    clock = FakeClock()
    store = CollectionStore(collection_env, clock=clock)
    view = store.start_campaign(mc_config(max_selected_replays=5))
    store.enqueue_replays(
        view.job.job_id,
        [SearchItem(replay_id=REPLAY_ID, uploadtime=1, source_format=SOURCE_FORMAT)],
    )

    def handler(request: httpx.Request) -> httpx.Response:
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
    run_collection_worker(
        collection_env,
        view.job.job_id,
        "dl-fetch",
        "download",
        client_factory=lambda _c: client,
        sleep=tick(clock),
        max_steps=1,
        clock=clock,
    )
    assert is_replay_archived(REPLAY_ID)
    client.close()
