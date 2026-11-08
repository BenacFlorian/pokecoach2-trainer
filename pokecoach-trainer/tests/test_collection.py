import httpx
import pytest

from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.collection_service import collect_replays
from pokecoach_trainer.errors import InvalidConfiguration, ProviderInvalidPayload, ProviderRateLimited
from pokecoach_trainer.schemas import CollectionConfig, ReplayFilterConfig
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout

SOURCE_FORMAT = "gen9championsvgc2026regmc"
REPLAY_ID = f"{SOURCE_FORMAT}-test1"


@pytest.fixture
def collection_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    yield tmp_path


def _mock_transport(pages: list[list[dict]], fetch_log: str = "|turn|1\n") -> httpx.MockTransport:
    search_calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "search.json" in str(request.url):
            idx = min(search_calls["count"], len(pages) - 1)
            search_calls["count"] += 1
            return httpx.Response(200, json=pages[idx])
        if request.url.path.endswith(".json"):
            replay_id = request.url.path.strip("/").replace(".json", "")
            return httpx.Response(
                200,
                json={
                    "id": replay_id,
                    "uploadtime": 1_700_000_000,
                    "log": fetch_log,
                    "format": SOURCE_FORMAT,
                },
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_search_empty_page(collection_env):
    transport = _mock_transport([[]])
    client = ShowdownReplayClient(transport=transport, sleep=lambda _: None)
    page = client.search(SOURCE_FORMAT, {"before": 999})
    assert page.replay_ids == []
    assert page.next_cursor is None
    client.close()


def test_collect_archives_replay(collection_env):
    page = [{"id": REPLAY_ID, "uploadtime": 1_700_000_000}]
    transport = _mock_transport([page, []])
    client = ShowdownReplayClient(
        transport=transport,
        sleep=lambda _: None,
        max_attempts=1,
    )
    config = CollectionConfig(
        format_ids=["mc"],
        max_new_replays=1,
        max_requests=10,
        max_duration_s=60,
        request_interval_s=1,
    )
    result = collect_replays(
        config,
        "job-test",
        client=client,
        sleep=lambda _: None,
        clock=lambda: 0.0,
    )
    assert result.new_replays == 1
    assert result.stop_reason in {"completed", "budget_reached", "coverage_incomplete"}


def test_collect_ignores_non_champions_replay_ids(collection_env):
    regf_id = "gen9vgc2026regfbo3-notchampions"
    page = [
        {"id": regf_id, "uploadtime": 1_700_000_000},
        {"id": REPLAY_ID, "uploadtime": 1_700_000_000},
    ]
    transport = _mock_transport([page, []])
    client = ShowdownReplayClient(
        transport=transport,
        sleep=lambda _: None,
        max_attempts=1,
    )
    config = CollectionConfig(
        format_ids=["mc"],
        max_new_replays=5,
        max_requests=10,
        max_duration_s=60,
        request_interval_s=1,
    )
    result = collect_replays(
        config,
        "job-test-filter",
        client=client,
        sleep=lambda _: None,
        clock=lambda: 0.0,
    )
    assert result.new_replays == 1
    assert result.failed_replays == 0


def test_fetch_invalid_json(collection_env):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(ProviderInvalidPayload):
        client.fetch(REPLAY_ID)
    client.close()


def test_client_single_attempt_on_429(collection_env):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "0"})

    client = ShowdownReplayClient(
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderRateLimited):
        client.fetch(REPLAY_ID)
    assert client.request_count == 1
    client.close()


def test_collect_replays_refuses_dataset_filters(collection_env):
    config = CollectionConfig(
        format_ids=["mc"],
        max_new_replays=1,
        dataset_filters=ReplayFilterConfig(),
    )
    with pytest.raises(InvalidConfiguration, match="collecte supervisée"):
        collect_replays(config, "job-filtered")
