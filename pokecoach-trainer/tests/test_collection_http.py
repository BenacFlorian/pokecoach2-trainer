import httpx
import pytest

from pokecoach_trainer.collection_client import ShowdownReplayClient
from pokecoach_trainer.errors import ProviderInvalidPayload, ProviderRateLimited


def test_fetch_once_single_attempt_on_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "1"})

    client = ShowdownReplayClient(
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderRateLimited):
        client.fetch_once("gen9championsvgc2026regmc-x")
    assert client.request_count == 1
    client.close()


def test_search_once_does_not_retry_on_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = ShowdownReplayClient(transport=httpx.MockTransport(handler), sleep=lambda _: None)
    with pytest.raises(Exception):
        client.search_once("gen9championsvgc2026regmc", before=999)
    assert client.request_count == 1
    client.close()


def test_fetch_once_invalid_json():
    client = ShowdownReplayClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="x")),
        sleep=lambda _: None,
    )
    with pytest.raises(ProviderInvalidPayload):
        client.fetch_once("gen9championsvgc2026regmc-x")
    client.close()
