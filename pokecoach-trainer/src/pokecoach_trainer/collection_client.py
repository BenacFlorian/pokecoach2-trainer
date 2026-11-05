"""Client HTTP Showdown pour recherche et téléchargement de replays."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from pokecoach_trainer.collection_types import DiscoveryPage, SearchItem
from pokecoach_trainer.errors import (
    InvalidConfiguration,
    ProviderInvalidPayload,
    ProviderRateLimited,
    ProviderUnavailable,
)
from pokecoach_trainer.schemas import SearchPage

DEFAULT_SHOWDOWN_SITE = "https://replay.pokemonshowdown.com"
SleepFn = Callable[[float], None]
ClockFn = Callable[[], float]


class ShowdownReplayClient:
    """Client Showdown avec transport injectable (tests sans réseau)."""

    def __init__(
        self,
        *,
        site: str = DEFAULT_SHOWDOWN_SITE,
        timeout_s: float = 30.0,
        max_attempts: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleep: SleepFn | None = None,
        clock: ClockFn | None = None,
    ) -> None:
        self._site = site.rstrip("/")
        self._timeout_s = timeout_s
        self._max_attempts = max(1, min(max_attempts, 5))
        self._transport = transport
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._client = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_s),
            follow_redirects=False,
        )
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def _request_once(self, method: str, url: str) -> httpx.Response:
        self.request_count += 1
        try:
            response = self._client.request(method, url)
        except httpx.RequestError as exc:
            raise ProviderUnavailable(f"erreur réseau Showdown: {exc}") from exc
        if response.status_code in {301, 302, 307, 308}:
            raise ProviderUnavailable("redirection Showdown inattendue")
        if response.status_code == 429:
            raise ProviderRateLimited(
                "Showdown a renvoyé 429",
                details={"retry_after": response.headers.get("Retry-After")},
            )
        if response.status_code in {404, 410}:
            raise ProviderUnavailable(f"replay indisponible ({response.status_code})")
        if response.status_code in {401, 403}:
            raise InvalidConfiguration(f"accès Showdown refusé ({response.status_code})")
        if response.status_code >= 500:
            raise ProviderUnavailable(f"Showdown indisponible ({response.status_code})")
        if response.status_code >= 400:
            raise ProviderInvalidPayload(
                f"requête Showdown refusée ({response.status_code})",
                details={"url": url},
            )
        return response

    def search_once(self, source_format: str, *, before: int) -> DiscoveryPage:
        url = f"{self._site}/search.json?format={source_format}&before={before + 1}"
        response = self._request_once("GET", url)
        try:
            items_raw = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ProviderInvalidPayload("JSON de recherche invalide") from exc
        if not isinstance(items_raw, list):
            raise ProviderInvalidPayload("réponse de recherche inattendue")
        if not items_raw:
            return DiscoveryPage(items=tuple(), next_before=None, exhausted=True, blocked_reason=None)
        page_items: list[SearchItem] = []
        uploadtimes: list[int] = []
        for item in items_raw:
            if not isinstance(item, dict):
                continue
            replay_id = item.get("id")
            uploadtime = item.get("uploadtime")
            if not isinstance(replay_id, str) or not str(replay_id).startswith(source_format):
                continue
            ts = int(uploadtime)
            uploadtimes.append(ts)
            page_items.append(
                SearchItem(replay_id=replay_id, uploadtime=ts, source_format=source_format)
            )
        oldest = min(uploadtimes) if uploadtimes else None
        return DiscoveryPage(
            items=tuple(page_items),
            next_before=oldest,
            exhausted=False,
            blocked_reason=None,
        )

    def fetch_once(self, replay_id: str) -> dict[str, Any]:
        url = f"{self._site}/{replay_id}.json"
        response = self._request_once("GET", url)
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise ProviderInvalidPayload(
                "JSON replay invalide",
                details={"replay_id": replay_id},
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderInvalidPayload("replay inattendu", details={"replay_id": replay_id})
        if payload.get("id") != replay_id:
            raise ProviderInvalidPayload(
                "identifiant replay incohérent",
                details={"requested": replay_id, "provider": payload.get("id")},
            )
        return payload

    def search(self, format_id: str, cursor: dict[str, Any] | None = None) -> SearchPage:
        """Compatibilité : une page via `search_once`."""

        state = dict(cursor or {})
        before = int(state.get("before", 2_000_000_000))
        page = self.search_once(format_id, before=before)
        replay_ids = [item.replay_id for item in page.items]
        uploadtimes = [item.uploadtime for item in page.items]
        next_cursor = None
        if page.next_before is not None and not page.exhausted:
            next_cursor = {**state, "before": page.next_before, "format_id": format_id}
        return SearchPage(
            replay_ids=replay_ids,
            next_cursor=next_cursor,
            page_uploadtimes=uploadtimes,
        )

    def fetch(self, replay_id: str) -> dict[str, Any]:
        return self.fetch_once(replay_id)

    def _request(self, method: str, url: str) -> httpx.Response:
        return self._request_once(method, url)
