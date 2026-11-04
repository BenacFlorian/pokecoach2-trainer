"""Client HTTP Showdown pour recherche et téléchargement de replays."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

from pokecoach_trainer.errors import InvalidConfiguration
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
            follow_redirects=True,
        )
        self.request_count = 0

    def close(self) -> None:
        self._client.close()

    def _request(self, method: str, url: str) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            self.request_count += 1
            try:
                response = self._client.request(method, url)
            except httpx.RequestError as exc:
                last_exc = exc
                if attempt >= self._max_attempts:
                    raise InvalidConfiguration(f"erreur réseau Showdown: {exc}") from exc
                self._sleep(min(2.0 ** (attempt - 1), 8.0))
                continue
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", "1"))
                self._sleep(retry_after)
                if attempt >= self._max_attempts:
                    raise InvalidConfiguration("Showdown a renvoyé 429 trop souvent")
                continue
            if response.status_code >= 500:
                if attempt >= self._max_attempts:
                    raise InvalidConfiguration(
                        f"Showdown indisponible ({response.status_code})"
                    )
                self._sleep(min(2.0 ** (attempt - 1), 8.0))
                continue
            if response.status_code >= 400:
                raise InvalidConfiguration(
                    f"requête Showdown refusée ({response.status_code})",
                    details={"url": url},
                )
            return response
        raise InvalidConfiguration("échec Showdown") from last_exc

    def search(self, format_id: str, cursor: dict[str, Any] | None = None) -> SearchPage:
        """
        Interroge l'API de recherche (`before` décroissant).

        Le curseur externe reprend la forme auditée VGC-Bench.
        """

        state = dict(cursor or {})
        before = int(state.get("before", 2_000_000_000))
        url = f"{self._site}/search.json?format={format_id}&before={before + 1}"
        response = self._request("GET", url)
        try:
            items = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise InvalidConfiguration("JSON de recherche invalide") from exc
        if not isinstance(items, list):
            raise InvalidConfiguration("réponse de recherche inattendue")
        if not items:
            return SearchPage(replay_ids=[], next_cursor=None, page_uploadtimes=[])
        uploadtimes = [int(item.get("uploadtime", 0)) for item in items if isinstance(item, dict)]
        oldest = int(items[-1]["uploadtime"])
        replay_ids = [
            str(item["id"])
            for item in items
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and str(item["id"]).startswith(format_id)
        ]
        next_cursor = {**state, "before": oldest, "format_id": format_id}
        return SearchPage(
            replay_ids=replay_ids,
            next_cursor=next_cursor,
            page_uploadtimes=uploadtimes,
        )

    def fetch(self, replay_id: str) -> dict[str, Any]:
        """Télécharge la réponse JSON complète d'un replay."""

        url = f"{self._site}/{replay_id}.json"
        response = self._request("GET", url)
        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise InvalidConfiguration(
                "JSON replay invalide",
                details={"replay_id": replay_id},
            ) from exc
        if not isinstance(payload, dict):
            raise InvalidConfiguration("replay inattendu", details={"replay_id": replay_id})
        if payload.get("id") != replay_id:
            raise InvalidConfiguration(
                "identifiant replay incohérent",
                details={"requested": replay_id, "provider": payload.get("id")},
            )
        return payload
