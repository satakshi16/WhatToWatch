"""Async client for Wikidata SPARQL and the Wikipedia REST summary API (P1-04b, D-011).

- Wikimedia User-Agent policy: every request sends
  `WhatToWatch/<version> (non-commercial; contact: <WIKIMEDIA_CONTACT>)`. The contact
  comes from config; the client refuses to start without it.
- Same resilience policy as the TMDB client (D-009): explicit timeout, tenacity retries
  with exponential backoff + jitter on network errors, 5xx and 429 (honoring
  `Retry-After`, clamped), semaphore held per attempt only.

Usage:
    async with WikimediaClient.from_settings(get_settings()) as wm:
        rows = await wm.sparql("SELECT ...")
        summary = await wm.page_summary("enwiki", "Inception")
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self
from urllib.parse import quote

import httpx
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app import __version__
from app.core.config import Settings
from app.core.logging import get_logger
from app.ingest.tmdb_client import MAX_RETRY_AFTER_SECONDS, parse_retry_after

log = get_logger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
SleepFn = Callable[[float], Awaitable[None]]


# --- Errors --------------------------------------------------------------------


class WikimediaError(Exception):
    """Base class for Wikimedia client errors."""


class WikimediaConfigError(WikimediaError):
    """Client misconfigured (e.g. no `WIKIMEDIA_CONTACT`)."""


class WikimediaHTTPError(WikimediaError):
    """Non-retryable HTTP error response (4xx other than 429)."""

    def __init__(self, status_code: int, url: str) -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"Wikimedia {status_code} on {url}")


class WikimediaUnavailableError(WikimediaError):
    """Retries exhausted on 5xx / 429 / network errors."""

    def __init__(self, url: str, attempts: int, last_status: int | None) -> None:
        self.url = url
        self.attempts = attempts
        self.last_status = last_status
        cause = f"HTTP {last_status}" if last_status is not None else "network error"
        super().__init__(f"Wikimedia unavailable on {url} after {attempts} attempts ({cause})")


class WikimediaResponseError(WikimediaError):
    """Response body is not valid JSON or has an unexpected shape."""


class _RetryableStatus(Exception):  # noqa: N818 - internal control-flow signal
    def __init__(self, status_code: int, retry_after: float | None) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"retryable HTTP {status_code}")


# --- Helpers -------------------------------------------------------------------


def wiki_language(site: str) -> str:
    """`enwiki` → `en`."""
    if not site.endswith("wiki") or len(site) <= len("wiki"):
        raise ValueError(f"Not a Wikipedia database name: {site!r}")
    return site.removesuffix("wiki")


def wiki_site(language: str) -> str:
    """`en` → `enwiki`."""
    return f"{language}wiki"


def wikipedia_base_url(site: str) -> str:
    return f"https://{wiki_language(site)}.wikipedia.org/"


def article_url(site: str, title: str) -> str:
    """Public article URL, for CC BY-SA attribution."""
    return f"{wikipedia_base_url(site)}wiki/{quote(title.replace(' ', '_'), safe='')}"


@dataclass(frozen=True)
class WikiSummary:
    """Lead-section summary from `/api/rest_v1/page/summary/{title}`."""

    site: str
    title: str
    revision_id: int | None
    extract: str
    page_type: str

    @property
    def usable(self) -> bool:
        """A real article with text (not a disambiguation page or an empty extract)."""
        return self.page_type == "standard" and bool(self.extract.strip())


def user_agent(contact: str, version: str = __version__) -> str:
    return f"WhatToWatch/{version} (non-commercial; contact: {contact})"


def _status_of(exc: BaseException | None) -> int | None:
    return exc.status_code if isinstance(exc, _RetryableStatus) else None


# --- Client --------------------------------------------------------------------


class WikimediaClient:
    def __init__(
        self,
        *,
        contact: str | None,
        sparql_url: str = WIKIDATA_SPARQL_URL,
        timeout: float = 30.0,
        max_attempts: int = 4,
        max_concurrency: int = 4,
        backoff_initial: float = 0.5,
        backoff_max: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        contact = (contact or "").strip()
        if not contact:
            raise WikimediaConfigError(
                "Set WIKIMEDIA_CONTACT (email or URL); Wikimedia's User-Agent policy requires it"
            )
        if any(c in contact for c in "\r\n"):
            raise WikimediaConfigError("WIKIMEDIA_CONTACT must be a single line")
        if max_attempts < 1:
            raise WikimediaConfigError("max_attempts must be >= 1")
        self.user_agent = user_agent(contact)
        self._sparql_url = sparql_url
        self._http = httpx.AsyncClient(
            headers={"User-Agent": self.user_agent, "Api-User-Agent": self.user_agent},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        self._max_attempts = max_attempts
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._backoff = wait_exponential_jitter(
            multiplier=backoff_initial, jitter=backoff_initial, max=backoff_max
        )
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> WikimediaClient:
        kwargs: dict[str, Any] = {
            "contact": settings.wikimedia_contact,
            "timeout": settings.wikimedia_timeout_seconds,
            "max_attempts": settings.wikimedia_max_attempts,
            "max_concurrency": settings.wikimedia_concurrency,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # --- Endpoints ---------------------------------------------------------------

    async def sparql(self, query: str) -> list[dict[str, str]]:
        """Run a SELECT query; returns bindings flattened to `{variable: value}`."""
        response = await self._send(
            "POST",
            self._sparql_url,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )
        _raise_for_status(response)
        body = _json(response)
        try:
            bindings = body["results"]["bindings"]
            return [{var: str(cell["value"]) for var, cell in row.items()} for row in bindings]
        except (KeyError, TypeError) as exc:
            raise WikimediaResponseError("Unexpected SPARQL response shape") from exc

    async def page_summary(self, site: str, title: str) -> WikiSummary | None:
        """Lead summary of `title` on `site` (e.g. `enwiki`); `None` if there is no page."""
        url = (
            f"{wikipedia_base_url(site)}api/rest_v1/page/summary/"
            f"{quote(title.replace(' ', '_'), safe='')}"
        )
        response = await self._send("GET", url, headers={"Accept": "application/json"})
        if response.status_code == 404:
            return None
        _raise_for_status(response)
        body = _json(response)
        if not isinstance(body, dict):
            raise WikimediaResponseError(f"Unexpected summary response for {url}")
        titles = body.get("titles") if isinstance(body.get("titles"), dict) else {}
        canonical = (titles or {}).get("normalized") or body.get("title") or title
        revision = str(body.get("revision") or "")
        return WikiSummary(
            site=site,
            title=str(canonical),
            revision_id=int(revision) if revision.isdigit() else None,
            extract=str(body.get("extract") or "").strip(),
            page_type=str(body.get("type") or "standard"),
        )

    # --- Transport ---------------------------------------------------------------

    async def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
            sleep=self._sleep,
            before_sleep=lambda state: _log_retry(url, state),
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._request_once(method, url, kwargs)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            raise WikimediaUnavailableError(
                url, exc.last_attempt.attempt_number, _status_of(last)
            ) from last
        raise AssertionError("unreachable")  # pragma: no cover

    async def _request_once(
        self, method: str, url: str, kwargs: Mapping[str, Any]
    ) -> httpx.Response:
        async with self._semaphore:
            response = await self._http.request(method, url, **kwargs)
        status = response.status_code
        if status == 429 or status >= 500:
            raise _RetryableStatus(status, parse_retry_after(response.headers.get("Retry-After")))
        return response

    def _wait(self, state: RetryCallState) -> float:
        backoff = self._backoff(state)
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, _RetryableStatus) and exc.retry_after is not None:
            return max(backoff, min(exc.retry_after, MAX_RETRY_AFTER_SECONDS))
        return backoff


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code >= 400:
        raise WikimediaHTTPError(response.status_code, str(response.request.url))


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise WikimediaResponseError(f"Invalid JSON from {response.request.url}") from exc


def _log_retry(url: str, state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "wikimedia_retry",
        url=url.split("?", 1)[0],
        attempt=state.attempt_number,
        status=_status_of(exc),
        error=type(exc).__name__ if exc else None,
    )
