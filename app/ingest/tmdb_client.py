"""Typed async TMDB v3 client (§5.5, NFR-17, NFR-22).

- Auth: v4 read token (`Authorization: Bearer`) if set, else v3 `api_key` query param.
- Every request has a timeout and is retried with exponential backoff + jitter
  on network errors, 5xx and 429 (honoring `Retry-After`). When attempts run
  out, `TMDBUnavailableError` is raised.
- A semaphore caps in-flight requests (`TMDB_MAX_CONCURRENCY`). It is held per
  attempt, not during backoff sleeps.
- Credentials are never logged; logs carry only the request path.

Usage:
    async with TMDBClient.from_settings(get_settings()) as tmdb:
        details = await tmdb.movie_details(27205)
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models.enums import TitleType
from app.ingest.tmdb_models import (
    MediaListItem,
    MovieDetails,
    Page,
    Provider,
    ProviderList,
    TVDetails,
    WatchProviders,
)

log = get_logger(__name__)

TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"

MOVIE_APPEND: tuple[str, ...] = (
    "keywords",
    "credits",
    "watch/providers",
    "release_dates",
    "translations",
)
TV_APPEND: tuple[str, ...] = (
    "keywords",
    "credits",
    "watch/providers",
    "content_ratings",
    "translations",
)

# Longest `Retry-After` we will wait; longer values are clamped.
MAX_RETRY_AFTER_SECONDS = 60.0

TrendingType = TitleType | Literal["all"]
TrendingWindow = Literal["day", "week"]
SleepFn = Callable[[float], Awaitable[None]]
M = TypeVar("M", bound=BaseModel)


# --- Errors --------------------------------------------------------------------


class TMDBError(Exception):
    """Base class for all TMDB client errors."""


class TMDBConfigError(TMDBError):
    """Client misconfigured (e.g. no credentials)."""


class TMDBHTTPError(TMDBError):
    """Non-retryable HTTP error response (4xx other than 429)."""

    def __init__(self, status_code: int, path: str, message: str | None = None) -> None:
        self.status_code = status_code
        self.path = path
        super().__init__(f"TMDB {status_code} on {path}" + (f": {message}" if message else ""))


class TMDBAuthError(TMDBHTTPError):
    """401: invalid or missing credentials."""


class TMDBNotFoundError(TMDBHTTPError):
    """404: resource does not exist."""


class TMDBUnavailableError(TMDBError):
    """Retries exhausted on 5xx / 429 / network errors."""

    def __init__(self, path: str, attempts: int, last_status: int | None) -> None:
        self.path = path
        self.attempts = attempts
        self.last_status = last_status  # None if the last failure was a network error
        cause = f"HTTP {last_status}" if last_status is not None else "network error"
        super().__init__(f"TMDB unavailable on {path} after {attempts} attempts ({cause})")


class TMDBResponseError(TMDBError):
    """Response body is not valid JSON or does not match the expected model."""


class _RetryableStatus(Exception):  # noqa: N818 - internal control-flow signal
    def __init__(self, status_code: int, retry_after: float | None) -> None:
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(f"retryable HTTP {status_code}")


# --- Helpers -------------------------------------------------------------------


def poster_url(path: str | None, size: str = "w342") -> str | None:
    """Full image URL for a TMDB poster path (`/abc.jpg`), or `None`."""
    return f"{TMDB_IMAGE_BASE_URL}/{size}{path}" if path else None


def logo_url(path: str | None, size: str = "w92") -> str | None:
    """Full image URL for a TMDB provider logo path, or `None`."""
    return f"{TMDB_IMAGE_BASE_URL}/{size}{path}" if path else None


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a `Retry-After` header (delta-seconds or HTTP date)."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def _status_of(exc: BaseException | None) -> int | None:
    return exc.status_code if isinstance(exc, _RetryableStatus) else None


# --- Client --------------------------------------------------------------------


class TMDBClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        read_token: str | None = None,
        base_url: str = TMDB_API_BASE_URL,
        timeout: float = 10.0,
        max_attempts: int = 4,
        max_concurrency: int = 20,
        backoff_initial: float = 0.5,
        backoff_max: float = 8.0,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        if not api_key and not read_token:
            raise TMDBConfigError("Set TMDB_READ_TOKEN or TMDB_API_KEY")
        if max_attempts < 1:
            raise TMDBConfigError("max_attempts must be >= 1")
        headers = {"Accept": "application/json"}
        self._auth_params: dict[str, str] = {}
        if read_token:
            headers["Authorization"] = f"Bearer {read_token}"
        else:
            assert api_key  # noqa: S101 - narrowed by the check above
            self._auth_params["api_key"] = api_key
        self._http = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )
        self._max_attempts = max_attempts
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Wait = backoff_initial * 2**(attempt-1) + U(0, backoff_initial), capped at backoff_max.
        self._backoff = wait_exponential_jitter(
            multiplier=backoff_initial, jitter=backoff_initial, max=backoff_max
        )
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> TMDBClient:
        kwargs: dict[str, Any] = {
            "api_key": settings.tmdb_api_key.get_secret_value() if settings.tmdb_api_key else None,
            "read_token": (
                settings.tmdb_read_token.get_secret_value() if settings.tmdb_read_token else None
            ),
            "timeout": settings.tmdb_timeout_seconds,
            "max_attempts": settings.tmdb_max_attempts,
            "max_concurrency": settings.tmdb_max_concurrency,
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

    async def search_multi(
        self,
        query: str,
        language: str = "en-US",
        *,
        page: int = 1,
        include_adult: bool = False,
    ) -> Page:
        """`/search/multi`: movies, TV and people (callers filter out `person`)."""
        params = {
            "query": query,
            "language": language,
            "page": page,
            "include_adult": include_adult,
        }
        return await self._get_model("/search/multi", Page, params)

    async def movie_details(
        self,
        tmdb_id: int,
        *,
        language: str = "en-US",
        append: tuple[str, ...] = MOVIE_APPEND,
    ) -> MovieDetails:
        params = {"language": language, **_append_param(append)}
        return await self._get_model(f"/movie/{tmdb_id}", MovieDetails, params)

    async def tv_details(
        self,
        tmdb_id: int,
        *,
        language: str = "en-US",
        append: tuple[str, ...] = TV_APPEND,
    ) -> TVDetails:
        params = {"language": language, **_append_param(append)}
        return await self._get_model(f"/tv/{tmdb_id}", TVDetails, params)

    async def details(
        self, type_: TitleType, tmdb_id: int, *, language: str = "en-US"
    ) -> MovieDetails | TVDetails:
        """Movie or TV details with the default appended sub-resources."""
        if type_ is TitleType.MOVIE:
            return await self.movie_details(tmdb_id, language=language)
        return await self.tv_details(tmdb_id, language=language)

    async def discover(
        self,
        type_: TitleType,
        *,
        original_language: str | None = None,
        sort_by: str = "popularity.desc",
        date_gte: date | None = None,
        date_lte: date | None = None,
        page: int = 1,
        include_adult: bool = False,
        language: str = "en-US",
    ) -> Page:
        """`/discover/{movie|tv}`. Dates filter on primary release / first air date.

        Returned items get `media_type` set to `type_`.
        """
        date_field = "primary_release_date" if type_ is TitleType.MOVIE else "first_air_date"
        params: dict[str, Any] = {
            "sort_by": sort_by,
            "page": page,
            "include_adult": include_adult,
            "language": language,
        }
        if original_language:
            params["with_original_language"] = original_language
        if date_gte:
            params[f"{date_field}.gte"] = date_gte.isoformat()
        if date_lte:
            params[f"{date_field}.lte"] = date_lte.isoformat()
        result = await self._get_model(f"/discover/{type_.value}", Page, params)
        return _with_media_type(result, type_)

    async def watch_providers(self, type_: TitleType, tmdb_id: int) -> WatchProviders:
        """`/{movie|tv}/{id}/watch/providers`: offers for every country."""
        return await self._get_model(f"/{type_.value}/{tmdb_id}/watch/providers", WatchProviders)

    async def provider_list(self, type_: TitleType, region: str) -> list[Provider]:
        """`/watch/providers/{movie|tv}?watch_region=`: providers available in a country."""
        params = {"watch_region": region.upper()}
        result = await self._get_model(f"/watch/providers/{type_.value}", ProviderList, params)
        return result.results

    async def trending(self, type_: TrendingType, window: TrendingWindow = "week") -> Page:
        """`/trending/{movie|tv|all}/{day|week}`."""
        kind = type_.value if isinstance(type_, TitleType) else type_
        result = await self._get_model(f"/trending/{kind}/{window}", Page)
        if isinstance(type_, TitleType):
            result = _with_media_type(result, type_)
        return result

    # --- Transport ---------------------------------------------------------------

    async def _get_model(
        self, path: str, model: type[M], params: Mapping[str, Any] | None = None
    ) -> M:
        data = await self._get_json(path, params)
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise TMDBResponseError(f"Unexpected TMDB response for {path}: {exc}") from exc

    async def _get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        query = {**_encode(params or {}), **self._auth_params}
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._max_attempts),
            wait=self._wait,
            retry=retry_if_exception_type((_RetryableStatus, httpx.TransportError)),
            sleep=self._sleep,
            before_sleep=lambda state: _log_retry(path, state),
        )
        try:
            async for attempt in retrying:
                with attempt:
                    return await self._request_once(path, query)
        except RetryError as exc:
            last = exc.last_attempt.exception()
            raise TMDBUnavailableError(
                path, exc.last_attempt.attempt_number, _status_of(last)
            ) from last
        raise AssertionError("unreachable")  # pragma: no cover

    async def _request_once(self, path: str, query: Mapping[str, str]) -> Any:
        async with self._semaphore:
            response = await self._http.get(path, params=query)
        status = response.status_code
        if status == 429 or status >= 500:
            raise _RetryableStatus(status, parse_retry_after(response.headers.get("Retry-After")))
        if status >= 400:
            message = _status_message(response)
            if status == 401:
                raise TMDBAuthError(status, path, message)
            if status == 404:
                raise TMDBNotFoundError(status, path, message)
            raise TMDBHTTPError(status, path, message)
        try:
            return response.json()
        except ValueError as exc:
            raise TMDBResponseError(f"Invalid JSON from TMDB for {path}") from exc

    def _wait(self, state: RetryCallState) -> float:
        backoff = self._backoff(state)
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, _RetryableStatus) and exc.retry_after is not None:
            return max(backoff, min(exc.retry_after, MAX_RETRY_AFTER_SECONDS))
        return backoff


def _append_param(append: tuple[str, ...]) -> dict[str, str]:
    return {"append_to_response": ",".join(append)} if append else {}


def _encode(params: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        out[key] = ("true" if value else "false") if isinstance(value, bool) else str(value)
    return out


def _with_media_type(page: Page, type_: TitleType) -> Page:
    items: list[MediaListItem] = [
        item if item.media_type else item.model_copy(update={"media_type": type_.value})
        for item in page.results
    ]
    return page.model_copy(update={"results": items})


def _status_message(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    message = body.get("status_message") if isinstance(body, dict) else None
    return message if isinstance(message, str) else None


def _log_retry(path: str, state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    log.warning(
        "tmdb_retry",
        path=path,
        attempt=state.attempt_number,
        status=_status_of(exc),
        error=type(exc).__name__ if exc else None,
        wait_s=round(state.upcoming_sleep, 3),
    )
