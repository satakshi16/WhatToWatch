from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from app.core.config import Settings
from app.core.logging import configure_logging
from app.db.models.enums import OfferType, TitleType
from app.ingest.genres import canonical_genres
from app.ingest.tmdb_client import (
    MAX_RETRY_AFTER_SECONDS,
    MOVIE_APPEND,
    TMDB_API_BASE_URL,
    TV_APPEND,
    TMDBAuthError,
    TMDBClient,
    TMDBConfigError,
    TMDBHTTPError,
    TMDBNotFoundError,
    TMDBResponseError,
    TMDBUnavailableError,
    logo_url,
    parse_retry_after,
    poster_url,
)
from app.ingest.tmdb_models import MovieDetails, TVDetails

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tmdb"
API_KEY = "test-api-key-123"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeps() -> SleepRecorder:
    return SleepRecorder()


@pytest.fixture
async def tmdb(sleeps: SleepRecorder) -> AsyncIterator[TMDBClient]:
    async with TMDBClient(api_key=API_KEY, max_attempts=3, sleep=sleeps) as client:
        yield client


@pytest.fixture
def api() -> respx.MockRouter:
    with respx.mock(base_url=TMDB_API_BASE_URL, assert_all_called=False) as router:
        yield router


def params(route: respx.Route) -> dict[str, str]:
    return dict(route.calls.last.request.url.params)


# --- Endpoints (recorded fixtures, see scripts/record_tmdb_fixtures.py) -------


async def test_search_multi(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/search/multi").respond(json=fixture("search_multi_inception.json"))

    page = await tmdb.search_multi("inception", "hi-IN")

    assert params(route) == {
        "query": "inception",
        "language": "hi-IN",
        "page": "1",
        "include_adult": "false",
        "api_key": API_KEY,
    }
    assert (page.page, page.total_results, len(page.results)) == (1, 13, 13)
    top = page.results[0]
    assert (top.id, top.title_type, top.display_title, top.year) == (
        27205,
        TitleType.MOVIE,
        "Inception",
        2010,
    )
    assert top.genre_ids == [28, 878, 80]
    undated = next(r for r in page.results if r.id == 1747666)
    assert undated.release_date is None  # TMDB sends "" for unknown dates
    assert undated.year is None
    tv = page.results[-1]
    assert (tv.title_type, tv.display_title, tv.year) == (TitleType.TV, "The Cruise", 2021)


async def test_search_multi_mixed_media_types(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    raw = fixture("search_multi_nolan.json")
    api.get("/search/multi").respond(json=raw)

    page = await tmdb.search_multi("nolan")

    expected = {
        mt: sum(r["media_type"] == mt for r in raw["results"]) for mt in ("movie", "tv", "person")
    }
    assert all(expected.values())
    assert sum(r.title_type is TitleType.MOVIE for r in page.results) == expected["movie"]
    assert sum(r.title_type is TitleType.TV for r in page.results) == expected["tv"]
    people = [r for r in page.results if r.media_type == "person"]
    assert len(people) == expected["person"]
    assert all(p.title_type is None for p in people)
    nolan = page.results[0]
    assert (nolan.id, nolan.display_title, nolan.title_type) == (525, "Christopher Nolan", None)
    tv = next(r for r in page.results if r.id == 35109)
    assert (tv.display_title, tv.first_air_date) == ("Nolan Knows Best", None)


async def test_movie_details_with_appends(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    raw = fixture("movie_27205.json")
    route = api.get("/movie/27205").respond(json=raw)

    movie = await tmdb.movie_details(27205)

    assert params(route)["append_to_response"] == ",".join(MOVIE_APPEND)
    assert params(route)["language"] == "en-US"
    assert movie.title == movie.original_title == "Inception"
    assert (movie.original_language, movie.adult) == ("en", False)
    assert movie.release_date == date(2010, 7, 15)
    assert movie.runtime == 148
    assert movie.genre_ids == [28, 878, 80]
    assert movie.vote_count > 0
    assert movie.poster_path
    assert movie.keywords is not None
    assert {"dreams", "heist"} <= {k.name for k in movie.keywords.keywords}
    assert movie.credits is not None
    assert [c.name for c in movie.credits.cast[:3]] == [
        "Leonardo DiCaprio",
        "Joseph Gordon-Levitt",
        "Ken Watanabe",
    ]
    directors = [c.name for c in movie.credits.crew if c.job == "Director"]
    assert directors == ["Christopher Nolan"]
    # Availability for every country in one call (D-004).
    assert movie.watch_providers is not None
    us = movie.watch_providers.results["US"]
    assert us.link == "https://www.themoviedb.org/movie/27205-inception/watch?locale=US"
    assert us.flatrate == []
    raw_us = raw["watch/providers"]["results"]["US"]
    assert [(t, p.provider_id) for t, p in us.offers()] == [
        (OfferType.RENT, p["provider_id"]) for p in raw_us["rent"]
    ] + [(OfferType.BUY, p["provider_id"]) for p in raw_us["buy"]]
    assert {t for t, _ in movie.watch_providers.results["IN"].offers()} == {
        OfferType.FLATRATE,
        OfferType.RENT,
        OfferType.BUY,
    }
    assert movie.release_dates is not None
    us_rd = next(r for r in movie.release_dates.results if r.iso_3166_1 == "US")
    assert {d.certification for d in us_rd.release_dates if d.type == 3} == {"PG-13"}
    assert movie.translations is not None
    by_lang = {t.iso_639_1: t.data for t in movie.translations.translations}
    assert by_lang["hi"].title == "चक्रव्युह"
    assert by_lang["ko"].title == "인셉션"
    assert by_lang["en"].title is None  # "" = no translated title


async def test_tv_details_with_appends(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/tv/1429").respond(json=fixture("tv_1429.json"))

    tv = await tmdb.tv_details(1429, language="ja-JP")

    assert params(route)["append_to_response"] == ",".join(TV_APPEND)
    assert params(route)["language"] == "ja-JP"
    assert (tv.name, tv.original_name, tv.original_language) == (
        "Attack on Titan",
        "進撃の巨人",
        "ja",
    )
    assert tv.first_air_date == date(2013, 4, 7)
    assert tv.genre_ids == [16, 10765, 10759]
    assert tv.number_of_seasons == 4
    # TMDB often leaves `episode_run_time` empty; the last episode has a runtime.
    assert tv.episode_run_time == []
    assert tv.last_episode_to_air is not None
    assert tv.last_episode_to_air.runtime == 23
    assert tv.created_by == []
    assert tv.keywords is not None  # TV keywords come back under `results`
    assert {"anime", "dystopia"} <= {k.name for k in tv.keywords.keywords}
    assert tv.credits is not None
    assert "Yuichiro Hayashi" in [c.name for c in tv.credits.crew if c.job == "Series Director"]
    assert tv.content_ratings is not None
    assert {r.iso_3166_1: r.rating for r in tv.content_ratings.results}["US"] == "TV-MA"
    assert tv.translations is not None
    by_lang = {t.iso_639_1: t.data for t in tv.translations.translations}
    assert by_lang["ko"].title == "진격의 거인"  # TV translations use `name`
    assert by_lang["ja"].title is None
    assert tv.watch_providers is not None
    assert [p.provider_name for p in tv.watch_providers.results["US"].flatrate] == [
        "Hulu",
        "Crunchyroll",
        "Crunchyroll Amazon Channel",
    ]
    assert canonical_genres(tv.genre_ids, tv.original_language) == [
        "animation",
        "anime",
        "science-fiction",
        "fantasy",
        "action",
        "adventure",
    ]


async def test_details_dispatches_on_type(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    api.get("/movie/27205").respond(json=fixture("movie_27205.json"))
    api.get("/tv/1429").respond(json=fixture("tv_1429.json"))

    assert isinstance(await tmdb.details(TitleType.MOVIE, 27205), MovieDetails)
    assert isinstance(await tmdb.details(TitleType.TV, 1429), TVDetails)


async def test_details_without_appends(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    body = {k: v for k, v in fixture("movie_27205.json").items() if k not in MOVIE_APPEND}
    route = api.get("/movie/27205").respond(json=body)

    movie = await tmdb.movie_details(27205, append=())

    assert "append_to_response" not in params(route)
    assert movie.watch_providers is None
    assert movie.keywords is None


async def test_discover_movie(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/discover/movie").respond(json=fixture("discover_movie_hi.json"))

    page = await tmdb.discover(
        TitleType.MOVIE,
        original_language="hi",
        sort_by="vote_count.desc",
        date_gte=date(2025, 11, 1),
        date_lte=date(2026, 11, 1),
        page=3,
    )

    assert params(route) == {
        "sort_by": "vote_count.desc",
        "page": "3",
        "include_adult": "false",
        "language": "en-US",
        "with_original_language": "hi",
        "primary_release_date.gte": "2025-11-01",
        "primary_release_date.lte": "2026-11-01",
        "api_key": API_KEY,
    }
    assert (page.total_pages, len(page.results)) == (532, 20)
    assert all(r.title_type is TitleType.MOVIE for r in page.results)  # filled in by client
    assert all(r.original_language == "hi" for r in page.results)


async def test_discover_tv(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/discover/tv").respond(json=fixture("discover_tv_ko.json"))

    page = await tmdb.discover(TitleType.TV, original_language="ko", date_gte=date(2024, 1, 1))

    p = params(route)
    assert p["first_air_date.gte"] == "2024-01-01"
    assert "first_air_date.lte" not in p
    assert all(r.title_type is TitleType.TV for r in page.results)
    assert (page.results[0].display_title, page.results[0].year) == ("Squid Game", 2021)


async def test_discover_without_language(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/discover/movie").respond(json=fixture("discover_movie_hi.json"))

    await tmdb.discover(TitleType.MOVIE)

    assert "with_original_language" not in params(route)


async def test_watch_providers(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    api.get("/movie/27205/watch/providers").respond(
        json=fixture("watch_providers_movie_27205.json")
    )

    wp = await tmdb.watch_providers(TitleType.MOVIE, 27205)

    assert set(wp.results) == {"US", "IN", "JP", "KR"}
    assert wp.results["US"].flatrate == []
    assert [p.provider_name for p in wp.results["KR"].flatrate] == ["wavve", "Watcha"]
    assert "Amazon Prime Video" in [p.provider_name for p in wp.results["IN"].flatrate]


async def test_provider_list(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/watch/providers/movie").respond(json=fixture("provider_list_movie_US.json"))

    providers = await tmdb.provider_list(TitleType.MOVIE, "us")

    assert params(route)["watch_region"] == "US"
    assert len(providers) == 298
    names = {p.provider_name for p in providers}
    assert {"Netflix", "Amazon Prime Video", "HBO Max", "Disney Plus", "Hulu"} <= names
    netflix = next(p for p in providers if p.provider_id == 8)
    assert netflix.display_priorities["US"] == 2
    assert logo_url(netflix.logo_path) == (
        "https://image.tmdb.org/t/p/w92/rK1KljqmbvO9HQa1PBFLILWah72.png"
    )


async def test_trending_all(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    raw = fixture("trending_all_week.json")
    api.get("/trending/all/week").respond(json=raw)

    page = await tmdb.trending("all")

    assert len(page.results) == 20
    assert [r.title_type for r in page.results] == [
        TitleType(r["media_type"]) for r in raw["results"]
    ]
    assert page.results[0].display_title == "Resident Evil"


async def test_trending_tv_day(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    api.get("/trending/tv/day").respond(json=fixture("trending_tv_day.json"))

    page = await tmdb.trending(TitleType.TV, "day")

    assert len(page.results) == 20
    assert all(r.title_type is TitleType.TV for r in page.results)


async def test_typed_trending_fills_missing_media_type(
    tmdb: TMDBClient, api: respx.MockRouter
) -> None:
    api.get("/trending/tv/day").respond(json={"page": 1, "results": [{"id": 5, "name": "Y"}]})

    page = await tmdb.trending(TitleType.TV, "day")

    assert page.results[0].title_type is TitleType.TV


# --- Auth ----------------------------------------------------------------------


async def test_read_token_sent_as_bearer(api: respx.MockRouter) -> None:
    route = api.get("/trending/all/week").respond(json={"results": []})

    async with TMDBClient(read_token="v4-token", api_key=API_KEY) as client:
        await client.trending("all")

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer v4-token"
    assert "api_key" not in request.url.params


def test_missing_credentials_rejected() -> None:
    with pytest.raises(TMDBConfigError):
        TMDBClient()


async def test_from_settings(api: respx.MockRouter) -> None:
    route = api.get("/trending/all/week").respond(json={"results": []})
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        tmdb_api_key=API_KEY,
        tmdb_max_attempts=2,
    )

    async with TMDBClient.from_settings(settings) as client:
        await client.trending("all")

    assert params(route)["api_key"] == API_KEY


# --- Errors, retries, rate limiting --------------------------------------------


async def test_retries_5xx_then_succeeds(
    tmdb: TMDBClient, api: respx.MockRouter, sleeps: SleepRecorder
) -> None:
    route = api.get("/trending/all/week").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(502),
            httpx.Response(200, json=fixture("trending_all_week.json")),
        ]
    )

    page = await tmdb.trending("all")

    assert len(page.results) == 20
    assert route.call_count == 3
    assert len(sleeps.calls) == 2
    assert all(0 <= s <= 8.0 for s in sleeps.calls)


async def test_429_honors_retry_after(
    tmdb: TMDBClient, api: respx.MockRouter, sleeps: SleepRecorder
) -> None:
    api.get("/trending/all/week").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(429, headers={"Retry-After": "3600"}),
            httpx.Response(200, json={"results": []}),
        ]
    )

    await tmdb.trending("all")

    assert sleeps.calls[0] >= 7
    assert sleeps.calls[1] == MAX_RETRY_AFTER_SECONDS  # clamped


async def test_gives_up_after_max_attempts(
    tmdb: TMDBClient, api: respx.MockRouter, sleeps: SleepRecorder
) -> None:
    route = api.get("/movie/1").respond(status_code=500)

    with pytest.raises(TMDBUnavailableError) as exc_info:
        await tmdb.movie_details(1)

    err = exc_info.value
    assert (err.attempts, err.last_status, err.path) == (3, 500, "/movie/1")
    assert route.call_count == 3
    assert len(sleeps.calls) == 2


async def test_gives_up_on_persistent_429(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    api.get("/movie/1").respond(status_code=429)

    with pytest.raises(TMDBUnavailableError) as exc_info:
        await tmdb.movie_details(1)

    assert exc_info.value.last_status == 429


async def test_network_errors_retried(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    route = api.get("/movie/1").mock(side_effect=httpx.ConnectTimeout("timed out"))

    with pytest.raises(TMDBUnavailableError) as exc_info:
        await tmdb.movie_details(1)

    assert exc_info.value.last_status is None
    assert isinstance(exc_info.value.__cause__, httpx.ConnectTimeout)
    assert route.call_count == 3


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (404, fixture("error_404.json"), TMDBNotFoundError),
        (401, fixture("error_401.json"), TMDBAuthError),
        (400, {"status_code": 22, "status_message": "Invalid page."}, TMDBHTTPError),
    ],
)
async def test_client_errors_not_retried(
    tmdb: TMDBClient,
    api: respx.MockRouter,
    sleeps: SleepRecorder,
    status: int,
    body: dict[str, Any],
    error: type[TMDBHTTPError],
) -> None:
    route = api.get("/movie/1").respond(status_code=status, json=body)

    with pytest.raises(error) as exc_info:
        await tmdb.movie_details(1)

    assert exc_info.value.status_code == status
    assert body["status_message"] in str(exc_info.value)
    assert route.call_count == 1
    assert sleeps.calls == []


async def test_invalid_json_is_response_error(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    api.get("/movie/1").respond(text="<html>oops</html>")

    with pytest.raises(TMDBResponseError):
        await tmdb.movie_details(1)


async def test_schema_mismatch_is_response_error(tmdb: TMDBClient, api: respx.MockRouter) -> None:
    api.get("/movie/1").respond(json={"id": "not-an-int"})

    with pytest.raises(TMDBResponseError):
        await tmdb.movie_details(1)


async def test_concurrency_limited_by_semaphore(api: respx.MockRouter) -> None:
    in_flight = 0
    peak = 0

    async def slow(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, json={"results": []})

    api.get("/trending/all/week").mock(side_effect=slow)

    async with TMDBClient(api_key=API_KEY, max_concurrency=2) as client:
        await asyncio.gather(*(client.trending("all") for _ in range(6)))

    assert peak == 2


async def test_api_key_never_logged(
    tmdb: TMDBClient, api: respx.MockRouter, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("DEBUG")
    api.get("/movie/1").mock(side_effect=[httpx.Response(503), httpx.Response(200, json={})])

    with pytest.raises(TMDBResponseError):
        await tmdb.movie_details(1)

    out = capsys.readouterr().out
    assert "tmdb_retry" in out
    assert API_KEY not in out


# --- Helpers -------------------------------------------------------------------


def test_image_url_helpers() -> None:
    assert poster_url("/abc.jpg") == "https://image.tmdb.org/t/p/w342/abc.jpg"
    assert poster_url("/abc.jpg", "w500") == "https://image.tmdb.org/t/p/w500/abc.jpg"
    assert poster_url(None) is None
    assert logo_url("") is None


def test_parse_retry_after() -> None:
    now = datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("5") == 5.0
    assert parse_retry_after("-1") == 0.0
    assert parse_retry_after("Tue, 03 Nov 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Tue, 03 Nov 2026 11:00:00 GMT", now=now) == 0.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None


def test_recorded_fixtures_contain_no_credentials() -> None:
    files = sorted(FIXTURES.glob("*.json"))
    assert len(files) >= 12
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "api_key=" not in text, path.name
        assert "Bearer " not in text, path.name
