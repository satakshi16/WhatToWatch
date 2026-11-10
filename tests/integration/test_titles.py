"""P1-08: `GET /titles/search` (TMDB `search/multi`, mocked with recorded fixtures)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.api.deps import get_tmdb_factory
from app.core.config import Settings
from app.db.models import TitleType
from app.ingest.tmdb_client import TMDB_API_BASE_URL, TMDBClient, TMDBConfigError
from app.main import create_app
from tests.catalog_factory import tmdb_title

PROBLEM = "application/problem+json"
PASSWORD = "s3cret-passw0rd"  # pragma: allowlist secret
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tmdb"
URL = "/api/v1/titles/search"

PARASITE_2019 = 496243
SQUID_GAME = 93405
SQUID_AND_WHALE = 10707
SQUID_CHALLENGE = 204082


def _fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return data


# (query, language) -> recorded response body
PAGES: dict[tuple[str, str], dict[str, Any]] = {
    ("parasite", "en-US"): _fixture("search_multi_parasite.json"),
    ("squid", "en-US"): _fixture("search_multi_squid.json"),
    ("squid", "ko"): _fixture("search_multi_squid_ko.json"),
    ("inception", "en-US"): _fixture("search_multi_inception.json"),
    ("nolan", "en-US"): _fixture("search_multi_nolan.json"),
}


class FakeTMDB:
    """respx route for `/search/multi`; records calls; `fail` languages answer 503."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.calls: list[dict[str, str]] = []
        self.fail: set[str] = set()
        self.extra: dict[tuple[str, str], dict[str, Any]] = {}
        router.get(f"{TMDB_API_BASE_URL}/search/multi").mock(side_effect=self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.calls.append(params)
        key = (params["query"], params["language"])
        if params["language"] in self.fail:
            return httpx.Response(503, json={"status_message": "down"})
        body = self.extra.get(key) or PAGES.get(key) or {"page": 1, "results": []}
        return httpx.Response(200, json=body)


async def _no_sleep(_: float) -> None:
    return None


def _use_fake_tmdb(app: FastAPI) -> None:
    app.dependency_overrides[get_tmdb_factory] = lambda: (
        lambda: TMDBClient(api_key="test-key", max_attempts=2, sleep=_no_sleep)
    )


@pytest.fixture
def tmdb(app: FastAPI) -> Iterator[FakeTMDB]:
    _use_fake_tmdb(app)
    with respx.mock(assert_all_called=False) as router:
        yield FakeTMDB(router)


@pytest.fixture
def api(migrated_engine: Engine, client: TestClient, tmdb: FakeTMDB) -> TestClient:
    return client


def _token(client: TestClient, email: str = "marcus@example.com") -> str:
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "display_name": "Marcus",
            "country": "US",
            "age_confirmed": True,
        },
    )
    assert resp.status_code == 201
    token: str = resp.json()["access_token"]
    return token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def token(api: TestClient) -> str:
    return _token(api)


def _search(client: TestClient, token: str, **params: Any) -> httpx.Response:
    return client.get(URL, params=params, headers=_auth(token))


def _set_prefs(client: TestClient, token: str, **body: Any) -> None:
    resp = client.put("/api/v1/me/preferences", headers=_auth(token), json=body)
    assert resp.status_code == 200, resp.text


def _ids(resp: httpx.Response) -> list[tuple[str, int]]:
    return [(r["type"], r["tmdb_id"]) for r in resp.json()["results"]]


# --- Acceptance ---


def test_parasite_returns_the_2019_film(api: TestClient, token: str) -> None:
    resp = _search(api, token, q="parasite")
    assert resp.status_code == 200
    first = resp.json()["results"][0]
    assert first == {
        "tmdb_id": PARASITE_2019,
        "type": "movie",
        "title": "Parasite",
        "original_title": "기생충",
        "year": 2019,
        "poster_url": first["poster_url"],
        "in_catalog": False,
    }
    assert first["poster_url"].startswith("https://image.tmdb.org/t/p/w342/")


def test_squid_returns_the_tv_series(api: TestClient, token: str) -> None:
    resp = _search(api, token, q="squid")
    assert resp.status_code == 200
    first = resp.json()["results"][0]
    assert (first["type"], first["tmdb_id"], first["title"]) == ("tv", SQUID_GAME, "Squid Game")
    assert first["original_title"] == "오징어 게임"
    assert first["year"] == 2021


def test_results_include_titles_not_in_catalog_and_flag_catalog_ones(
    api: TestClient, token: str, db_session: Session
) -> None:
    db_session.add(tmdb_title(PARASITE_2019, TitleType.MOVIE, language="ko"))
    # Same TMDB ID as the film but a TV row: must not mark the film as in catalog.
    db_session.add(tmdb_title(48311, TitleType.TV))
    db_session.commit()
    results = _search(api, token, q="parasite").json()["results"]
    flags = {(r["type"], r["tmdb_id"]): r["in_catalog"] for r in results}
    assert flags[("movie", PARASITE_2019)] is True
    assert flags[("movie", 48311)] is False
    assert sum(flags.values()) == 1
    assert len(results) == 10


# --- Shape, filtering, limits ---


def test_people_are_filtered_out(api: TestClient, token: str) -> None:
    raw = PAGES[("squid", "en-US")]["results"]
    assert any(r["media_type"] == "person" for r in raw)
    resp = _search(api, token, q="squid", limit=20)
    types = {r["type"] for r in resp.json()["results"]}
    assert types <= {"movie", "tv"}
    assert len(resp.json()["results"]) == sum(r["media_type"] != "person" for r in raw)


def test_people_only_search_returns_empty_list(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    tmdb.extra[("nobody", "en-US")] = {
        "page": 1,
        "results": [{"id": 1, "media_type": "person", "name": "Nobody"}],
    }
    resp = _search(api, token, q="nobody")
    assert resp.status_code == 200
    assert resp.json()["results"] == []


@pytest.mark.parametrize("type_", ["movie", "tv"])
def test_type_filter(api: TestClient, token: str, type_: str) -> None:
    resp = _search(api, token, q="squid", type=type_, limit=20)
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == type_
    assert body["results"]
    assert {r["type"] for r in body["results"]} == {type_}


def test_default_limit_is_10_and_limit_is_respected(api: TestClient, token: str) -> None:
    assert len(_search(api, token, q="parasite").json()["results"]) == 10
    assert len(_search(api, token, q="parasite", limit=3).json()["results"]) == 3
    assert len(_search(api, token, q="parasite", limit=20).json()["results"]) == 20


def test_keeps_tmdb_order_and_handles_missing_dates_and_posters(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    tmdb.extra[("odd", "en-US")] = {
        "page": 1,
        "results": [
            {"id": 2, "media_type": "tv", "name": "B", "first_air_date": ""},
            {"id": 1, "media_type": "movie", "title": "A", "poster_path": None},
        ],
    }
    results = _search(api, token, q="odd").json()["results"]
    assert [(r["type"], r["tmdb_id"]) for r in results] == [("tv", 2), ("movie", 1)]
    assert results[0]["year"] is None
    assert results[0]["original_title"] == "B"  # falls back to the display title
    assert results[1]["poster_url"] is None


def test_response_envelope(api: TestClient, token: str) -> None:
    body = _search(api, token, q="  Inception ").json()
    assert body["query"] == "inception"
    assert body["type"] == "all"
    assert body["languages"] == ["en-US"]
    assert body["results"][0]["tmdb_id"] == 27205


# --- Validation & auth ---


@pytest.mark.parametrize(
    "params",
    [
        {"q": "a"},
        {"q": "  a  "},
        {"q": "x" * 101},
        {},
        {"q": "squid", "type": "person"},
        {"q": "squid", "limit": 0},
        {"q": "squid", "limit": 21},
    ],
)
def test_invalid_params_are_422(
    api: TestClient, token: str, tmdb: FakeTMDB, params: dict[str, Any]
) -> None:
    resp = api.get(URL, params=params, headers=_auth(token))
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM
    assert resp.json()["errors"]
    assert tmdb.calls == []


def test_two_characters_is_enough(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    assert _search(api, token, q="ab").status_code == 200
    assert tmdb.calls[0]["query"] == "ab"


def test_requires_auth(api: TestClient, tmdb: FakeTMDB) -> None:
    resp = api.get(URL, params={"q": "squid"})
    assert resp.status_code == 401
    assert resp.headers["content-type"] == PROBLEM
    assert tmdb.calls == []


# --- Language ---


def test_english_user_searches_once_in_english(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    _search(api, token, q="squid")
    assert len(tmdb.calls) == 1
    assert tmdb.calls[0]["language"] == "en-US"
    assert tmdb.calls[0]["include_adult"] == "false"


def test_non_english_user_gets_localized_titles_with_english_fallback(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    _set_prefs(api, token, languages=["ko", "en"])
    resp = _search(api, token, q="squid", limit=20)
    assert resp.status_code == 200
    body = resp.json()
    assert body["languages"] == ["ko", "en-US"]
    assert sorted(c["language"] for c in tmdb.calls) == ["en-US", "ko"]
    by_id = {(r["type"], r["tmdb_id"]): r for r in body["results"]}
    # Found only by the English search (TMDB's Korean search misses it): English title.
    assert by_id[("tv", SQUID_GAME)]["title"] == "Squid Game"
    # Found by both: the Korean record wins.
    assert by_id[("tv", SQUID_CHALLENGE)]["title"] == "오징어 게임: 더 챌린지"
    assert by_id[("movie", SQUID_AND_WHALE)]["title"] == "오징어와 고래"
    # Interleaved (ko, en, ko, en, ...) so English-only hits are not pushed past the limit.
    assert _ids(resp)[:3] == [
        ("tv", SQUID_CHALLENGE),
        ("tv", SQUID_GAME),
        ("movie", SQUID_AND_WHALE),
    ]
    assert len(set(_ids(resp))) == len(_ids(resp))


def test_non_english_search_survives_one_language_failing(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    _set_prefs(api, token, languages=["ko"])
    tmdb.fail = {"ko"}
    resp = _search(api, token, q="squid")
    assert resp.status_code == 200
    assert _ids(resp)[0] == ("tv", SQUID_GAME)


# --- Adult ---


def _adult_page() -> dict[str, Any]:
    return {
        "page": 1,
        "results": [
            {"id": 5, "media_type": "movie", "title": "Grown-up", "adult": True},
            {"id": 6, "media_type": "movie", "title": "Family", "adult": False},
        ],
    }


def test_adult_titles_hidden_unless_allowed(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    tmdb.extra[("grown", "en-US")] = _adult_page()
    assert [r["tmdb_id"] for r in _search(api, token, q="grown").json()["results"]] == [6]
    assert tmdb.calls[-1]["include_adult"] == "false"

    _set_prefs(api, token, allow_adult=True)
    assert [r["tmdb_id"] for r in _search(api, token, q="grown").json()["results"]] == [5, 6]
    assert tmdb.calls[-1]["include_adult"] == "true"


# --- Cache ---


def test_repeat_search_is_served_from_cache(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    _search(api, token, q="squid")
    _search(api, token, q="SQUID", type="tv", limit=3)
    _search(api, token, q="  squid ")
    assert len(tmdb.calls) == 1


def test_cache_is_shared_across_users(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    _search(api, token, q="squid")
    _search(api, _token(api, "other@example.com"), q="squid")
    assert len(tmdb.calls) == 1


def test_english_results_cached_for_non_english_users(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    _search(api, token, q="squid")
    other = _token(api, "ji@example.com")
    _set_prefs(api, other, languages=["ko"])
    _search(api, other, q="squid")
    assert [c["language"] for c in tmdb.calls] == ["en-US", "ko"]


def test_failures_are_not_cached(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    tmdb.fail = {"en-US"}
    assert _search(api, token, q="squid").status_code == 503
    tmdb.fail = set()
    assert _search(api, token, q="squid").status_code == 200


def test_cache_can_be_disabled(migrated_engine: Engine, settings: Settings, tmdb: FakeTMDB) -> None:
    no_cache = create_app(settings.model_copy(update={"title_search_cache_ttl_seconds": 0}))
    _use_fake_tmdb(no_cache)
    with TestClient(no_cache) as c:
        token = _token(c)
        _search(c, token, q="squid")
        _search(c, token, q="squid")
    assert len(tmdb.calls) == 2


# --- TMDB errors ---


def test_tmdb_down_is_503(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    tmdb.fail = {"en-US"}
    resp = _search(api, token, q="squid")
    assert resp.status_code == 503
    assert resp.headers["content-type"] == PROBLEM
    assert resp.headers["retry-after"] == "60"
    assert len(tmdb.calls) == 2  # retried (max_attempts=2)


def test_all_languages_down_is_503(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    _set_prefs(api, token, languages=["ko"])
    tmdb.fail = {"ko", "en-US"}
    assert _search(api, token, q="squid").status_code == 503


def test_missing_tmdb_credentials_is_503(app: FastAPI, api: TestClient, token: str) -> None:
    def factory() -> TMDBClient:
        raise TMDBConfigError("Set TMDB_READ_TOKEN or TMDB_API_KEY")

    app.dependency_overrides[get_tmdb_factory] = lambda: factory
    assert _search(api, token, q="squid").status_code == 503


def test_cached_search_needs_no_tmdb_client(app: FastAPI, api: TestClient, token: str) -> None:
    assert _search(api, token, q="squid").status_code == 200

    def factory() -> TMDBClient:
        raise AssertionError("TMDB client should not be opened")

    app.dependency_overrides[get_tmdb_factory] = lambda: factory
    assert _search(api, token, q="squid").status_code == 200
