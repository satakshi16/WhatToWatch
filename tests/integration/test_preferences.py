"""P1-07: `GET /providers`, `GET/PUT /me/preferences`, platform cleanup on `PATCH /me`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from app.api.deps import get_tmdb_factory
from app.db.base import utcnow
from app.db.models import OnboardingStep, Provider, User, UserPlatform, UserPreferences
from app.ingest.tmdb_client import TMDB_API_BASE_URL, TMDBClient

PROBLEM = "application/problem+json"
PASSWORD = "s3cret-passw0rd"  # pragma: allowlist secret
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tmdb"

NETFLIX, PRIME, DISNEY, HULU, HBO_MAX = 8, 9, 337, 15, 1899
TV_ONLY_US = 9999  # synthetic provider that only appears in the US TV list


def _provider(pid: int, name: str, priorities: dict[str, int]) -> dict[str, Any]:
    return {
        "provider_id": pid,
        "provider_name": name,
        "logo_path": f"/logo{pid}.png",
        "display_priority": 99,
        "display_priorities": priorities,
    }


# Recorded US movie list (298 providers); TV lists and KR are synthetic.
US_MOVIE = json.loads((FIXTURES / "provider_list_movie_US.json").read_text())
US_TV = {
    "results": [
        _provider(NETFLIX, "Netflix", {"US": 1}),
        _provider(HULU, "Hulu", {"US": 2}),
        _provider(TV_ONLY_US, "TV Only", {"US": 3}),
    ]
}
KR_MOVIE = {"results": [_provider(NETFLIX, "Netflix", {"KR": 1}), _provider(97, "Watcha", {})]}
KR_TV = {"results": [_provider(NETFLIX, "Netflix", {"KR": 1})]}
LISTS = {
    ("movie", "US"): US_MOVIE,
    ("tv", "US"): US_TV,
    ("movie", "KR"): KR_MOVIE,
    ("tv", "KR"): KR_TV,
}


class FakeTMDB:
    """respx router serving provider lists; `fail=True` makes every call return 503."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.fail = False
        self.calls: list[tuple[str, str]] = []
        router.get(url__regex=rf"^{TMDB_API_BASE_URL}/watch/providers/(movie|tv)").mock(
            side_effect=self._handle
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        kind = request.url.path.rsplit("/", 1)[-1]
        region = request.url.params["watch_region"]
        self.calls.append((kind, region))
        if self.fail:
            return httpx.Response(503, json={"status_message": "down"})
        return httpx.Response(200, json=LISTS.get((kind, region), {"results": []}))


@pytest.fixture
def tmdb(app: FastAPI) -> Iterator[FakeTMDB]:
    async def no_sleep(_: float) -> None:
        return None

    app.dependency_overrides[get_tmdb_factory] = lambda: (
        lambda: TMDBClient(api_key="test-key", max_attempts=2, sleep=no_sleep)
    )
    with respx.mock(assert_all_called=False) as router:
        yield FakeTMDB(router)


@pytest.fixture
def api(migrated_engine: Engine, client: TestClient, tmdb: FakeTMDB) -> TestClient:
    return client


def _token(client: TestClient, email: str = "marcus@example.com", country: str = "US") -> str:
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "display_name": "Marcus",
            "country": country,
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


def _put(client: TestClient, token: str, **body: Any) -> httpx.Response:
    return client.put("/api/v1/me/preferences", headers=_auth(token), json=body)


def _user(engine: Engine, email: str = "marcus@example.com") -> User:
    with Session(engine) as s:
        user = s.scalar(select(User).where(User.email == email))
        assert user is not None
        return user


# --- GET /providers ---


def test_us_provider_list_has_major_platforms_with_logos(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    resp = api.get("/api/v1/providers", params={"country": "US"}, headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["country"] == "US"
    by_name = {p["name"]: p for p in body["providers"]}
    for name in ("Netflix", "Amazon Prime Video", "HBO Max", "Disney Plus", "Hulu"):
        assert by_name[name]["logo_url"].startswith("https://image.tmdb.org/t/p/w92/")
    assert by_name["Netflix"]["provider_id"] == NETFLIX
    assert sorted(tmdb.calls) == [("movie", "US"), ("tv", "US")]


def test_provider_list_is_union_sorted_by_display_priority(api: TestClient, token: str) -> None:
    providers = api.get("/api/v1/providers", headers=_auth(token)).json()["providers"]
    ids = [p["provider_id"] for p in providers]
    assert len(ids) == len(set(ids)) == len(US_MOVIE["results"]) + 1  # + TV-only provider
    assert TV_ONLY_US in ids
    priorities = [p["display_priority"] for p in providers]
    assert priorities == sorted(priorities)
    # Movie entry wins on overlap: Netflix keeps the movie list's US priority.
    assert providers[ids.index(NETFLIX)]["display_priority"] == 2


def test_provider_list_defaults_to_user_country(api: TestClient, tmdb: FakeTMDB) -> None:
    token = _token(api, email="jiwoo@example.com", country="KR")
    body = api.get("/api/v1/providers", headers=_auth(token)).json()
    assert body["country"] == "KR"
    assert [p["name"] for p in body["providers"]] == ["Netflix", "Watcha"]  # no priority → last
    assert tmdb.calls
    assert {region for _, region in tmdb.calls} == {"KR"}


def test_provider_list_is_cached(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    first = api.get("/api/v1/providers", headers=_auth(token)).json()
    tmdb.calls.clear()
    second = api.get("/api/v1/providers", headers=_auth(token)).json()
    assert tmdb.calls == []
    assert second == first


def test_stale_provider_list_is_refreshed(
    api: TestClient, token: str, tmdb: FakeTMDB, migrated_engine: Engine
) -> None:
    api.get("/api/v1/providers", headers=_auth(token))
    old = utcnow() - timedelta(days=8)
    with Session(migrated_engine) as s:
        s.execute(update(Provider).where(Provider.country == "US").values(fetched_at=old))
        s.commit()
    tmdb.calls.clear()
    body = api.get("/api/v1/providers", headers=_auth(token)).json()
    assert len(tmdb.calls) == 2
    assert body["fetched_at"] > old.isoformat()


def test_stale_list_served_when_tmdb_fails(
    api: TestClient, token: str, tmdb: FakeTMDB, migrated_engine: Engine
) -> None:
    fresh = api.get("/api/v1/providers", headers=_auth(token)).json()
    with Session(migrated_engine) as s:
        s.execute(update(Provider).values(fetched_at=utcnow() - timedelta(days=30)))
        s.commit()
    tmdb.fail = True
    resp = api.get("/api/v1/providers", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["providers"] == fresh["providers"]


def test_no_cache_and_tmdb_down_is_503(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    tmdb.fail = True
    resp = api.get("/api/v1/providers", headers=_auth(token))
    assert resp.status_code == 503
    assert resp.headers["content-type"] == PROBLEM
    assert resp.headers["retry-after"] == "60"


def test_tmdb_not_configured_is_503(api: TestClient, app: FastAPI, token: str) -> None:
    app.dependency_overrides.pop(get_tmdb_factory)  # real factory; test settings have no key
    resp = api.get("/api/v1/providers", headers=_auth(token))
    assert resp.status_code == 503


def test_providers_requires_auth(api: TestClient) -> None:
    assert api.get("/api/v1/providers", params={"country": "US"}).status_code == 401


@pytest.mark.parametrize("country", ["XX", "USA", "1"])
def test_providers_rejects_invalid_country(api: TestClient, token: str, country: str) -> None:
    resp = api.get("/api/v1/providers", params={"country": country}, headers=_auth(token))
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM


def test_lowercase_country_accepted(api: TestClient, token: str) -> None:
    resp = api.get("/api/v1/providers", params={"country": "us"}, headers=_auth(token))
    assert resp.json()["country"] == "US"


# --- GET/PUT /me/preferences ---


def test_defaults_before_first_save(
    api: TestClient, token: str, tmdb: FakeTMDB, migrated_engine: Engine
) -> None:
    resp = api.get("/api/v1/me/preferences", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json() == {
        "languages": ["en"],
        "platforms": [],
        "content_types": ["movie", "tv"],
        "era": "any",
        "avoid_genres": [],
        "max_runtime": None,
        "include_rent_buy": False,
        "allow_adult": False,
        "onboarding_step": "preferences",
        "updated_at": None,
    }
    with Session(migrated_engine) as s:
        assert s.get(UserPreferences, _user(migrated_engine).id) is None  # GET doesn't create
    assert tmdb.calls == []


def test_put_minimal_applies_defaults_and_advances_onboarding(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    resp = _put(api, token)  # empty body: every field takes its default
    assert resp.status_code == 200
    body = resp.json()
    assert body["languages"] == ["en"]
    assert body["content_types"] == ["movie", "tv"]
    assert body["era"] == "any"
    assert body["include_rent_buy"] is False
    assert body["allow_adult"] is False
    assert body["platforms"] == []
    assert body["onboarding_step"] == "seeds"
    assert body["updated_at"] is not None
    assert api.get("/api/v1/me/preferences", headers=_auth(token)).json() == body
    assert tmdb.calls == []  # no platforms → no provider lookup


def test_empty_languages_mean_english(api: TestClient, token: str) -> None:
    resp = _put(api, token, languages=[])
    assert resp.status_code == 200
    assert resp.json()["languages"] == ["en"]


def test_put_full_preferences_round_trip(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    resp = _put(
        api,
        token,
        languages=["EN", "hi", "en", " ko "],
        platforms=[NETFLIX, HBO_MAX, NETFLIX],
        content_types=["movie"],
        era="last_10y",
        avoid_genres=["horror", "Reality", "horror"],
        max_runtime=120,
        include_rent_buy=True,
        allow_adult=True,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["languages"] == ["en", "hi", "ko"]  # lower-cased, deduped, order kept
    assert body["platforms"] == [NETFLIX, HBO_MAX]
    assert body["content_types"] == ["movie"]
    assert body["era"] == "last_10y"
    assert body["avoid_genres"] == ["horror", "reality"]
    assert body["max_runtime"] == 120
    assert body["include_rent_buy"] is True
    assert body["allow_adult"] is True
    with Session(migrated_engine) as s:
        rows = s.scalars(select(UserPlatform)).all()
        assert {(r.provider_id, r.country) for r in rows} == {(NETFLIX, "US"), (HBO_MAX, "US")}


def test_put_replaces_platforms_and_resets_omitted_fields(api: TestClient, token: str) -> None:
    _put(api, token, languages=["en"], platforms=[NETFLIX, HULU], max_runtime=100, era="last_2y")
    body = _put(api, token, languages=["en"], platforms=[DISNEY]).json()
    assert body["platforms"] == [DISNEY]
    assert body["max_runtime"] is None
    assert body["era"] == "any"


def test_invalid_provider_for_country_is_422(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    resp = _put(api, token, languages=["en"], platforms=[NETFLIX, 97, 123456])
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM
    errors = resp.json()["errors"]
    assert [e["loc"] for e in errors] == [["body", "platforms", 1], ["body", "platforms", 2]]
    with Session(migrated_engine) as s:  # nothing saved
        assert s.scalars(select(UserPreferences)).all() == []
        assert s.scalars(select(UserPlatform)).all() == []


def test_platforms_need_provider_list(api: TestClient, token: str, tmdb: FakeTMDB) -> None:
    tmdb.fail = True
    assert _put(api, token, languages=["en"], platforms=[NETFLIX]).status_code == 503
    assert _put(api, token, languages=["en"]).status_code == 200  # no platforms: no lookup


@pytest.mark.parametrize(
    "body",
    [
        {"languages": ["eng"]},
        {"languages": ["e1"]},
        {"languages": ["en"], "content_types": []},
        {"languages": ["en"], "content_types": ["anime"]},
        {"languages": ["en"], "era": "1990s"},
        {"languages": ["en"], "avoid_genres": ["not-a-genre"]},
        {"languages": ["en"], "max_runtime": 0},
        {"languages": ["en"], "max_runtime": 5000},
        {"languages": ["en"], "platforms": [0]},
        {"languages": ["en"], "platforms": ["netflix"]},
        {"languages": ["en"], "onboarding_step": "done"},
        {"languages": ["en"], "country": "IN"},
        {"languages": ["en"], "include_rent_buy": None},
    ],
)
def test_put_validation_errors(api: TestClient, token: str, body: dict[str, Any]) -> None:
    resp = api.put("/api/v1/me/preferences", headers=_auth(token), json=body)
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM


def test_onboarding_step_not_moved_back(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    _put(api, token, languages=["en"])
    with Session(migrated_engine) as s:
        s.execute(update(UserPreferences).values(onboarding_step=OnboardingStep.DONE))
        s.commit()
    assert _put(api, token, languages=["hi"]).json()["onboarding_step"] == "done"


def test_preferences_require_auth(api: TestClient) -> None:
    assert api.get("/api/v1/me/preferences").status_code == 401
    assert api.put("/api/v1/me/preferences", json={"languages": ["en"]}).status_code == 401


def test_preferences_are_per_user(api: TestClient, token: str) -> None:
    _put(api, token, languages=["en"], platforms=[NETFLIX])
    other = _token(api, email="priya@example.com")
    body = api.get("/api/v1/me/preferences", headers=_auth(other)).json()
    assert body["languages"] == ["en"]
    assert body["platforms"] == []


# --- PATCH /me country change ---


def test_country_change_keeps_shared_platforms_and_reports_removed(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    _put(api, token, languages=["en"], platforms=[NETFLIX, HULU, HBO_MAX])
    resp = api.patch("/api/v1/me", headers=_auth(token), json={"country": "KR"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["country"] == "KR"
    assert body["removed_platforms"] == [HULU, HBO_MAX]
    prefs = api.get("/api/v1/me/preferences", headers=_auth(token)).json()
    assert prefs["platforms"] == [NETFLIX]
    with Session(migrated_engine) as s:
        rows = s.scalars(select(UserPlatform)).all()
        assert [(r.provider_id, r.country) for r in rows] == [(NETFLIX, "KR")]


def test_country_change_without_platforms_skips_tmdb(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    resp = api.patch("/api/v1/me", headers=_auth(token), json={"country": "KR"})
    assert resp.status_code == 200
    assert resp.json()["removed_platforms"] == []
    assert tmdb.calls == []


def test_same_country_or_other_fields_keep_platforms(
    api: TestClient, token: str, tmdb: FakeTMDB
) -> None:
    _put(api, token, languages=["en"], platforms=[HULU])
    tmdb.calls.clear()
    for change in ({"country": "us"}, {"display_name": "M"}):
        resp = api.patch("/api/v1/me", headers=_auth(token), json=change)
        assert resp.json()["removed_platforms"] == []
    assert api.get("/api/v1/me/preferences", headers=_auth(token)).json()["platforms"] == [HULU]
    assert tmdb.calls == []


def test_country_change_fails_cleanly_when_provider_list_unavailable(
    api: TestClient, token: str, tmdb: FakeTMDB, migrated_engine: Engine
) -> None:
    _put(api, token, languages=["en"], platforms=[NETFLIX, HULU])
    tmdb.fail = True
    resp = api.patch(
        "/api/v1/me", headers=_auth(token), json={"country": "KR", "display_name": "New"}
    )
    assert resp.status_code == 503
    user = _user(migrated_engine)
    assert (user.country, user.display_name) == ("US", "Marcus")  # nothing changed
    prefs = api.get("/api/v1/me/preferences", headers=_auth(token)).json()
    assert prefs["platforms"] == [NETFLIX, HULU]
