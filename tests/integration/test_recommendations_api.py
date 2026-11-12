"""P1-12 `GET /recommendations` end to end: fixture catalog, real auth/onboarding, mocked TMDB.

The fixture catalog (tests/recsys_catalog.py) has warm US availability, so TMDB is mocked
with a factory that records calls; a test with one cold title checks the fetch path.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete, update
from sqlalchemy.orm import Session

from app.api.deps import get_tmdb_factory
from app.api.recommendations import localized_title, watch_page_url
from app.core.config import Settings
from app.db.models import Title, TitleAvailability, TitleType, User
from app.ingest.tmdb_models import WatchProviders
from app.recsys.availability import allowed_offer_types
from app.schemas.recommendations import RecommendationsOut
from tests.feedback_env import auth, register
from tests.recsys_catalog import NETFLIX, PRIME, FixtureCatalog, build_fixture_catalog

M, T = TitleType.MOVIE, TitleType.TV
URL = "/api/v1/recommendations"


class FakeTMDB:
    """`watch_providers` only: every title streams on Netflix in the US. Records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, TitleType]] = []

    def __call__(self) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def watch_providers(self, type_: TitleType, tmdb_id: int) -> WatchProviders:
        self.calls.append((tmdb_id, type_))
        offer = {"provider_id": NETFLIX, "provider_name": "Netflix", "display_priority": 1}
        return WatchProviders.model_validate(
            {
                "results": {
                    "US": {
                        "link": f"https://www.themoviedb.org/{type_.value}/{tmdb_id}/watch?locale=US",
                        "flatrate": [offer],
                    }
                }
            }
        )


@pytest.fixture
def catalog(db_session: Session, settings: Settings) -> FixtureCatalog:
    fixture = build_fixture_catalog(db_session, settings.embeddings_dir)
    # Posters for the absolute-URL check.
    db_session.execute(update(Title).values(poster_path="/poster.jpg"))
    db_session.commit()
    return fixture


@pytest.fixture
def tmdb() -> FakeTMDB:
    return FakeTMDB()


@pytest.fixture
def rec_client(app: FastAPI, catalog: FixtureCatalog, tmdb: FakeTMDB) -> Iterator[TestClient]:
    # Built after the catalog so the lifespan loads the fixture embedding index.
    app.dependency_overrides[get_tmdb_factory] = lambda: tmdb
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def onboard(
    client: TestClient,
    catalog: FixtureCatalog,
    *,
    email: str = "priya@example.com",
    seeds: int = 3,
    preferences: dict[str, Any] | None = None,
) -> str:
    token = register(client, email)
    body = {"languages": ["en", "hi", "ko"], "platforms": [NETFLIX, PRIME], **(preferences or {})}
    resp = client.put("/api/v1/me/preferences", headers=auth(token), json=body)
    assert resp.status_code == 200, resp.text
    picks = [k for k in catalog.keys if k in catalog.embedded][:seeds]
    resp = client.post(
        "/api/v1/onboarding/seeds",
        headers=auth(token),
        json={"seeds": [{"tmdb_id": i, "type": t.value} for i, t in picks]},
    )
    assert resp.status_code == 201, resp.text
    return token


def keys_of(body: dict[str, Any]) -> list[tuple[int, str]]:
    return [(i["tmdb_id"], i["type"]) for i in body["items"]]


def make_admin(db_session: Session, email: str) -> None:
    db_session.execute(update(User).where(User.email == email).values(is_admin=True))
    db_session.commit()


def seed_with(client: TestClient, token: str, keys: list[tuple[int, TitleType]]) -> None:
    resp = client.post(
        "/api/v1/onboarding/seeds",
        headers=auth(token),
        json={"seeds": [{"tmdb_id": i, "type": t.value} for i, t in keys]},
    )
    assert resp.status_code == 201, resp.text


# --- Happy path / §7 shape -----------------------------------------------------


def test_returns_items_in_section_7_shape(
    rec_client: TestClient, catalog: FixtureCatalog, db_session: Session, tmdb: FakeTMDB
) -> None:
    token = onboard(rec_client, catalog)

    resp = rec_client.get(URL, headers=auth(token))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    out = RecommendationsOut.model_validate(body)  # validates against the schema
    assert out.n == 10
    assert len(out.items) == 10
    assert [i.rank for i in out.items] == list(range(1, 11))
    assert out.attribution.model_dump() == {"availability": "JustWatch", "metadata": "TMDB"}
    assert body["attribution"] == {"availability": "JustWatch", "metadata": "TMDB"}
    assert out.model_version == "fake/e5@v1"
    assert len(out.config_hash) == 16
    assert out.cold_start is False
    assert out.debug is None
    assert tmdb.calls == []  # warm availability cache

    allowed = allowed_offer_types(False)
    for item in out.items:
        title = db_session.get(Title, (item.tmdb_id, item.type))
        assert title is not None
        assert item.original_title == title.original_title
        assert item.language == title.original_language
        assert item.genres == title.genres
        assert item.runtime_min == title.runtime
        assert item.year == (title.release_date.year if title.release_date else None)
        if item.type is M:
            assert item.seasons is None
        assert item.poster_url == "https://image.tmdb.org/t/p/w342/poster.jpg"
        assert item.watch_link.startswith("https://www.themoviedb.org/")
        assert item.availability_checked_at is not None
        assert item.reason
        assert item.reason_source == "template"
        assert item.slot == "core"
        assert item.pool is None
        assert item.components is None
        # FR-22: every item streams on one of the user's platforms; logos are absolute.
        assert item.providers
        for p in item.providers:
            assert p.provider_id in {NETFLIX, PRIME}
            assert p.offer_type in allowed
            assert p.name
            assert p.logo_url is not None
            assert p.logo_url.startswith("https://image.tmdb.org/t/p/")
    assert [i.score for i in out.items] == [round(i.score, 6) for i in out.items]


def test_rating_a_returned_title_changes_the_list(
    rec_client: TestClient, catalog: FixtureCatalog
) -> None:
    token = onboard(rec_client, catalog)
    first = rec_client.get(URL, headers=auth(token)).json()
    top = first["items"][0]

    resp = rec_client.post(
        "/api/v1/interactions",
        headers=auth(token),
        json={"tmdb_id": top["tmdb_id"], "type": top["type"], "kind": "rating", "value": 2},
    )
    assert resp.status_code in (200, 201), resp.text

    second = rec_client.get(URL, headers=auth(token)).json()
    assert keys_of(second) != keys_of(first)
    assert (top["tmdb_id"], top["type"]) not in keys_of(second)  # FR-23
    assert len(second["items"]) == 10


def test_not_interested_title_disappears(rec_client: TestClient, catalog: FixtureCatalog) -> None:
    token = onboard(rec_client, catalog)
    first = rec_client.get(URL, headers=auth(token)).json()
    target = first["items"][3]

    resp = rec_client.post(
        "/api/v1/interactions",
        headers=auth(token),
        json={"tmdb_id": target["tmdb_id"], "type": target["type"], "kind": "not_interested"},
    )
    assert resp.status_code in (200, 201), resp.text

    second = rec_client.get(URL, headers=auth(token)).json()
    assert (target["tmdb_id"], target["type"]) not in keys_of(second)


def test_seeds_are_never_recommended(rec_client: TestClient, catalog: FixtureCatalog) -> None:
    token = onboard(rec_client, catalog, seeds=5)
    seeds = {(i, t.value) for i, t in [k for k in catalog.keys if k in catalog.embedded][:5]}

    body = rec_client.get(URL, headers=auth(token)).json()

    assert seeds.isdisjoint(keys_of(body))


@pytest.mark.parametrize("n", [1, 5, 20])
def test_n_controls_list_size(rec_client: TestClient, catalog: FixtureCatalog, n: int) -> None:
    token = onboard(rec_client, catalog)

    body = rec_client.get(URL, params={"n": n}, headers=auth(token)).json()

    assert body["n"] == n
    assert len(body["items"]) == n


def test_preferences_are_respected(rec_client: TestClient, catalog: FixtureCatalog) -> None:
    token = onboard(rec_client, catalog, preferences={"languages": ["ja"], "content_types": ["tv"]})

    body = rec_client.get(URL, headers=auth(token)).json()

    assert body["items"]
    assert {i["language"] for i in body["items"]} == {"ja"}
    assert {i["type"] for i in body["items"]} == {"tv"}


def test_no_platforms_returns_empty_list(rec_client: TestClient, catalog: FixtureCatalog) -> None:
    token = onboard(rec_client, catalog, preferences={"platforms": []})

    resp = rec_client.get(URL, headers=auth(token))

    assert resp.status_code == 200
    assert resp.json()["items"] == []
    assert resp.json()["attribution"] == {"availability": "JustWatch", "metadata": "TMDB"}


def test_cold_title_is_fetched_from_mocked_tmdb(
    rec_client: TestClient, catalog: FixtureCatalog, db_session: Session, tmdb: FakeTMDB
) -> None:
    token = onboard(rec_client, catalog)
    first = rec_client.get(URL, headers=auth(token)).json()
    top = first["items"][0]
    key = (top["tmdb_id"], TitleType(top["type"]))
    db_session.execute(
        delete(TitleAvailability).where(
            TitleAvailability.tmdb_id == key[0], TitleAvailability.type == key[1]
        )
    )
    db_session.commit()

    second = rec_client.get(URL, headers=auth(token)).json()

    assert key in tmdb.calls
    item = next(i for i in second["items"] if i["tmdb_id"] == top["tmdb_id"])
    assert [p["provider_id"] for p in item["providers"]] == [NETFLIX]
    assert item["providers"][0]["name"] == "Netflix"
    assert item["providers"][0]["logo_url"] == f"https://image.tmdb.org/t/p/w92/{NETFLIX}.png"


# --- Localization -----------------------------------------------------------------


def test_titles_are_localized_to_first_language(
    rec_client: TestClient, catalog: FixtureCatalog, db_session: Session
) -> None:
    db_session.execute(
        update(Title).values(localized_titles={"ko-KR": "한국어 제목", "hi-IN": "हिंदी"})
    )
    db_session.commit()
    token = onboard(rec_client, catalog, preferences={"languages": ["ko", "en"]})

    body = rec_client.get(URL, headers=auth(token)).json()

    assert body["items"]
    assert {i["title"] for i in body["items"]} == {"한국어 제목"}
    assert all(i["original_title"].startswith("Original ") for i in body["items"])


def test_localized_title_fallbacks() -> None:
    title = Title(
        tmdb_id=1,
        type=M,
        title="Inception",
        original_title="Inception",
        localized_titles={"ko-KR": "인셉션", "es-ES": "Origen", "es-MX": "El origen"},
    )
    assert localized_title(title, ["en"], "US") == "Inception"
    assert localized_title(title, ["ko"], "US") == "인셉션"
    assert localized_title(title, ["es"], "MX") == "El origen"
    assert localized_title(title, ["es"], "US") == "Origen"
    assert localized_title(title, ["ja"], "JP") == "Inception"
    assert localized_title(title, [], "US") == "Inception"
    assert watch_page_url(27205, M, "IN") == (
        "https://www.themoviedb.org/movie/27205/watch?locale=IN"
    )


# --- Debug (admins) -----------------------------------------------------------------


def test_debug_for_admin_shows_components(
    rec_client: TestClient, catalog: FixtureCatalog, db_session: Session
) -> None:
    token = onboard(rec_client, catalog)
    make_admin(db_session, "priya@example.com")

    resp = rec_client.get(URL, params={"debug": "true"}, headers=auth(token))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["debug"]["counts"]["catalog"] == len(catalog.keys)
    assert "timings_ms" in body["debug"]
    for item in body["items"]:
        assert set(item["components"]) == {"sim", "quality", "pop", "recency", "neg", "score"}
        assert item["pool"] in {"similar", "trending", "cold_start"}
        assert item["components"]["score"] == pytest.approx(item["score"], abs=1e-6)


def test_debug_for_non_admin_is_forbidden(rec_client: TestClient, catalog: FixtureCatalog) -> None:
    token = onboard(rec_client, catalog)

    resp = rec_client.get(URL, params={"debug": "true"}, headers=auth(token))

    assert resp.status_code == 403
    assert resp.headers["content-type"].startswith("application/problem+json")


# --- Errors -------------------------------------------------------------------------


def test_requires_auth(rec_client: TestClient) -> None:
    resp = rec_client.get(URL)
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/problem+json")


def test_409_without_preferences(rec_client: TestClient) -> None:
    token = register(rec_client)

    resp = rec_client.get(URL, headers=auth(token))

    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["status"] == 409
    assert body["instance"] == URL
    assert "onboarding" in body["detail"].lower()


def test_409_before_seeds(rec_client: TestClient) -> None:
    token = register(rec_client)
    resp = rec_client.put(
        "/api/v1/me/preferences", headers=auth(token), json={"platforms": [NETFLIX]}
    )
    assert resp.status_code == 200
    assert resp.json()["onboarding_step"] == "seeds"

    resp = rec_client.get(URL, headers=auth(token))

    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/problem+json")


@pytest.mark.parametrize("n", ["0", "21", "-1", "ten"])
def test_n_out_of_range_is_422(rec_client: TestClient, catalog: FixtureCatalog, n: str) -> None:
    token = onboard(rec_client, catalog)

    resp = rec_client.get(URL, params={"n": n}, headers=auth(token))

    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["errors"][0]["loc"] == ["query", "n"]


def test_tv_seasons_are_reported(
    rec_client: TestClient, catalog: FixtureCatalog, db_session: Session
) -> None:
    db_session.execute(update(Title).where(Title.type == T).values(seasons=3))
    db_session.commit()
    token = onboard(rec_client, catalog, preferences={"content_types": ["tv"]})

    body = rec_client.get(URL, headers=auth(token)).json()

    assert body["items"]
    assert {i["seasons"] for i in body["items"]} == {3}


# --- Refresh (exclude) and seed replacement -----------------------------------


def test_exclude_returns_the_next_best_titles(
    rec_client: TestClient, catalog: FixtureCatalog
) -> None:
    token = onboard(rec_client, catalog)
    first = rec_client.get(URL, headers=auth(token), params={"n": 5}).json()
    shown = [f"{t}:{i}" for i, t in keys_of(first)]

    resp = rec_client.get(URL, headers=auth(token), params={"n": 5, "exclude": shown})

    assert resp.status_code == 200, resp.text
    second = keys_of(resp.json())
    assert len(second) == 5
    assert not set(second) & set(keys_of(first))
    # Without exclude the list is unchanged (the exclusion is per request, not stored).
    again = rec_client.get(URL, headers=auth(token), params={"n": 5}).json()
    assert keys_of(again) == keys_of(first)


@pytest.mark.parametrize("bad", ["movie", "movie:", "film:1", "tv:abc", "movie:0", ":5"])
def test_malformed_exclude_is_422(
    rec_client: TestClient, catalog: FixtureCatalog, bad: str
) -> None:
    token = onboard(rec_client, catalog)
    resp = rec_client.get(URL, headers=auth(token), params={"exclude": [bad]})
    assert resp.status_code == 422, resp.text


def test_too_many_excludes_is_422(rec_client: TestClient, catalog: FixtureCatalog) -> None:
    token = onboard(rec_client, catalog)
    params = {"exclude": [f"movie:{i}" for i in range(1, 202)]}
    resp = rec_client.get(URL, headers=auth(token), params=params)
    assert resp.status_code == 422, resp.text


def test_replacing_seeds_drops_old_seeds_from_the_profile(
    rec_client: TestClient, catalog: FixtureCatalog, db_session: Session
) -> None:
    embedded = [k for k in catalog.keys if k in catalog.embedded]
    old, new = embedded[:3], embedded[-3:]
    assert not set(old) & set(new)
    token = register(rec_client, "ana@example.com")
    body = {"languages": ["en", "hi", "ko"], "platforms": [NETFLIX, PRIME]}
    assert rec_client.put("/api/v1/me/preferences", headers=auth(token), json=body).is_success
    make_admin(db_session, "ana@example.com")

    seed_with(rec_client, token, old)
    before = rec_client.get(URL, headers=auth(token), params={"debug": True}).json()
    seed_with(rec_client, token, new)
    after = rec_client.get(URL, headers=auth(token), params={"debug": True}).json()

    assert before["debug"]["profile"]["signals_used"] == 3
    assert after["debug"]["profile"]["signals_used"] == 3  # not 6: old seeds are stale
    assert keys_of(after) != keys_of(before)
    returned = {(i, TitleType(t)) for i, t in keys_of(after)}
    assert not returned & (set(old) | set(new))  # removed seeds stay excluded (FR-23)
