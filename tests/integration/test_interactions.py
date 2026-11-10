"""P1-09: `POST /interactions` (rating, not_interested) and the FR-23 exclusion set."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.db.models import (
    Interaction,
    InteractionKind,
    InteractionSource,
    Rating,
    Title,
    User,
)
from app.recsys.embeddings import OnDemandEmbedder
from app.recsys.feedback import excluded_title_keys
from tests.fakes import FakeEncoder
from tests.feedback_env import (
    ATTACK_ON_TITAN,
    INCEPTION,
    PROBLEM,
    FakeTMDBDetails,
    M,
    T,
    add_inception_open_content,
    auth,
    build_catalog,
    make_embedder,
    register,
    save_preferences,
    seed,
    use_fake_tmdb,
)
from tests.wikimedia_fakes import FakeWikimedia

URL = "/api/v1/interactions"


@pytest.fixture
def mocks(app: FastAPI) -> Iterator[tuple[FakeTMDBDetails, FakeWikimedia]]:
    use_fake_tmdb(app)
    with respx.mock(assert_all_called=False) as router:
        wiki = FakeWikimedia()
        wiki.install(router)
        yield FakeTMDBDetails(router), wiki


@pytest.fixture
def tmdb(mocks: tuple[FakeTMDBDetails, FakeWikimedia]) -> FakeTMDBDetails:
    return mocks[0]


@pytest.fixture
def wiki(mocks: tuple[FakeTMDBDetails, FakeWikimedia]) -> FakeWikimedia:
    return mocks[1]


@pytest.fixture
def embedder(app: FastAPI, migrated_engine: Engine, tmp_path: Path) -> OnDemandEmbedder:
    emb_dir = tmp_path / "embeddings"
    build_catalog(migrated_engine, emb_dir, FakeEncoder())
    embedder = make_embedder(emb_dir)
    app.state.embedder = embedder
    return embedder


@pytest.fixture
def api(
    client: TestClient, embedder: OnDemandEmbedder, mocks: tuple[FakeTMDBDetails, FakeWikimedia]
) -> TestClient:
    return client


@pytest.fixture
def token(api: TestClient) -> str:
    return register(api)


def _post(client: TestClient, token: str, **body: Any) -> httpx.Response:
    return client.post(URL, headers=auth(token), json=body)


def _user_id(engine: Engine, email: str = "priya@example.com") -> Any:
    with Session(engine) as session:
        return session.scalars(select(User.id).where(User.email == email)).one()


def _excluded(engine: Engine, email: str = "priya@example.com") -> set[tuple[int, Any]]:
    with Session(engine) as session:
        return excluded_title_keys(session, _user_id(engine, email))


# --- Rating ---


def test_rating_is_recorded(api: TestClient, token: str, migrated_engine: Engine) -> None:
    resp = _post(api, token, tmdb_id=1, type="movie", kind="rating", value=4)

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["tmdb_id"] == 1
    assert body["type"] == "movie"
    assert body["kind"] == "rating"
    assert body["value"] == 4.0
    assert body["source"] == "browse"
    assert body["added_to_catalog"] is False
    assert body["embedded"] is True
    assert isinstance(body["id"], int)
    with Session(migrated_engine) as session:
        row = session.get(Interaction, body["id"])
        assert row is not None
        assert (row.kind, row.value, row.source) == (
            InteractionKind.RATING,
            4.0,
            InteractionSource.BROWSE,
        )


def test_rerating_appends_and_latest_counts(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    assert _post(api, token, tmdb_id=1, type="movie", kind="rating", value=5).status_code == 201
    assert _post(api, token, tmdb_id=1, type="movie", kind="rating", value=2).status_code == 201
    with Session(migrated_engine) as session:
        assert len(list(session.scalars(select(Interaction)))) == 2
        (rating,) = session.scalars(select(Rating)).all()
        assert rating.rating == 2.0


@pytest.mark.parametrize("value", [0, 6, 4.5, None, "five"])
def test_bad_rating_value_is_422(api: TestClient, token: str, value: Any) -> None:
    resp = _post(api, token, tmdb_id=1, type="movie", kind="rating", value=value)
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM


# --- Not interested ---


def test_not_interested_is_recorded(api: TestClient, token: str) -> None:
    resp = _post(api, token, tmdb_id=2, type="movie", kind="not_interested")
    assert resp.status_code == 201
    assert resp.json()["kind"] == "not_interested"
    assert resp.json()["value"] is None


def test_not_interested_with_value_is_422(api: TestClient, token: str) -> None:
    resp = _post(api, token, tmdb_id=2, type="movie", kind="not_interested", value=1)
    assert resp.status_code == 422


# --- Validation ---


@pytest.mark.parametrize(
    "body",
    [
        {"tmdb_id": 1, "type": "movie", "kind": "seen"},  # P2-06
        {"tmdb_id": 1, "type": "movie", "kind": "thumb", "value": 1},  # P2-06
        {"tmdb_id": 1, "type": "person", "kind": "not_interested"},
        {"tmdb_id": -1, "type": "movie", "kind": "not_interested"},
        {"type": "movie", "kind": "not_interested"},
        {"tmdb_id": 1, "type": "movie", "kind": "not_interested", "source": "agent"},
        {"tmdb_id": 1, "type": "movie", "kind": "not_interested", "user_id": "x"},
    ],
)
def test_invalid_bodies_are_422(
    api: TestClient, token: str, migrated_engine: Engine, body: dict[str, Any]
) -> None:
    resp = api.post(URL, headers=auth(token), json=body)
    assert resp.status_code == 422
    with Session(migrated_engine) as session:
        assert list(session.scalars(select(Interaction))) == []


def test_requires_auth(api: TestClient) -> None:
    resp = api.post(URL, json={"tmdb_id": 1, "type": "movie", "kind": "not_interested"})
    assert resp.status_code == 401


def test_does_not_require_finished_onboarding(api: TestClient, token: str) -> None:
    # No preferences saved yet: feedback is still accepted.
    assert _post(api, token, tmdb_id=1, type="movie", kind="not_interested").status_code == 201


# --- Titles not in the catalog ---


def test_title_not_in_catalog_is_fetched_on_demand(
    api: TestClient,
    token: str,
    migrated_engine: Engine,
    tmdb: FakeTMDBDetails,
    wiki: FakeWikimedia,
    embedder: OnDemandEmbedder,
) -> None:
    add_inception_open_content(wiki)
    resp = _post(api, token, tmdb_id=INCEPTION, type="movie", kind="rating", value=5)
    assert resp.status_code == 201
    assert resp.json()["added_to_catalog"] is True
    assert resp.json()["embedded"] is True
    assert tmdb.calls == [("movie", INCEPTION)]
    assert (INCEPTION, M) in embedder.store
    with Session(migrated_engine) as session:
        assert session.get(Title, (INCEPTION, M)) is not None

    # Second time: already in the catalog and the index, so no TMDB or Wikimedia calls.
    calls = len(wiki.sparql_calls)
    resp = _post(api, token, tmdb_id=INCEPTION, type="movie", kind="not_interested")
    assert resp.status_code == 201
    assert resp.json()["added_to_catalog"] is False
    assert tmdb.calls == [("movie", INCEPTION)]
    assert len(wiki.sparql_calls) == calls


def test_unknown_title_is_422(api: TestClient, token: str, migrated_engine: Engine) -> None:
    resp = _post(api, token, tmdb_id=424242, type="tv", kind="not_interested")
    assert resp.status_code == 422
    assert resp.json()["errors"] == [
        {
            "loc": ["body", "tmdb_id"],
            "msg": "no tv with TMDB id 424242",
            "type": "value_error.title_not_found",
        }
    ]
    with Session(migrated_engine) as session:
        assert list(session.scalars(select(Interaction))) == []


def test_tmdb_down_is_503(api: TestClient, token: str, tmdb: FakeTMDBDetails) -> None:
    tmdb.status = 500
    resp = _post(api, token, tmdb_id=ATTACK_ON_TITAN, type="tv", kind="rating", value=3)
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "60"


# --- FR-23: rated / not-interested titles are excluded from recommendations ---


def test_rated_and_not_interested_titles_are_excluded(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    assert _post(api, token, tmdb_id=1, type="movie", kind="rating", value=1).status_code == 201
    assert _post(api, token, tmdb_id=2, type="movie", kind="not_interested").status_code == 201
    assert (
        _post(api, token, tmdb_id=ATTACK_ON_TITAN, type="tv", kind="rating", value=3).status_code
        == 201
    )
    assert _excluded(migrated_engine) == {(1, M), (2, M), (ATTACK_ON_TITAN, T)}


def test_exclusions_are_per_user(api: TestClient, token: str, migrated_engine: Engine) -> None:
    other = register(api, "marcus@example.com")
    assert _post(api, other, tmdb_id=3, type="movie", kind="not_interested").status_code == 201
    assert _post(api, token, tmdb_id=1, type="movie", kind="rating", value=5).status_code == 201
    assert _excluded(migrated_engine) == {(1, M)}
    assert _excluded(migrated_engine, "marcus@example.com") == {(3, M)}


def test_seeds_and_feedback_combine(api: TestClient, token: str, migrated_engine: Engine) -> None:
    save_preferences(api, token)
    resp = api.post(
        "/api/v1/onboarding/seeds",
        headers=auth(token),
        json={"seeds": [seed(1), seed(2), seed(3)]},
    )
    assert resp.status_code == 201
    assert _post(api, token, tmdb_id=4, type="movie", kind="not_interested").status_code == 201
    assert _excluded(migrated_engine) == {(1, M), (2, M), (3, M), (4, M)}
