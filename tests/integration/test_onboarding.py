"""P1-09: `POST /onboarding/seeds` (TMDB and Wikimedia mocked, fake encoder)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select
from sqlalchemy import update as sa_update
from sqlalchemy.orm import Session

from app.api.deps import get_tmdb_factory
from app.db.models import (
    Interaction,
    InteractionKind,
    InteractionSource,
    Title,
    TitleAvailability,
    TitleOpenContent,
    UserPreferences,
)
from app.ingest.build_embeddings import ID_MAP_FILE, embed_one
from app.ingest.tmdb_client import TMDBConfigError
from app.recsys.embeddings import OnDemandEmbedder
from app.recsys.feedback import current_seed_keys, excluded_title_keys, stale_seed_ids
from tests.catalog_factory import add_catalog, tmdb_title
from tests.fakes import FakeEncoder
from tests.feedback_env import (
    ATTACK_ON_TITAN,
    INCEPTION,
    PROBLEM,
    UNCHECKED_MOVIE,
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
from tests.wikimedia_fakes import Entity, FakeWikimedia

URL = "/api/v1/onboarding/seeds"


@pytest.fixture
def emb_dir(tmp_path: Path) -> Path:
    return tmp_path / "embeddings"


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


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
def embedder(
    app: FastAPI, migrated_engine: Engine, emb_dir: Path, encoder: FakeEncoder
) -> OnDemandEmbedder:
    build_catalog(migrated_engine, emb_dir, FakeEncoder())
    embedder = make_embedder(emb_dir, encoder=encoder)
    app.state.embedder = embedder
    return embedder


@pytest.fixture
def api(
    client: TestClient, embedder: OnDemandEmbedder, mocks: tuple[FakeTMDBDetails, FakeWikimedia]
) -> TestClient:
    return client


@pytest.fixture
def token(api: TestClient) -> str:
    token = register(api)
    save_preferences(api, token)
    return token


def _post(client: TestClient, token: str, seeds: list[dict[str, Any]]) -> httpx.Response:
    return client.post(URL, headers=auth(token), json={"seeds": seeds})


def _interactions(engine: Engine) -> list[Interaction]:
    with Session(engine) as session:
        return list(session.scalars(select(Interaction).order_by(Interaction.id)))


def _count(engine: Engine, model: type[Any]) -> int:
    with Session(engine) as session:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)


# --- Happy path ---


def test_catalog_seeds_are_stored_as_five_star_ratings(
    api: TestClient, token: str, migrated_engine: Engine, tmdb: FakeTMDBDetails
) -> None:
    resp = _post(api, token, [seed(1), seed(2), seed(3)])

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["onboarding_step"] == "done"
    assert [(s["tmdb_id"], s["type"]) for s in body["seeds"]] == [
        (1, "movie"),
        (2, "movie"),
        (3, "movie"),
    ]
    assert all(s["embedded"] and not s["added_to_catalog"] for s in body["seeds"])
    rows = _interactions(migrated_engine)
    assert [(r.tmdb_id, r.type, r.kind, r.value, r.source) for r in rows] == [
        (i, M, InteractionKind.RATING, 5.0, InteractionSource.ONBOARDING_SEED) for i in (1, 2, 3)
    ]
    assert tmdb.calls == []  # all in the catalog: TMDB not called


def test_seeding_finishes_onboarding(api: TestClient, token: str) -> None:
    assert _post(api, token, [seed(1), seed(2), seed(3)]).status_code == 201
    prefs = api.get("/api/v1/me/preferences", headers=auth(token)).json()
    assert prefs["onboarding_step"] == "done"


def test_seed_titles_are_excluded_from_recommendations(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    assert _post(api, token, [seed(1), seed(2), seed(3)]).status_code == 201
    with Session(migrated_engine) as session:
        user_id = session.scalars(select(UserPreferences.user_id)).one()
        assert excluded_title_keys(session, user_id) == {(1, M), (2, M), (3, M)}


def test_resubmitting_replaces_the_seed_set(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    assert _post(api, token, [seed(1), seed(2), seed(3)]).status_code == 201
    resp = _post(api, token, [seed(3), seed(4), seed(5)])
    assert resp.status_code == 201
    assert resp.json()["onboarding_step"] == "done"
    # The log is append-only: the first set's rows stay but become stale.
    rows = _interactions(migrated_engine)
    assert len(rows) == 6
    first = rows[:3]
    sets = [(r.payload or {}).get("seed_set") for r in rows]
    assert None not in sets
    assert len(set(sets[:3])) == 1
    assert len(set(sets[3:])) == 1
    assert sets[0] != sets[3]
    with Session(migrated_engine) as session:
        user_id = session.scalars(select(UserPreferences.user_id)).one()
        assert current_seed_keys(session, user_id) == [(3, M), (4, M), (5, M)]
        assert stale_seed_ids(session, user_id) == {r.id for r in first}
        # Removed seeds are still never recommended (FR-23).
        assert excluded_title_keys(session, user_id) >= {(1, M), (2, M)}

    listed = api.get("/api/v1/onboarding/seeds", headers=auth(token))
    assert listed.status_code == 200
    assert [(s["tmdb_id"], s["type"]) for s in listed.json()["seeds"]] == [
        (3, "movie"),
        (4, "movie"),
        (5, "movie"),
    ]
    assert set(listed.json()["seeds"][0]) == {
        "tmdb_id",
        "type",
        "title",
        "original_title",
        "year",
    }


def test_legacy_seeds_without_a_set_id_stay_current_until_resubmitted(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    assert _post(api, token, [seed(1), seed(2), seed(3)]).status_code == 201
    with Session(migrated_engine) as session:
        user_id = session.scalars(select(UserPreferences.user_id)).one()
        # Rows saved before seed sets existed have no payload.
        session.execute(sa_update(Interaction).values(payload=None))
        session.commit()
        assert current_seed_keys(session, user_id) == [(1, M), (2, M), (3, M)]
        assert stale_seed_ids(session, user_id) == set()
    assert _post(api, token, [seed(4), seed(5), seed(6)]).status_code == 201
    with Session(migrated_engine) as session:
        assert current_seed_keys(session, user_id) == [(4, M), (5, M), (6, M)]
        assert len(stale_seed_ids(session, user_id)) == 3


def test_listing_seeds_before_onboarding_is_empty(api: TestClient, token: str) -> None:
    resp = api.get("/api/v1/onboarding/seeds", headers=auth(token))
    assert resp.status_code == 200
    assert resp.json() == {"seeds": []}


# --- Validation ---


@pytest.mark.parametrize("count", [0, 1, 2, 11])
def test_wrong_number_of_seeds_is_422(
    api: TestClient, token: str, migrated_engine: Engine, count: int
) -> None:
    resp = _post(api, token, [seed(100 + i) for i in range(count)])
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM
    assert resp.json()["errors"][0]["loc"] == ["body", "seeds"]
    assert _interactions(migrated_engine) == []


def test_exactly_ten_seeds_is_accepted(
    api: TestClient, token: str, migrated_engine: Engine, wiki: FakeWikimedia
) -> None:
    with Session(migrated_engine) as session:
        add_catalog(session, tmdb_title(7, M), tmdb_title(8, M))
    seeds = [seed(i) for i in (1, 2, 3, 4, 5, UNCHECKED_MOVIE, 7, 8)]
    seeds += [seed(INCEPTION), seed(ATTACK_ON_TITAN, "tv")]
    resp = _post(api, token, seeds)
    assert resp.status_code == 201, resp.text
    assert len(resp.json()["seeds"]) == 10
    assert len(_interactions(migrated_engine)) == 10


def test_unknown_titles_report_their_positions(api: TestClient, token: str) -> None:
    resp = _post(api, token, [seed(1), seed(2), seed(3), seed(1, "tv"), seed(2, "tv")])
    assert resp.status_code == 422
    assert [e["loc"] for e in resp.json()["errors"]] == [["body", "seeds", 3], ["body", "seeds", 4]]


def test_duplicates_are_deduped(api: TestClient, token: str, migrated_engine: Engine) -> None:
    resp = _post(api, token, [seed(1), seed(1), seed(2), seed(3), seed(2)])
    assert resp.status_code == 201
    assert [s["tmdb_id"] for s in resp.json()["seeds"]] == [1, 2, 3]
    assert [r.tmdb_id for r in _interactions(migrated_engine)] == [1, 2, 3]


def test_duplicates_do_not_count_towards_the_minimum(api: TestClient, token: str) -> None:
    resp = _post(api, token, [seed(1), seed(1), seed(2)])
    assert resp.status_code == 422
    assert "got 2" in resp.json()["errors"][0]["msg"]


def test_same_id_different_type_are_distinct(
    api: TestClient, token: str, migrated_engine: Engine
) -> None:
    resp = _post(api, token, [seed(1), seed(2), seed(ATTACK_ON_TITAN, "tv"), seed(2)])
    assert resp.status_code == 201
    assert [(s["tmdb_id"], s["type"]) for s in resp.json()["seeds"]] == [
        (1, "movie"),
        (2, "movie"),
        (ATTACK_ON_TITAN, "tv"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        {"seeds": [seed(1), seed(2), {"tmdb_id": 3, "type": "person"}]},
        {"seeds": [seed(1), seed(2), {"tmdb_id": 0, "type": "movie"}]},
        {"seeds": [seed(1), seed(2), {"tmdb_id": 3}]},
        {"seeds": [seed(1), seed(2), seed(3)], "user_id": "someone-else"},
        {"seeds": [seed(i) for i in range(1, 22)]},  # over the raw list cap
        {},
    ],
)
def test_invalid_bodies_are_422(api: TestClient, token: str, body: dict[str, Any]) -> None:
    resp = api.post(URL, headers=auth(token), json=body)
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM


def test_requires_auth(api: TestClient) -> None:
    resp = api.post(URL, json={"seeds": [seed(1), seed(2), seed(3)]})
    assert resp.status_code == 401


def test_requires_saved_preferences(api: TestClient, migrated_engine: Engine) -> None:
    token = register(api, "new@example.com")
    resp = _post(api, token, [seed(1), seed(2), seed(3)])
    assert resp.status_code == 409
    assert resp.headers["content-type"] == PROBLEM
    assert _interactions(migrated_engine) == []


# --- Titles not in the catalog ---


def test_seed_not_in_catalog_is_fetched_inserted_and_embedded(
    api: TestClient,
    token: str,
    migrated_engine: Engine,
    tmdb: FakeTMDBDetails,
    wiki: FakeWikimedia,
    embedder: OnDemandEmbedder,
    encoder: FakeEncoder,
    emb_dir: Path,
) -> None:
    add_inception_open_content(wiki)
    ids_on_disk = json.loads((emb_dir / ID_MAP_FILE).read_text())

    resp = _post(api, token, [seed(1), seed(2), seed(INCEPTION)])

    assert resp.status_code == 201, resp.text
    inception = resp.json()["seeds"][2]
    assert inception == {
        "tmdb_id": INCEPTION,
        "type": "movie",
        "title": "Inception",
        "original_title": "Inception",
        "added_to_catalog": True,
        "embedded": True,
    }
    assert tmdb.calls == [("movie", INCEPTION)]
    with Session(migrated_engine) as session:
        title = session.get(Title, (INCEPTION, M))
        assert title is not None
        assert title.runtime == 148
        # Availability stored for the seed countries (US by default).
        countries = set(
            session.scalars(
                select(TitleAvailability.country).where(TitleAvailability.tmdb_id == INCEPTION)
            )
        )
        assert countries == {"US"}
        assert session.get(TitleOpenContent, (INCEPTION, M)) is not None
        # Usable immediately: the vector is in the in-memory index (no re-embed) and equals
        # what `make embed --incremental` would compute from the stored open content.
        vector = embedder.store.vector((INCEPTION, M))
        assert vector is not None
        expected = await_embed_one(session, embedder)
        assert np.allclose(vector, expected, atol=1e-6)
    assert embedder.store.appended == 1
    assert encoder.encoded_texts == [
        "passage: Inception. Inception is a 2010 science fiction action film."
    ]
    # Files on disk are untouched (persisted by the next `make embed`).
    assert json.loads((emb_dir / ID_MAP_FILE).read_text()) == ids_on_disk


def await_embed_one(session: Session, embedder: OnDemandEmbedder) -> Any:
    """What the index row for Inception would be, computed independently."""
    meta = embedder.store.meta
    assert meta is not None
    return asyncio.run(embed_one((INCEPTION, M), session=session, encoder=FakeEncoder(), meta=meta))


def test_seed_without_open_content_is_stored_but_not_embedded(
    api: TestClient, token: str, migrated_engine: Engine, embedder: OnDemandEmbedder
) -> None:
    # Attack on Titan has no Wikidata match in the fake: stored, but not embeddable.
    resp = _post(api, token, [seed(1), seed(2), seed(ATTACK_ON_TITAN, "tv")])
    assert resp.status_code == 201
    aot = resp.json()["seeds"][2]
    assert aot["added_to_catalog"] is True
    assert aot["embedded"] is False
    assert (ATTACK_ON_TITAN, T) not in embedder.store
    assert len(_interactions(migrated_engine)) == 3
    with Session(migrated_engine) as session:
        checked = session.get(TitleOpenContent, (ATTACK_ON_TITAN, T))
        assert checked is not None
        assert checked.wikidata_qid is None


def test_catalog_title_never_checked_gets_open_content_on_demand(
    api: TestClient, token: str, wiki: FakeWikimedia, embedder: OnDemandEmbedder
) -> None:
    wiki.add(
        UNCHECKED_MOVIE,
        M,
        Entity(qid="Q6", label="Six", sitelinks={"en": "Six"}),
        summary="Six is a film.",
    )
    resp = _post(api, token, [seed(1), seed(2), seed(UNCHECKED_MOVIE)])
    assert resp.status_code == 201
    assert resp.json()["seeds"][2]["added_to_catalog"] is False
    assert resp.json()["seeds"][2]["embedded"] is True
    assert (UNCHECKED_MOVIE, M) in embedder.store
    # Only the unchecked title was looked up on Wikidata.
    assert wiki.sparql_calls == [(M, [UNCHECKED_MOVIE])]


def test_unknown_title_is_422_and_nothing_is_saved(
    api: TestClient, token: str, migrated_engine: Engine, wiki: FakeWikimedia
) -> None:
    add_inception_open_content(wiki)
    resp = _post(api, token, [seed(1), seed(INCEPTION), seed(999_999)])
    assert resp.status_code == 422
    errors = resp.json()["errors"]
    assert errors == [
        {
            "loc": ["body", "seeds", 2],
            "msg": "no movie with TMDB id 999999",
            "type": "value_error.title_not_found",
        }
    ]
    assert _interactions(migrated_engine) == []
    with Session(migrated_engine) as session:
        assert session.get(Title, (INCEPTION, M)) is None  # all or nothing
        prefs = session.scalars(select(UserPreferences)).one()
        assert prefs.onboarding_step.value == "seeds"


def test_tmdb_down_is_503(
    api: TestClient, token: str, migrated_engine: Engine, tmdb: FakeTMDBDetails
) -> None:
    tmdb.status = 503
    resp = _post(api, token, [seed(1), seed(2), seed(INCEPTION)])
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "60"
    assert resp.headers["content-type"] == PROBLEM
    assert _interactions(migrated_engine) == []


def test_missing_tmdb_credentials_is_503(app: FastAPI, api: TestClient, token: str) -> None:
    def factory() -> Any:
        raise TMDBConfigError("Set TMDB_READ_TOKEN or TMDB_API_KEY")

    app.dependency_overrides[get_tmdb_factory] = lambda: factory
    resp = _post(api, token, [seed(1), seed(2), seed(INCEPTION)])
    assert resp.status_code == 503


# --- Embedding is best effort ---


def test_without_index_seeds_are_stored_not_embedded(
    app: FastAPI, api: TestClient, token: str, tmp_path: Path, migrated_engine: Engine
) -> None:
    app.state.embedder = make_embedder(tmp_path / "no-index")
    resp = _post(api, token, [seed(1), seed(2), seed(3)])
    assert resp.status_code == 201
    assert not any(s["embedded"] for s in resp.json()["seeds"])
    assert len(_interactions(migrated_engine)) == 3


def test_encoder_unavailable_still_stores_seeds(
    app: FastAPI, api: TestClient, token: str, emb_dir: Path, wiki: FakeWikimedia
) -> None:
    add_inception_open_content(wiki)
    app.state.embedder = make_embedder(emb_dir, encoder_error=True)
    resp = _post(api, token, [seed(1), seed(2), seed(INCEPTION)])
    assert resp.status_code == 201
    assert [s["embedded"] for s in resp.json()["seeds"]] == [True, True, False]


def test_wikimedia_down_still_stores_seeds(
    api: TestClient, token: str, wiki: FakeWikimedia, migrated_engine: Engine
) -> None:
    wiki.sparql_status = 500
    resp = _post(api, token, [seed(1), seed(2), seed(INCEPTION)])
    assert resp.status_code == 201
    assert resp.json()["seeds"][2]["embedded"] is False
    with Session(migrated_engine) as session:
        # Not marked as checked, so a later request retries the lookup.
        assert session.get(TitleOpenContent, (INCEPTION, M)) is None
