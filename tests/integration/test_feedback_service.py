"""P1-09 service tests: `ensure_titles`, `insert_missing_titles`, `excluded_title_keys`,
`OnDemandEmbedder` (TMDB + Wikimedia mocked, fake encoder)."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.db.models import (
    InteractionKind,
    InteractionSource,
    Title,
    TitleAvailability,
    TitleOpenContent,
    User,
)
from app.ingest.ingest_catalog import insert_missing_titles
from app.ingest.tmdb_client import TMDBClient, TMDBUnavailableError
from app.recsys.feedback import (
    UnknownTitlesError,
    add_interactions,
    ensure_titles,
    excluded_title_keys,
)
from tests.catalog_factory import open_content
from tests.fakes import FakeEncoder
from tests.feedback_env import (
    ATTACK_ON_TITAN,
    INCEPTION,
    UNCHECKED_MOVIE,
    FakeTMDBDetails,
    M,
    T,
    add_inception_open_content,
    build_catalog,
    make_embedder,
    no_sleep,
)
from tests.wikimedia_fakes import FakeWikimedia


@pytest.fixture
def emb_dir(tmp_path: Path) -> Path:
    return tmp_path / "embeddings"


@pytest.fixture
def catalog(migrated_engine: Engine, emb_dir: Path) -> Engine:
    build_catalog(migrated_engine, emb_dir, FakeEncoder())
    return migrated_engine


@pytest.fixture
def mocks() -> Iterator[tuple[FakeTMDBDetails, FakeWikimedia]]:
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


def _client() -> TMDBClient:
    key = "test-key"  # pragma: allowlist secret
    return TMDBClient(api_key=key, max_attempts=1, sleep=no_sleep)


def _no_tmdb() -> Any:
    raise AssertionError("TMDB must not be called")


def _user(session: Session, email: str = "u@example.com") -> uuid.UUID:
    user = User(email=email, password_hash="x", display_name="U", country="US", timezone="UTC")
    session.add(user)
    session.commit()
    return user.id


# --- insert_missing_titles ---


async def test_insert_missing_titles_inserts_with_availability(
    catalog: Engine, tmdb: FakeTMDBDetails
) -> None:
    with Session(catalog) as session:
        async with _client() as client:
            inserted, not_found = await insert_missing_titles(
                client, session, [(1, M), (INCEPTION, M), (ATTACK_ON_TITAN, T)], ["US", "KR"]
            )
        assert inserted == [(INCEPTION, M), (ATTACK_ON_TITAN, T)]
        assert not_found == []
        assert tmdb.calls == [("movie", INCEPTION), ("tv", ATTACK_ON_TITAN)]
        aot = session.get(Title, (ATTACK_ON_TITAN, T))
        assert aot is not None
        assert aot.original_language == "ja"
        countries = set(
            session.scalars(
                select(TitleAvailability.country).where(TitleAvailability.tmdb_id == INCEPTION)
            )
        )
        assert countries == {"US", "KR"}


async def test_insert_missing_titles_is_all_or_nothing(
    catalog: Engine, tmdb: FakeTMDBDetails
) -> None:
    with Session(catalog) as session:
        async with _client() as client:
            inserted, not_found = await insert_missing_titles(
                client, session, [(INCEPTION, M), (5, T)], ["US"]
            )
        assert (inserted, not_found) == ([], [(5, T)])
        assert session.get(Title, (INCEPTION, M)) is None


async def test_insert_missing_titles_raises_on_tmdb_failure(
    catalog: Engine, tmdb: FakeTMDBDetails
) -> None:
    tmdb.status = 503
    with Session(catalog) as session:
        async with _client() as client:
            with pytest.raises(TMDBUnavailableError):
                await insert_missing_titles(client, session, [(INCEPTION, M)], ["US"])
        assert session.get(Title, (INCEPTION, M)) is None


# --- ensure_titles ---


async def test_ensure_titles_skips_tmdb_when_all_in_catalog(catalog: Engine, emb_dir: Path) -> None:
    embedder = make_embedder(emb_dir, wikimedia=False)
    with Session(catalog) as session:
        result = await ensure_titles(
            session, [(1, M), (2, M)], tmdb_factory=_no_tmdb, countries=["US"], embedder=embedder
        )
    assert result.inserted == []
    assert result.embedded == {(1, M): True, (2, M): True}


async def test_ensure_titles_unknown_raises(catalog: Engine, tmdb: FakeTMDBDetails) -> None:
    with Session(catalog) as session, pytest.raises(UnknownTitlesError) as info:
        await ensure_titles(
            session, [(1, M), (9, T)], tmdb_factory=_client, countries=["US"], embedder=None
        )
    assert info.value.keys == [(9, T)]


async def test_ensure_titles_without_embedder(catalog: Engine, tmdb: FakeTMDBDetails) -> None:
    with Session(catalog) as session:
        result = await ensure_titles(
            session, [(INCEPTION, M)], tmdb_factory=_client, countries=["US"], embedder=None
        )
    assert result.inserted == [(INCEPTION, M)]
    assert result.embedded == {(INCEPTION, M): False}


# --- OnDemandEmbedder ---


async def test_embedder_fetches_open_content_only_for_unchecked_titles(
    catalog: Engine, emb_dir: Path, tmdb: FakeTMDBDetails, wiki: FakeWikimedia
) -> None:
    add_inception_open_content(wiki)
    embedder = make_embedder(emb_dir)
    with Session(catalog) as session:
        await ensure_titles(
            session, [(INCEPTION, M)], tmdb_factory=_client, countries=["US"], embedder=None
        )
        # Catalog movie 1 is already embedded; 6 and Inception were never checked.
        embedded = await embedder.ensure(session, [(1, M), (UNCHECKED_MOVIE, M), (INCEPTION, M)])
        assert embedded == {(1, M): True, (UNCHECKED_MOVIE, M): False, (INCEPTION, M): True}
        assert wiki.sparql_calls == [(M, [UNCHECKED_MOVIE, INCEPTION])]  # one batch
        # Checked-without-match rows are stored, so the next call makes no Wikimedia request.
        assert session.get(TitleOpenContent, (UNCHECKED_MOVIE, M)) is not None
        await embedder.ensure(session, [(UNCHECKED_MOVIE, M)])
        assert len(wiki.sparql_calls) == 1


async def test_embedder_without_wikimedia_uses_stored_content_only(
    catalog: Engine, emb_dir: Path, wiki: FakeWikimedia
) -> None:
    embedder = make_embedder(emb_dir, wikimedia=False)
    with Session(catalog) as session:
        assert await embedder.ensure(session, [(UNCHECKED_MOVIE, M)]) == {
            (UNCHECKED_MOVIE, M): False
        }
    assert wiki.sparql_calls == []


async def test_embedder_creates_encoder_once(catalog: Engine, emb_dir: Path) -> None:
    created: list[FakeEncoder] = []

    def factory() -> FakeEncoder:
        created.append(FakeEncoder())
        return created[-1]

    embedder = make_embedder(emb_dir, wikimedia=False)
    embedder._encoder_factory = factory
    with Session(catalog) as session:
        session.add(open_content(UNCHECKED_MOVIE, M))
        session.commit()
        assert await embedder.ensure(session, [(UNCHECKED_MOVIE, M)]) == {
            (UNCHECKED_MOVIE, M): True
        }
        await embedder.ensure(session, [(UNCHECKED_MOVIE, M), (1, M)])
    assert len(created) == 1


async def test_embedder_skips_vectors_from_another_model(catalog: Engine, emb_dir: Path) -> None:
    embedder = make_embedder(emb_dir, wikimedia=False, encoder=FakeEncoder(name="other/model"))
    with Session(catalog) as session:
        session.add(open_content(UNCHECKED_MOVIE, M))
        session.commit()
        assert await embedder.ensure(session, [(UNCHECKED_MOVIE, M)]) == {
            (UNCHECKED_MOVIE, M): False
        }


# --- Interactions + FR-23 exclusions ---


def test_excluded_title_keys_kinds(catalog: Engine) -> None:
    with Session(catalog) as session:
        user = _user(session)
        other = _user(session, "o@example.com")
        add = add_interactions
        add(session, user, [(1, M)], kind=InteractionKind.RATING, source=InteractionSource.BROWSE,
            value=4)  # fmt: skip
        add(session, user, [(2, M)], kind=InteractionKind.SEEN, source=InteractionSource.BROWSE)
        add(session, user, [(3, M)], kind=InteractionKind.NOT_INTERESTED,
            source=InteractionSource.BROWSE)  # fmt: skip
        add(session, user, [(4, M)], kind=InteractionKind.WATCHLIST,
            source=InteractionSource.BROWSE)  # fmt: skip
        add(session, user, [(5, M)], kind=InteractionKind.CLICK, source=InteractionSource.BROWSE)
        add(session, other, [(UNCHECKED_MOVIE, M)], kind=InteractionKind.NOT_INTERESTED,
            source=InteractionSource.BROWSE)  # fmt: skip
        session.commit()
        assert excluded_title_keys(session, user) == {(1, M), (2, M), (3, M)}
        assert excluded_title_keys(session, other) == {(UNCHECKED_MOVIE, M)}
        assert excluded_title_keys(session, uuid.uuid4()) == set()
