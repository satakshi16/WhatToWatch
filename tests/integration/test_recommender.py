"""P1-11 `RecommendationService` end to end over the fixture catalog (warm US availability).

Covers the acceptance list: 3 seeds only, hard filters + FR-22 on every result, FR-25
caps / single-language relaxation, cold start, FR-23 exclusions, runtime-appended
vectors, availability budget, config hash and debug output.
"""

from __future__ import annotations

import random
import uuid
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import (
    Era,
    InteractionKind,
    InteractionSource,
    OfferType,
    Title,
    TitleAvailability,
    TitleType,
    User,
)
from app.ingest.tmdb_models import WatchProviders
from app.recsys.availability import AvailabilityService, allowed_offer_types
from app.recsys.embeddings import EmbeddingStore
from app.recsys.feedback import excluded_title_keys
from app.recsys.service import (
    CatalogCache,
    PreferencesMissingError,
    Recommendations,
    RecommendationService,
    UserNotFoundError,
    config_hash,
)
from tests.recsys_catalog import (
    DIM,
    GENRES,
    LANGUAGES,
    NETFLIX,
    NOW,
    PRIME,
    TUBI,
    FixtureCatalog,
    add_signal,
    build_fixture_catalog,
    make_user,
    passes_hard_filters,
)

M, T = TitleType.MOVIE, TitleType.TV


class NoTMDB:
    """TMDB factory for a warm cache: any use is a test failure."""

    def __call__(self) -> Any:
        raise AssertionError("TMDB must not be called when availability is cached")


class FakeTMDB:
    """Every title streams on Netflix in the US; records calls."""

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
            {"results": {"US": {"link": "https://tmdb.test/watch", "flatrate": [offer]}}}
        )


@pytest.fixture
def emb_dir(tmp_path: Path) -> Path:
    return tmp_path / "emb"


@pytest.fixture
def catalog(db_session: Session, emb_dir: Path) -> FixtureCatalog:
    return build_fixture_catalog(db_session, emb_dir)


@pytest.fixture
def store(emb_dir: Path, catalog: FixtureCatalog) -> EmbeddingStore:
    return EmbeddingStore(emb_dir)


def service(
    session: Session,
    store: EmbeddingStore,
    settings: Settings,
    *,
    tmdb: Any = None,
    cache: CatalogCache | None = None,
) -> RecommendationService:
    availability = AvailabilityService(
        session,
        tmdb or NoTMDB(),
        ttl=timedelta(days=settings.availability_ttl_days),
        max_fetches=settings.availability_max_fetches_per_request,
    )
    return RecommendationService(
        session,
        store=store,
        availability=availability,
        settings=settings,
        catalog_cache=cache,
        clock=lambda: NOW,
    )


async def recommend(
    session: Session,
    store: EmbeddingStore,
    settings: Settings,
    user: User,
    n: int = 10,
    **kwargs: Any,
) -> Recommendations:
    return await service(session, store, settings, **kwargs).recommend(user.id, n)


def seed_titles(
    session: Session, user: User, cat: FixtureCatalog, *, genre: str, count: int = 3
) -> list[tuple[int, TitleType]]:
    """Rate `count` embedded titles of `genre` in the user's languages 5 stars (seeds)."""
    assert user.preferences is not None
    langs = set(user.preferences.languages)
    picks = []
    for key in cat.keys:
        t = session.get(Title, key)
        assert t is not None
        if key in cat.embedded and t.genres[0] == genre and t.original_language in langs:
            picks.append(key)
        if len(picks) == count:
            break
    assert len(picks) == count
    for key in picks:
        add_signal(session, user, key, source=InteractionSource.ONBOARDING_SEED)
    return picks


def assert_valid(session: Session, user: User, recs: Recommendations, settings: Settings) -> None:
    """Every item satisfies §8.4 (reference implementation) and FR-22."""
    prefs = user.preferences
    assert prefs is not None
    excluded = excluded_title_keys(session, user.id)
    platforms = {p.provider_id for p in user.platforms}
    allowed = allowed_offer_types(prefs.include_rent_buy)
    keys = [item.key for item in recs.items]
    assert len(keys) == len(set(keys))
    for item in recs.items:
        title = session.get(Title, item.key)
        assert title is not None
        assert passes_hard_filters(
            title,
            prefs,
            country=user.country,
            excluded=excluded,
            adult_certs=settings.adult_certifications.get(user.country, []),
        ), item.key
        assert item.offers
        assert all(o.provider_id in platforms and o.offer_type in allowed for o in item.offers)
        assert item.availability.checked


# --- Acceptance ----------------------------------------------------------------


async def test_works_with_exactly_three_seeds(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session)
    seeds = seed_titles(db_session, user, catalog, genre="drama")
    recs = await recommend(db_session, store, settings, user)

    assert len(recs.items) == 10
    assert not recs.cold_start
    assert recs.debug["profile"]["signals_used"] == 3
    assert [i.rank for i in recs.items] == list(range(1, 11))
    assert not {i.key for i in recs.items} & set(seeds)
    assert_valid(db_session, user, recs, settings)
    assert all(i.reason_source == "template" and i.slot == "core" for i in recs.items)
    assert any(i.reason.startswith("Because you liked Title ") for i in recs.items)
    # Similarity drives the list: the top pick is a drama (the seeds' cluster).
    top = db_session.get(Title, recs.items[0].key)
    assert top is not None
    assert top.genres[0] == "drama"


@pytest.mark.parametrize("trial", range(12))
async def test_every_result_satisfies_hard_filters_and_fr22(
    trial: int,
    db_session: Session,
    store: EmbeddingStore,
    catalog: FixtureCatalog,
    settings: Settings,
) -> None:
    """Property-style: random preferences and feedback; results always pass every filter."""
    rng = random.Random(trial)  # noqa: S311 - test data
    user = make_user(
        db_session,
        languages=rng.sample(LANGUAGES, rng.randint(1, 4)),
        content_types=rng.choice([("movie",), ("tv",), ("movie", "tv")]),
        era=rng.choice(list(Era)),
        avoid_genres=rng.sample(GENRES, rng.randint(0, 3)),
        max_runtime=rng.choice([None, 90, 120, 150]),
        include_rent_buy=rng.random() < 0.5,
        allow_adult=rng.random() < 0.3,
        platforms=rng.sample([NETFLIX, PRIME, TUBI, 2], rng.randint(1, 3)),
    )
    for key in rng.sample(catalog.keys, 8):
        kind = rng.choice([InteractionKind.RATING, InteractionKind.NOT_INTERESTED])
        add_signal(
            db_session, user, key, kind=kind, value=rng.randint(1, 5), days_ago=rng.uniform(0, 400)
        )
    recs = await recommend(db_session, store, settings, user, n=rng.choice([5, 10, 15]))
    assert_valid(db_session, user, recs, settings)
    assert len(recs.items) <= recs.n


async def test_caps_hold_for_multi_language_user(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session, languages=LANGUAGES, platforms=[NETFLIX, PRIME, TUBI])
    seed_titles(db_session, user, catalog, genre="action")
    recs = await recommend(db_session, store, settings, user)

    assert len(recs.items) == 10
    assert recs.debug["relaxations"] == []
    assert recs.debug["caps"] == {"language": 4, "genre": 3}
    titles = [db_session.get(Title, i.key) for i in recs.items]
    langs = Counter(t.original_language for t in titles if t)
    genres = Counter(t.genres[0] for t in titles if t)
    assert max(langs.values()) <= 4
    assert max(genres.values()) <= 3


async def test_single_language_skips_language_cap_and_returns_n(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session, languages=["en"], platforms=[NETFLIX, PRIME, TUBI])
    seed_titles(db_session, user, catalog, genre="comedy")
    recs = await recommend(db_session, store, settings, user)

    assert len(recs.items) == 10
    assert recs.debug["caps"]["language"] is None
    langs = Counter(t.original_language for i in recs.items if (t := db_session.get(Title, i.key)))
    assert langs == {"en": 10}  # far above the multi-language cap of 4
    genres = Counter(t.genres[0] for i in recs.items if (t := db_session.get(Title, i.key)))
    assert max(genres.values()) <= recs.debug["caps"]["genre"]
    assert_valid(db_session, user, recs, settings)


async def test_genre_cap_relaxes_when_few_genres_remain(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    """Only 2 primary genres can survive → 2 x 3 < 10, so the genre cap is raised."""
    keep = {"drama", "comedy"}
    user = make_user(db_session, languages=LANGUAGES, platforms=[NETFLIX, PRIME, TUBI])
    # Exclude every title whose primary genre isn't in `keep`, via not_interested.
    for key in catalog.keys:
        t = db_session.get(Title, key)
        assert t is not None
        if t.genres[0] not in keep:
            add_signal(db_session, user, key, kind=InteractionKind.NOT_INTERESTED, days_ago=500)
    seed_titles(db_session, user, catalog, genre="drama", count=1)
    recs = await recommend(db_session, store, settings, user)

    assert len(recs.items) == 10
    assert recs.debug["relaxations"]
    assert recs.debug["relaxations"][0]["cap"] == "genre"
    assert_valid(db_session, user, recs, settings)


async def test_cold_start_when_profile_is_zero(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session)
    for key in [k for k in catalog.keys if k in catalog.embedded][:3]:
        add_signal(db_session, user, key, value=3)  # weight 0 → zero profile
    recs = await recommend(db_session, store, settings, user)

    assert recs.cold_start
    assert recs.debug["cold_start"] is True
    assert len(recs.items) == 10
    assert all(i.pool == "cold_start" for i in recs.items)
    assert all(i.reason.startswith("Popular in ") for i in recs.items)
    assert any(" on Netflix" in i.reason or " on Amazon" in i.reason for i in recs.items)
    assert_valid(db_session, user, recs, settings)


async def test_cold_start_without_any_signal(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session)
    recs = await recommend(db_session, store, settings, user)
    assert recs.cold_start
    assert len(recs.items) == 10


async def test_excluded_titles_never_recommended(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session)
    seed_titles(db_session, user, catalog, genre="thriller")
    first = await recommend(db_session, store, settings, user)
    a, b, c = (i.key for i in first.items[:3])
    add_signal(db_session, user, a, kind=InteractionKind.NOT_INTERESTED, days_ago=0)
    add_signal(db_session, user, b, value=2, days_ago=0)
    add_signal(db_session, user, c, kind=InteractionKind.SEEN, days_ago=0)

    second = await recommend(db_session, store, settings, user)
    assert {a, b, c}.isdisjoint(i.key for i in second.items)
    assert second.debug["counts"]["excluded"] == 6
    assert len(second.items) == 10
    assert [i.key for i in first.items] != [i.key for i in second.items]


async def test_runtime_appended_vectors_are_used(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    """Titles added + embedded at runtime (OnDemandEmbedder) count as signal and candidate."""
    cache = CatalogCache()
    user = make_user(db_session, languages=["en"], content_types=["movie"])
    await recommend(db_session, store, settings, user, cache=cache)  # warm the cache
    assert cache.builds == 1

    direction = np.zeros(DIM, dtype=np.float32)
    direction[0] = 1.0  # a direction no fixture vector is aligned with
    new_seed, new_pick = (90001, M), (90002, M)
    for tmdb_id, _ in (new_seed, new_pick):
        db_session.add(
            Title(
                tmdb_id=tmdb_id,
                type=M,
                title=f"New {tmdb_id}",
                original_title=f"New {tmdb_id}",
                original_language="en",
                genres=["mystery"],
                release_date=date(2026, 10, 1),
                runtime=100,
                popularity=0.5,
            )
        )
        db_session.flush()
        db_session.add(
            TitleAvailability(
                tmdb_id=tmdb_id,
                type=M,
                country="US",
                provider_id=NETFLIX,
                offer_type=OfferType.FLATRATE,
                link="https://tmdb.test/watch",
            )
        )
    db_session.commit()
    assert store.append(new_seed, direction)
    assert store.append(new_pick, direction)
    add_signal(db_session, user, new_seed, days_ago=0)

    recs = await recommend(db_session, store, settings, user, cache=cache)
    assert cache.builds == 2  # catalog snapshot rebuilt after the insert
    assert recs.debug["profile"]["signals_used"] == 1
    assert recs.items[0].key == new_pick
    assert recs.items[0].pool == "similar"
    assert recs.items[0].reason == "Because you liked New 90001"
    assert new_seed not in {i.key for i in recs.items}


async def test_availability_fetches_respect_budget(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    db_session.execute(delete(TitleAvailability))
    db_session.commit()
    user = make_user(db_session)
    seed_titles(db_session, user, catalog, genre="romance")
    fake = FakeTMDB()

    small = settings.model_copy(update={"availability_max_fetches_per_request": 5})
    recs = await recommend(db_session, store, small, user, tmdb=fake)
    assert len(fake.calls) == 5
    assert len(recs.items) == 5  # only the fetched titles are known to be streamable
    assert_valid(db_session, user, recs, small)

    fake.calls.clear()
    recs = await recommend(db_session, store, settings, user, tmdb=fake)
    assert len(recs.items) == 10
    assert len(fake.calls) <= settings.availability_max_fetches_per_request


async def test_no_platforms_returns_nothing(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session, platforms=[])
    recs = await recommend(db_session, store, settings, user)
    assert recs.items == []
    assert recs.debug["counts"]["platforms"] == 0


async def test_missing_preferences_and_unknown_user(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session, with_preferences=False)
    with pytest.raises(PreferencesMissingError):
        await recommend(db_session, store, settings, user)
    ghost = User(id=uuid.uuid4())
    with pytest.raises(UserNotFoundError):
        await recommend(db_session, store, settings, ghost)
    with pytest.raises(ValueError, match="n must be"):
        await service(db_session, store, settings).recommend(user.id, -1)


async def test_works_without_embedding_index(
    db_session: Session, catalog: FixtureCatalog, settings: Settings, tmp_path: Path
) -> None:
    """No index on disk: profile can't be built → cold start still returns N."""
    user = make_user(db_session)
    seed_titles(db_session, user, catalog, genre="drama")
    empty = EmbeddingStore(tmp_path / "missing")
    recs = await recommend(db_session, empty, settings, user)
    assert recs.cold_start
    assert recs.debug["profile"]["signals_unembedded"] == 3
    assert len(recs.items) == 10
    assert recs.model_version is None


async def test_debug_output_config_hash_and_determinism(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session)
    seed_titles(db_session, user, catalog, genre="horror")
    a = await recommend(db_session, store, settings, user)
    b = await recommend(db_session, store, settings, user)

    assert [i.key for i in a.items] == [i.key for i in b.items]
    assert a.config_hash == b.config_hash == config_hash(settings)
    assert len(a.config_hash) == 16
    assert a.model_version == "fake/e5@v1"
    assert a.generated_at == NOW
    assert set(a.items[0].components) == {"sim", "quality", "pop", "recency", "neg", "score"}
    counts = a.debug["counts"]
    assert counts["catalog"] == 200
    assert counts["embedded"] == 180
    assert counts["candidates"] >= counts["available"] >= 10
    assert set(a.debug["timings_ms"]) >= {"catalog", "profile", "filters", "availability", "total"}
    assert a.debug["weights"]["w_qual"] == 0.0  # D-017
    changed = settings.model_copy(update={"w_rec": 0.2})
    assert config_hash(changed) != a.config_hash


async def test_negative_profile_penalty(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session)
    seed_titles(db_session, user, catalog, genre="drama")
    for key in [k for k in catalog.keys if k in catalog.embedded][-3:]:
        add_signal(db_session, user, key, value=1)
    with_neg = settings.model_copy(update={"w_neg": 0.3})
    recs = await recommend(db_session, store, with_neg, user)
    assert len(recs.items) == 10
    assert any(i.components["neg"] > 0 for i in recs.items)


async def test_small_n_scales_caps(
    db_session: Session, store: EmbeddingStore, catalog: FixtureCatalog, settings: Settings
) -> None:
    user = make_user(db_session, languages=LANGUAGES)
    seed_titles(db_session, user, catalog, genre="animation")
    recs = await recommend(db_session, store, settings, user, n=3)
    assert len(recs.items) == 3
    caps = recs.debug["caps"]
    # Scaled caps for N = 3 are 2 / 1; relaxations (recorded) may only raise them.
    assert caps["language"] >= 2
    assert caps["genre"] >= 1
    assert len(recs.debug["relaxations"]) == (caps["language"] - 2) + (caps["genre"] - 1)
    titles = [t for i in recs.items if (t := db_session.get(Title, i.key))]
    assert max(Counter(t.genres[0] for t in titles).values()) <= caps["genre"]
    assert max(Counter(t.original_language for t in titles).values()) <= caps["language"]
