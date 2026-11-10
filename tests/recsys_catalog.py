"""Deterministic fixture catalog for recommender tests (§5.6).

`build_fixture_catalog` writes ~200 titles across 6 languages, both types, 8 primary
genres, with fake 16-d embeddings clustered by primary genre (plus a smaller language
component), warm US availability for a few providers, and the provider list. Every 10th
title has no vector (stands in for titles without open content, §13 Q7).

Also: `make_user` (user + preferences + platforms + interactions straight into the DB),
and `passes_hard_filters`, an independent scalar re-implementation of §8.4 used to check
recommender output.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import (
    NO_PROVIDER_ID,
    Era,
    Interaction,
    InteractionKind,
    InteractionSource,
    OfferType,
    OnboardingStep,
    Provider,
    Title,
    TitleAvailability,
    TitleType,
    User,
    UserPlatform,
    UserPreferences,
)
from app.ingest.build_embeddings import EmbeddingIndex, TitleKey, key_str, save_index

M, T = TitleType.MOVIE, TitleType.TV
NOW = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
TODAY = NOW.date()
DIM = 16
LANGUAGES = ("en", "hi", "ko", "ja", "es", "fr")
GENRES = (
    "drama",
    "comedy",
    "action",
    "thriller",
    "romance",
    "science-fiction",
    "horror",
    "animation",
)
NETFLIX, PRIME, APPLE, TUBI = 8, 9, 2, 73
PROVIDER_NAMES = {NETFLIX: "Netflix", PRIME: "Amazon Prime Video", APPLE: "Apple TV", TUBI: "Tubi"}
N_TITLES = 200


@dataclass(frozen=True)
class FixtureCatalog:
    keys: list[TitleKey]
    embedded: set[TitleKey]
    vectors: dict[TitleKey, np.ndarray]

    def of_genre(self, genre: str, session: Session) -> list[TitleKey]:
        return [k for k in self.keys if (t := session.get(Title, k)) and t.genres[:1] == [genre]]


def _unit(v: np.ndarray) -> np.ndarray:
    out: np.ndarray = (v / np.linalg.norm(v)).astype(np.float32)
    return out


def build_fixture_catalog(
    session: Session, embeddings_dir: Path, *, n_titles: int = N_TITLES, seed: int = 7
) -> FixtureCatalog:
    rng = np.random.default_rng(seed)
    genre_dirs = {g: _unit(rng.normal(size=DIM)) for g in GENRES}
    lang_dirs = {lang: _unit(rng.normal(size=DIM)) for lang in LANGUAGES}
    fetched = utcnow()
    titles: list[Title] = []
    avail: list[TitleAvailability] = []
    ids: list[str] = []
    rows: list[np.ndarray] = []
    vectors: dict[TitleKey, np.ndarray] = {}
    keys: list[TitleKey] = []
    for i in range(n_titles):
        tmdb_id = 1000 + i
        type_ = M if i % 3 else T
        lang = LANGUAGES[i % len(LANGUAGES)]
        primary = GENRES[(i // len(LANGUAGES)) % len(GENRES)]
        secondary = GENRES[(i * 5 + 3) % len(GENRES)]
        genres = [primary] if secondary == primary else [primary, secondary]
        year = 1985 + (i * 7) % 42  # 1985..2026
        release = None if i % 37 == 5 else date(year, 1 + i % 12, 1 + i % 28)
        if release is not None and release > TODAY:
            release = date(2026, 6, 1)
        runtime = None if i % 41 == 7 else (80 + (i * 13) % 100 if type_ is M else 20 + i % 40)
        cert = ["PG-13", "R", "NC-17", "PG", "TV-MA", "TV-14"][i % 6] if i % 4 else None
        titles.append(
            Title(
                tmdb_id=tmdb_id,
                type=type_,
                title=f"Title {i}",
                original_title=f"Original {i}",
                original_language=lang,
                genres=genres,
                release_date=release,
                runtime=runtime,
                vote_avg=round(float(rng.uniform(4, 9)), 1),
                vote_count=int(rng.integers(0, 5000)),
                popularity=round(float(rng.uniform(1, 500)), 2),
                adult=i % 53 == 11,
                certification={"US": cert} if cert else {},
            )
        )
        key = (tmdb_id, type_)
        keys.append(key)
        if i % 10 != 9:
            v = _unit(genre_dirs[primary] + 0.35 * lang_dirs[lang] + 0.25 * rng.normal(size=DIM))
            ids.append(key_str(key))
            rows.append(v)
            vectors[key] = v
        # Warm US availability: Netflix ~1/2, Prime ~1/3, Apple rent, else marker row.
        offers: list[tuple[int, OfferType]] = []
        if i % 2 == 0:
            offers.append((NETFLIX, OfferType.FLATRATE))
        if i % 3 == 0:
            offers.append((PRIME, OfferType.FLATRATE))
        if i % 5 == 1:
            offers.append((APPLE, OfferType.RENT))
        if i % 7 == 3:
            offers.append((TUBI, OfferType.ADS))
        if not offers:
            offers.append((NO_PROVIDER_ID, OfferType.NONE))
        link = f"https://www.themoviedb.org/{type_.value}/{tmdb_id}/watch?locale=US"
        avail.extend(
            TitleAvailability(
                tmdb_id=tmdb_id,
                type=type_,
                country="US",
                provider_id=pid,
                offer_type=ot,
                link=link,
                fetched_at=fetched,
            )
            for pid, ot in offers
        )
    session.add_all(titles)
    session.flush()
    session.add_all(avail)
    session.add_all(
        Provider(
            provider_id=pid, country="US", name=name, logo_path=f"/{pid}.png", display_priority=pid
        )
        for pid, name in PROVIDER_NAMES.items()
    )
    session.commit()
    save_index(
        embeddings_dir,
        EmbeddingIndex(
            ids=ids,
            vectors=np.stack(rows).astype(np.float32),
            meta={"model": "fake/e5", "model_version": "v1", "dim": DIM},
        ),
    )
    return FixtureCatalog(keys=keys, embedded=set(vectors), vectors=vectors)


def make_user(
    session: Session,
    *,
    languages: Sequence[str] = ("en", "hi", "ko"),
    content_types: Sequence[str] = ("movie", "tv"),
    era: Era = Era.ANY,
    avoid_genres: Sequence[str] = (),
    max_runtime: int | None = None,
    include_rent_buy: bool = False,
    allow_adult: bool = False,
    platforms: Iterable[int] = (NETFLIX, PRIME),
    country: str = "US",
    with_preferences: bool = True,
) -> User:
    user = User(
        email=f"{uuid.uuid4().hex[:10]}@example.com",
        password_hash="x",
        display_name="Tester",
        country=country,
    )
    session.add(user)
    session.flush()
    if with_preferences:
        session.add(
            UserPreferences(
                user_id=user.id,
                languages=list(languages),
                content_types=list(content_types),
                era=era,
                avoid_genres=list(avoid_genres),
                max_runtime=max_runtime,
                include_rent_buy=include_rent_buy,
                allow_adult=allow_adult,
                onboarding_step=OnboardingStep.DONE,
            )
        )
        session.add_all(
            UserPlatform(user_id=user.id, provider_id=p, country=country) for p in platforms
        )
    session.commit()
    session.refresh(user)
    return user


def add_signal(
    session: Session,
    user: User,
    key: TitleKey,
    *,
    kind: InteractionKind = InteractionKind.RATING,
    value: float | None = 5,
    days_ago: float = 1.0,
    source: InteractionSource = InteractionSource.BROWSE,
) -> None:
    session.add(
        Interaction(
            user_id=user.id,
            tmdb_id=key[0],
            type=key[1],
            kind=kind,
            value=value if kind is InteractionKind.RATING else None,
            source=source,
            created_at=NOW - timedelta(days=days_ago),
        )
    )
    session.commit()


def passes_hard_filters(
    title: Title,
    prefs: UserPreferences,
    *,
    country: str,
    excluded: set[TitleKey],
    adult_certs: Sequence[str] = ("NC-17",),
    today: date = TODAY,
) -> bool:
    """Scalar §8.4 reference (except availability), written independently of filters.py."""
    if title.original_language not in prefs.languages:
        return False
    if title.type.value not in prefs.content_types:
        return False
    rd = title.release_date
    if prefs.era is Era.LAST_2Y and (rd is None or rd < today.replace(year=today.year - 2)):
        return False
    if prefs.era is Era.LAST_10Y and (rd is None or rd < today.replace(year=today.year - 10)):
        return False
    if prefs.era is Era.CLASSICS_PRE_2000 and (rd is None or rd >= date(2000, 1, 1)):
        return False
    if set(title.genres) & set(prefs.avoid_genres):
        return False
    if (
        prefs.max_runtime is not None
        and title.type is M
        and (title.runtime is None or title.runtime > prefs.max_runtime)
    ):
        return False
    if not prefs.allow_adult and (
        title.adult or (title.certification or {}).get(country) in adult_certs
    ):
        return False
    return (title.tmdb_id, title.type) not in excluded
