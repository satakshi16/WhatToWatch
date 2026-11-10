"""§8.4 hard-filter mask: each rule, fail-closed unknowns, and a seeded property test
comparing the vectorized mask against an independent scalar predicate."""

from __future__ import annotations

import random
from datetime import date

import pytest

from app.db.models import Era, TitleType
from app.recsys.catalog import Catalog, TitleRecord
from app.recsys.filters import UserFilters, era_bounds, filter_mask, years_before

M, T = TitleType.MOVIE, TitleType.TV
TODAY = date(2026, 11, 5)
GENRES = ["drama", "comedy", "action", "horror", "anime"]
LANGS = ["en", "hi", "ko", "ja"]


def rec(
    tmdb_id: int,
    type_: TitleType = M,
    *,
    lang: str = "en",
    genres: tuple[str, ...] = ("drama",),
    release: date | None = date(2020, 1, 1),
    runtime: int | None = 100,
    adult: bool = False,
    cert: str | None = None,
) -> TitleRecord:
    return TitleRecord(
        tmdb_id=tmdb_id,
        type=type_,
        title=f"T{tmdb_id}",
        original_language=lang,
        genres=genres,
        release_date=release,
        runtime=runtime,
        vote_avg=7.0,
        vote_count=100,
        popularity=1.0,
        adult=adult,
        certification={"US": cert} if cert else {},
    )


def base(**kw: object) -> UserFilters:
    args: dict[str, object] = {"languages": ["en"], "content_types": [M, T]}
    args.update(kw)
    return UserFilters(**args)  # type: ignore[arg-type]


def passing(records: list[TitleRecord], f: UserFilters) -> list[int]:
    cat = Catalog.from_rows(records)
    return [records[i].tmdb_id for i in filter_mask(cat, f, TODAY).nonzero()[0]]


def test_language_and_type() -> None:
    rows = [rec(1), rec(2, lang="ko"), rec(3, T), rec(4, T, lang="hi")]
    assert passing(rows, base()) == [1, 3]
    assert passing(rows, base(languages=["ko", "hi"], content_types=[T])) == [4]
    assert passing(rows, base(languages=["xx"])) == []


@pytest.mark.parametrize(
    ("era", "expected"),
    [
        (Era.ANY, [1, 2, 3, 4, 5]),
        (Era.LAST_2Y, [3, 4]),  # ≥ 2024-11-05
        (Era.LAST_10Y, [2, 3, 4]),  # ≥ 2016-11-05
        (Era.CLASSICS_PRE_2000, [1]),  # < 2000-01-01; unknown date excluded
    ],
)
def test_era(era: Era, expected: list[int]) -> None:
    rows = [
        rec(1, release=date(1999, 12, 31)),
        rec(2, release=date(2018, 1, 1)),
        rec(3, release=date(2024, 11, 5)),
        rec(4, release=date(2026, 1, 1)),
        rec(5, release=None),
    ]
    assert passing(rows, base(era=era)) == expected


def test_years_before_handles_leap_day() -> None:
    assert years_before(date(2028, 2, 29), 2) == date(2026, 2, 28)
    assert era_bounds(Era.ANY, TODAY) == (None, None)


def test_avoid_genres_any_position() -> None:
    rows = [rec(1, genres=("drama", "horror")), rec(2, genres=("comedy",)), rec(3, genres=())]
    assert passing(rows, base(avoid_genres=["horror", "unknown-genre"])) == [2, 3]


def test_max_runtime_movies_only_by_default() -> None:
    rows = [rec(1, runtime=90), rec(2, runtime=150), rec(3, runtime=None), rec(4, T, runtime=200)]
    assert passing(rows, base(max_runtime=120)) == [1, 4]
    assert passing(rows, base(max_runtime=120, max_runtime_applies_to_tv=True)) == [1]


def test_adult_flag_and_certifications() -> None:
    rows = [rec(1, adult=True), rec(2, cert="NC-17"), rec(3, cert="R"), rec(4)]
    hidden = base(adult_certifications=["NC-17"])
    assert passing(rows, hidden) == [3, 4]
    assert passing(rows, base(adult_certifications=["NC-17"], allow_adult=True)) == [1, 2, 3, 4]
    # Certifications are read for the user's country only.
    assert passing(rows, base(adult_certifications=["NC-17"], country="IN")) == [2, 3, 4]


def test_excluded_keys() -> None:
    rows = [rec(1), rec(2), rec(1, T)]
    assert passing(rows, base(excluded=frozenset({(1, M), (77, M)}))) == [2, 1]


def test_empty_catalog() -> None:
    assert filter_mask(Catalog.from_rows([]), base(), TODAY).shape == (0,)


def _reference(r: TitleRecord, f: UserFilters) -> bool:
    if r.original_language not in f.languages or r.type not in f.content_types:
        return False
    lo, before = era_bounds(f.era, TODAY)
    if lo and (r.release_date is None or r.release_date < lo):
        return False
    if before and (r.release_date is None or r.release_date >= before):
        return False
    if set(r.genres) & set(f.avoid_genres):
        return False
    applies = r.type is M or f.max_runtime_applies_to_tv
    if f.max_runtime is not None and applies and (r.runtime is None or r.runtime > f.max_runtime):
        return False
    if not f.allow_adult and (
        r.adult or r.certification.get(f.country) in set(f.adult_certifications)
    ):
        return False
    return r.key not in f.excluded


@pytest.mark.parametrize("trial", range(40))
def test_mask_matches_scalar_reference(trial: int) -> None:
    rng = random.Random(trial)  # noqa: S311 - test data
    rows = [
        rec(
            i,
            rng.choice([M, T]),
            lang=rng.choice(LANGS),
            genres=tuple(rng.sample(GENRES, rng.randint(0, 3))),
            release=None if rng.random() < 0.1 else date(rng.randint(1960, 2026), 6, 1),
            runtime=None if rng.random() < 0.1 else rng.randint(20, 200),
            adult=rng.random() < 0.05,
            cert=rng.choice([None, "PG", "R", "NC-17", "TV-MA"]),
        )
        for i in range(120)
    ]
    f = UserFilters(
        languages=rng.sample(LANGS, rng.randint(1, 3)),
        content_types=rng.choice([[M], [T], [M, T]]),
        era=rng.choice(list(Era)),
        avoid_genres=rng.sample(GENRES, rng.randint(0, 2)),
        max_runtime=rng.choice([None, 90, 130]),
        allow_adult=rng.random() < 0.3,
        adult_certifications=rng.choice([[], ["NC-17"], ["NC-17", "TV-MA"]]),
        excluded=frozenset(r.key for r in rng.sample(rows, 10)),
        max_runtime_applies_to_tv=rng.random() < 0.5,
    )
    assert passing(rows, f) == [r.tmdb_id for r in rows if _reference(r, f)]
