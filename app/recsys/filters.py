"""Hard filters as a boolean mask over the catalog (§8.4, FR-21, FR-23, D-003).

The mask is built **before** similarity top-k, so narrow preferences never starve the
candidate set. Availability (FR-22) is the one hard filter applied after scoring
(§8.6), because it may need network calls; repeat suppression (FR-24) is Phase 2.

Values that cannot be checked fail closed: a title without a release date is excluded
by every era except `any`, and a movie without a runtime is excluded when the user set
`max_runtime`. Every recommended title then provably satisfies the filters.
"""

from __future__ import annotations

from collections.abc import Collection, Set
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from app.db.models import Era, TitleType
from app.ingest.build_embeddings import TitleKey
from app.recsys.catalog import TYPE_CODES, BoolArray, Catalog

CLASSICS_BEFORE = date(2000, 1, 1)


@dataclass(frozen=True, slots=True)
class UserFilters:
    languages: Collection[str]
    content_types: Collection[TitleType]
    era: Era = Era.ANY
    avoid_genres: Collection[str] = ()
    max_runtime: int | None = None
    allow_adult: bool = False
    # Certifications hidden in the user's country unless `allow_adult` (§13 Q2).
    adult_certifications: Collection[str] = ()
    country: str = "US"
    # FR-23: rated / seen / not interested.
    excluded: Set[TitleKey] = field(default_factory=frozenset)
    max_runtime_applies_to_tv: bool = False


def years_before(day: date, years: int) -> date:
    """Same calendar day `years` earlier (Feb 29 → Feb 28)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def era_bounds(era: Era, today: date) -> tuple[date | None, date | None]:
    """(earliest allowed release, release must be before) for an era; None = open."""
    if era is Era.LAST_2Y:
        return years_before(today, 2), None
    if era is Era.LAST_10Y:
        return years_before(today, 10), None
    if era is Era.CLASSICS_PRE_2000:
        return None, CLASSICS_BEFORE
    return None, None


def filter_mask(catalog: Catalog, f: UserFilters, today: date) -> BoolArray:
    """True for catalog rows that pass every §8.4 filter except availability."""
    n = len(catalog)
    if n == 0:
        return np.zeros(0, dtype=np.bool_)

    lang_codes = [catalog.lang_vocab[lang] for lang in f.languages if lang in catalog.lang_vocab]
    mask = np.isin(catalog.lang_code, np.array(lang_codes, dtype=np.int64))

    type_codes = [TYPE_CODES[t] for t in f.content_types]
    mask &= np.isin(catalog.type_code, np.array(type_codes, dtype=np.int8))

    earliest, before = era_bounds(f.era, today)
    if earliest is not None:
        mask &= catalog.has_release & (catalog.release_ordinal >= earliest.toordinal())
    if before is not None:
        mask &= catalog.has_release & (catalog.release_ordinal < before.toordinal())

    for genre in f.avoid_genres:
        rows = catalog.genre_rows.get(genre)
        if rows is not None:
            mask[rows] = False

    if f.max_runtime is not None:
        fits = catalog.runtime <= f.max_runtime  # NaN compares False → unknown excluded
        applies = (
            np.ones(n, dtype=np.bool_)
            if f.max_runtime_applies_to_tv
            else catalog.type_code == TYPE_CODES[TitleType.MOVIE]
        )
        mask &= fits | ~applies

    if not f.allow_adult:
        mask &= ~catalog.adult
        hidden = set(f.adult_certifications)
        if hidden:
            certs = catalog.certification_for(f.country)
            mask &= np.array([c not in hidden for c in certs], dtype=np.bool_)

    for key in f.excluded:
        row = catalog.index.get(key)
        if row is not None:
            mask[row] = False
    return mask
