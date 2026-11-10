"""Column-oriented, in-memory snapshot of the `titles` table for the recommender (§8).

`Catalog.from_rows` is pure: `service.py` selects the rows and builds the snapshot once,
then reuses it until the table changes. Every per-title attribute the filters, scorer
and re-ranker need is a NumPy array indexed by catalog row, so a request is a handful of
vectorized operations over ~15k rows instead of a Python loop.

TMDB fields used here (language, type, dates, runtime, genres, adult, certification,
popularity, votes) serve display, hard filters and the non-personalized trending /
cold-start order only; none of them reach the embedding vectors (D-011, D-017).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt

from app.db.models import TitleType
from app.ingest.build_embeddings import TitleKey, key_str

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

TYPE_CODES: dict[TitleType, int] = {TitleType.MOVIE: 0, TitleType.TV: 1}
NO_ROW = -1  # `vector_rows` value for titles without an embedding


@dataclass(frozen=True, slots=True)
class TitleRecord:
    """One `titles` row, reduced to what the recommender reads."""

    tmdb_id: int
    type: TitleType
    title: str
    original_language: str
    genres: Sequence[str]
    release_date: date | None
    runtime: int | None
    vote_avg: float | None
    vote_count: int
    popularity: float
    adult: bool
    certification: dict[str, Any]

    @property
    def key(self) -> TitleKey:
        return self.tmdb_id, self.type


@dataclass(eq=False)
class Catalog:
    """Arrays over catalog rows (row i ↔ `keys[i]`). Treat as immutable."""

    keys: list[TitleKey]
    titles: list[str]
    languages: list[str]
    genres: list[tuple[str, ...]]
    type_code: npt.NDArray[np.int8]
    lang_code: IntArray
    lang_vocab: dict[str, int]
    # Days since 0001-01-01 (`date.toordinal`); `has_release` marks known dates.
    release_ordinal: IntArray
    has_release: BoolArray
    runtime: FloatArray  # NaN = unknown
    vote_avg: FloatArray  # NaN = no votes
    vote_count: FloatArray
    popularity: FloatArray
    adult: BoolArray
    certifications: list[dict[str, Any]]
    # Canonical genre slug → rows having it (any position).
    genre_rows: dict[str, IntArray]
    index: dict[TitleKey, int] = field(default_factory=dict)
    _cert_cache: dict[str, list[str | None]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[TitleRecord]) -> Catalog:
        records = list(rows)
        n = len(records)
        lang_vocab: dict[str, int] = {}
        genre_lists: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            lang_vocab.setdefault(r.original_language, len(lang_vocab))
            for g in dict.fromkeys(r.genres):
                genre_lists.setdefault(g, []).append(i)
        release = [r.release_date for r in records]
        return cls(
            keys=[r.key for r in records],
            titles=[r.title for r in records],
            languages=[r.original_language for r in records],
            genres=[tuple(r.genres) for r in records],
            type_code=np.array([TYPE_CODES[r.type] for r in records], dtype=np.int8),
            lang_code=np.array([lang_vocab[r.original_language] for r in records], dtype=np.int64),
            lang_vocab=lang_vocab,
            release_ordinal=np.array([d.toordinal() if d else 0 for d in release], dtype=np.int64),
            has_release=np.array([d is not None for d in release], dtype=np.bool_),
            runtime=_floats(r.runtime for r in records),
            vote_avg=_floats(r.vote_avg if r.vote_count > 0 else None for r in records),
            vote_count=np.array([max(r.vote_count, 0) for r in records], dtype=np.float64),
            popularity=np.array([max(r.popularity, 0.0) for r in records], dtype=np.float64),
            adult=np.array([r.adult for r in records], dtype=np.bool_),
            certifications=[dict(r.certification or {}) for r in records],
            genre_rows={g: np.array(ix, dtype=np.int64) for g, ix in genre_lists.items()},
            index={r.key: i for i, r in enumerate(records)} if n else {},
        )

    def __len__(self) -> int:
        return len(self.keys)

    def primary_genre(self, row: int) -> str | None:
        genres = self.genres[row]
        return genres[0] if genres else None

    def certification_for(self, country: str) -> list[str | None]:
        """Per-row certification in `country` (None if unknown), cached per country."""
        country = country.upper()
        cached = self._cert_cache.get(country)
        if cached is None:
            cached = [_cert(c.get(country)) for c in self.certifications]
            self._cert_cache[country] = cached
        return cached

    def vector_rows(self, embedding_ids: Sequence[str]) -> IntArray:
        """Row of each catalog title in an embedding matrix with `embedding_ids` (or NO_ROW)."""
        position = {name: i for i, name in enumerate(embedding_ids)}
        return np.array([position.get(key_str(k), NO_ROW) for k in self.keys], dtype=np.int64)


def _floats(values: Iterable[float | int | None]) -> FloatArray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _cert(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
