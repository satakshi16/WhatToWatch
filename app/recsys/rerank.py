"""Diversity re-rank: MMR with per-language / per-primary-genre caps (§8.7, FR-25).

Greedy MMR picks the next item maximizing `λ·score - (1-λ)·max_cos(item, selected)`,
among items that keep the caps (≤ `CAP_PER_LANGUAGE` per language, ≤ `CAP_PER_GENRE`
per primary genre for N = 10, scaled for other N). Titles without a vector have a zero
row, so their `max_cos` is 0; titles without a genre are not genre-capped.

Cap relaxation: the language cap is skipped when the user picked only one language.
When no remaining item fits the caps, the genre cap is raised by 1, then (if still none
fits) the language cap, and so on until N items are picked or candidates run out.
Every change is recorded for the debug output.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from app.ingest.build_embeddings import Vectors
from app.recsys.catalog import FloatArray

CAP_BASE_N = 10  # the spec's caps are given for N = 10


def scaled_cap(cap: int, n: int) -> int:
    """Cap for list size `n`, proportional to the N = 10 value (at least 1)."""
    return max(1, math.ceil(cap * n / CAP_BASE_N))


@dataclass
class RerankResult:
    order: list[int]  # indices into the candidate list, in final rank order
    language_cap: int | None  # final cap (None = skipped, single-language user)
    genre_cap: int
    relaxations: list[dict[str, int | str]] = field(default_factory=list)


def mmr_rerank(
    scores: FloatArray,
    vectors: Vectors,
    languages: Sequence[str],
    primary_genres: Sequence[str | None],
    *,
    n: int,
    lam: float,
    cap_per_language: int,
    cap_per_genre: int,
    single_language: bool,
) -> RerankResult:
    m = scores.shape[0]
    lang_cap: int | None = None if single_language else scaled_cap(cap_per_language, n)
    genre_cap = scaled_cap(cap_per_genre, n)
    result = RerankResult(order=[], language_cap=lang_cap, genre_cap=genre_cap)
    if m == 0 or n <= 0:
        return result

    remaining = np.ones(m, dtype=np.bool_)
    max_sim = np.zeros(m, dtype=np.float64)
    lang_count: Counter[str] = Counter()
    genre_count: Counter[str] = Counter()
    lang_arr = list(languages)
    genre_arr = list(primary_genres)

    def fits(i: int) -> bool:
        if lang_cap is not None and lang_count[lang_arr[i]] >= lang_cap:
            return False
        g = genre_arr[i]
        return g is None or genre_count[g] < genre_cap

    while len(result.order) < n and remaining.any():
        eligible = np.array([remaining[i] and fits(i) for i in range(m)], dtype=np.bool_)
        if not eligible.any():
            # Relax: genre cap first; then the language cap if the genre bump didn't help.
            genre_cap += 1
            result.relaxations.append({"cap": "genre", "value": genre_cap})
            if lang_cap is not None and not any(remaining[i] and fits(i) for i in range(m)):
                lang_cap += 1
                result.relaxations.append({"cap": "language", "value": lang_cap})
            continue
        mmr = lam * scores - (1.0 - lam) * max_sim
        mmr = np.where(eligible, mmr, -np.inf)
        # Ties: higher raw score, then lower index (np.lexsort sorts by last key first).
        best = int(np.lexsort((np.arange(m), -scores, -mmr))[0])
        result.order.append(best)
        remaining[best] = False
        lang_count[lang_arr[best]] += 1
        g = genre_arr[best]
        if g is not None:
            genre_count[g] += 1
        max_sim = np.maximum(max_sim, vectors @ vectors[best])

    result.language_cap, result.genre_cap = lang_cap, genre_cap
    return result
