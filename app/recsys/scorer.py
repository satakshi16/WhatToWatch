"""Weighted Phase 1 scorer (§8.5); the baseline and fallback once LTR lands (P3-03).

    score = W_SIM·sim + W_QUAL·quality + W_POP·pop + W_REC·recency - W_NEG·neg

- `sim`: cosine to the profile, min-max normalized within the candidate set; titles
  without a vector get 0.
- `quality`: Bayesian average / 10, `(n/(n+m))·R + (m/(n+m))·C`, C = catalog mean per type.
- `pop`: `log1p(popularity) / log1p(max popularity in catalog)`.
- `recency`: `0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)`; unknown date → 0.
- `neg`: cosine to the negative profile, clipped at 0 (off by default, `W_NEG = 0`).

`quality` and `pop` come from TMDB; D-017 sets `W_QUAL = W_POP = 0` until TMDB grants
permission, so by default they only drive the non-personalized cold-start order
(`cold_start_scores`) and the trending pool, never the personalized score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from app.recsys.catalog import Catalog, FloatArray, IntArray

# Cold start (§8.2): "rank by quality + popularity", equal weights, recency breaks ties.
COLD_START_W_QUAL = 0.5
COLD_START_W_POP = 0.5
COLD_START_W_REC = 0.01
_RANGE_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class Weights:
    sim: float
    qual: float
    pop: float
    rec: float
    neg: float = 0.0


@dataclass(frozen=True, slots=True)
class Components:
    """Per-candidate score parts, aligned with the candidate rows passed to `score`."""

    rows: IntArray
    sim: FloatArray
    quality: FloatArray
    pop: FloatArray
    recency: FloatArray
    neg: FloatArray
    score: FloatArray

    def of(self, i: int) -> dict[str, float]:
        return {
            "sim": float(self.sim[i]),
            "quality": float(self.quality[i]),
            "pop": float(self.pop[i]),
            "recency": float(self.recency[i]),
            "neg": float(self.neg[i]),
            "score": float(self.score[i]),
        }


def bayesian_quality(catalog: Catalog, m: float) -> FloatArray:
    """Bayesian average rating / 10 per row; C is the mean `vote_avg` per type."""
    n = catalog.vote_count
    r = np.nan_to_num(catalog.vote_avg, nan=0.0)
    c = np.zeros(len(catalog), dtype=np.float64)
    for code in np.unique(catalog.type_code):
        of_type = catalog.type_code == code
        rated = of_type & ~np.isnan(catalog.vote_avg)
        c[of_type] = float(catalog.vote_avg[rated].mean()) if rated.any() else 0.0
    denom = n + m
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(denom > 0, (n / denom) * r + (m / denom) * c, c)
    return np.clip(q / 10.0, 0.0, 1.0)


def popularity_score(catalog: Catalog) -> FloatArray:
    if len(catalog) == 0:
        return np.zeros(0, dtype=np.float64)
    top = float(np.log1p(catalog.popularity.max()))
    if top <= 0:
        return np.zeros(len(catalog), dtype=np.float64)
    return np.log1p(catalog.popularity) / top


def recency_score(catalog: Catalog, today: date, half_life_days: float) -> FloatArray:
    age = np.maximum(today.toordinal() - catalog.release_ordinal, 0).astype(np.float64)
    return np.where(catalog.has_release, 0.5 ** (age / half_life_days), 0.0)


def min_max(values: FloatArray) -> FloatArray:
    """Min-max normalize, NaN → 0. A constant (non-NaN) set maps to 1."""
    out = np.zeros(values.shape[0], dtype=np.float64)
    known = ~np.isnan(values)
    if not known.any():
        return out
    lo, hi = float(values[known].min()), float(values[known].max())
    span = hi - lo
    out[known] = 1.0 if span < _RANGE_EPS else (values[known] - lo) / span
    return out


def score(
    rows: IntArray,
    *,
    sims: FloatArray,
    neg_sims: FloatArray | None,
    quality: FloatArray,
    pop: FloatArray,
    recency: FloatArray,
    weights: Weights,
) -> Components:
    """Score candidate `rows`; `sims` / `neg_sims` / `quality` / … are per catalog row."""
    sim = min_max(sims[rows])
    neg = (
        np.clip(np.nan_to_num(neg_sims[rows], nan=0.0), 0.0, None)
        if neg_sims is not None
        else np.zeros(rows.shape[0], dtype=np.float64)
    )
    q, p, r = quality[rows], pop[rows], recency[rows]
    total = weights.sim * sim + weights.qual * q + weights.pop * p + weights.rec * r
    total = total - weights.neg * neg
    return Components(rows=rows, sim=sim, quality=q, pop=p, recency=r, neg=neg, score=total)


def cold_start_scores(quality: FloatArray, pop: FloatArray, recency: FloatArray) -> FloatArray:
    """Non-personalized order for users without a profile (per catalog row)."""
    out: FloatArray = (
        COLD_START_W_QUAL * quality + COLD_START_W_POP * pop + COLD_START_W_REC * recency
    )
    return out


def cold_start_components(
    rows: IntArray, *, quality: FloatArray, pop: FloatArray, recency: FloatArray
) -> Components:
    zeros = np.zeros(rows.shape[0], dtype=np.float64)
    q, p, r = quality[rows], pop[rows], recency[rows]
    return Components(
        rows=rows,
        sim=zeros,
        quality=q,
        pop=p,
        recency=r,
        neg=zeros.copy(),
        score=cold_start_scores(q, p, r),
    )
