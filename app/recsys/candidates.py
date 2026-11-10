"""Candidate generation over the filter mask (§8.3).

1. Top `CANDIDATES_SIMILAR` titles by cosine(profile, item) among masked, embedded rows.
2. Union with the top `CANDIDATES_TRENDING` by `popularity` among masked rows. The mask
   already holds the language filter, so these are "trending in the user's languages".
   Titles without a vector (§13 Q7) can only enter through this pool.
Cold start (no profile): the top `CANDIDATES_SIMILAR` masked rows by the cold-start
score (`scorer.cold_start_scores`) replace the similarity pool.

Ties break by catalog row, so results are deterministic.
"""

from __future__ import annotations

import numpy as np

from app.ingest.build_embeddings import Vectors
from app.recsys.catalog import NO_ROW, BoolArray, FloatArray, IntArray


def top_k(values: FloatArray, eligible: BoolArray, k: int) -> IntArray:
    """Indices of the `k` largest `values` where `eligible`, best first (ties → lower index)."""
    rows = np.flatnonzero(eligible)
    if k <= 0 or rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    vals = values[rows]
    if rows.size > k:
        # Partition on (-value) then sort only the k winners; lexsort keeps ties stable.
        part = np.argpartition(-vals, k - 1)[:k]
        cutoff = vals[part].min()
        # Include every row tied at the cutoff before the stable sort, then trim.
        part = np.flatnonzero(vals >= cutoff)
        rows, vals = rows[part], vals[part]
    order = np.lexsort((rows, -vals))[:k]
    out: IntArray = rows[order].astype(np.int64)
    return out


def item_similarity(matrix: Vectors, vector_rows: IntArray, profile: Vectors) -> FloatArray:
    """Cosine(profile, item) per catalog row (vectors are L2-normalized); NaN = no vector."""
    sims = np.full(vector_rows.shape[0], np.nan, dtype=np.float64)
    has = vector_rows != NO_ROW
    if has.any():
        sims[has] = matrix[vector_rows[has]] @ profile
    return sims


def similar_candidates(sims: FloatArray, mask: BoolArray, k: int) -> IntArray:
    return top_k(np.nan_to_num(sims, nan=-np.inf), mask & ~np.isnan(sims), k)


def trending_candidates(popularity: FloatArray, mask: BoolArray, k: int) -> IntArray:
    return top_k(popularity, mask, k)


def union(*pools: IntArray) -> IntArray:
    """Rows from every pool, first occurrence order kept."""
    seen: dict[int, None] = {}
    for pool in pools:
        for row in pool.tolist():
            seen.setdefault(int(row), None)
    return np.array(list(seen), dtype=np.int64)
