"""§8.3 candidates and §8.5 scorer math; catalog snapshot helpers."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from app.db.models import TitleType
from app.recsys.candidates import (
    item_similarity,
    similar_candidates,
    top_k,
    trending_candidates,
    union,
)
from app.recsys.catalog import NO_ROW, Catalog, TitleRecord
from app.recsys.scorer import (
    Weights,
    bayesian_quality,
    cold_start_components,
    min_max,
    popularity_score,
    recency_score,
    score,
)

M, T = TitleType.MOVIE, TitleType.TV


def rec(
    tmdb_id: int,
    type_: TitleType = M,
    *,
    vote_avg: float | None = 7.0,
    vote_count: int = 100,
    popularity: float = 1.0,
    release: date | None = date(2020, 1, 1),
    genres: tuple[str, ...] = ("drama",),
    cert: dict[str, str] | None = None,
) -> TitleRecord:
    return TitleRecord(
        tmdb_id=tmdb_id,
        type=type_,
        title=f"T{tmdb_id}",
        original_language="en",
        genres=genres,
        release_date=release,
        runtime=100,
        vote_avg=vote_avg,
        vote_count=vote_count,
        popularity=popularity,
        adult=False,
        certification=cert or {},
    )


# --- candidates ------------------------------------------------------------------


def test_top_k_orders_and_breaks_ties_by_row() -> None:
    values = np.array([0.5, 0.9, 0.5, 0.1, 0.9])
    eligible = np.ones(5, dtype=bool)
    assert top_k(values, eligible, 3).tolist() == [1, 4, 0]
    assert top_k(values, eligible, 10).tolist() == [1, 4, 0, 2, 3]
    assert top_k(values, eligible, 0).tolist() == []
    eligible[1] = False
    assert top_k(values, eligible, 2).tolist() == [4, 0]
    assert top_k(values, np.zeros(5, dtype=bool), 2).tolist() == []


def test_top_k_many_ties_at_cutoff() -> None:
    values = np.array([1.0] + [0.5] * 50)
    assert top_k(values, np.ones(51, dtype=bool), 3).tolist() == [0, 1, 2]


def test_similarity_ignores_unembedded_and_masked_rows() -> None:
    matrix = np.eye(3, dtype=np.float32)
    vector_rows = np.array([2, NO_ROW, 0, 1])
    sims = item_similarity(matrix, vector_rows, np.array([1, 0, 0], dtype=np.float32))
    assert np.isnan(sims[1])
    assert sims.tolist()[2] == 1.0
    mask = np.array([True, True, True, False])
    assert similar_candidates(sims, mask, 5).tolist() == [2, 0]
    assert np.isnan(item_similarity(matrix, np.full(2, NO_ROW), matrix[0])).all()


def test_trending_and_union() -> None:
    pop = np.array([5.0, 1.0, 9.0, 3.0])
    assert trending_candidates(pop, np.array([True, True, False, True]), 2).tolist() == [0, 3]
    assert union(np.array([3, 1]), np.array([1, 0, 3, 2])).tolist() == [3, 1, 0, 2]


# --- scorer ----------------------------------------------------------------------


def test_bayesian_quality_uses_type_mean() -> None:
    cat = Catalog.from_rows(
        [
            rec(1, vote_avg=8.0, vote_count=100),
            rec(2, vote_avg=6.0, vote_count=300),
            rec(3, vote_avg=None, vote_count=0),
            rec(4, T, vote_avg=9.0, vote_count=50),
        ]
    )
    q = bayesian_quality(cat, m=100)
    c_movie = 7.0  # mean of 8 and 6 (unrated title ignored)
    assert q[0] == pytest.approx((0.5 * 8 + 0.5 * c_movie) / 10)
    assert q[1] == pytest.approx((0.75 * 6 + 0.25 * c_movie) / 10)
    assert q[2] == pytest.approx(c_movie / 10)
    assert q[3] == pytest.approx(0.9)  # only TV title: its own mean
    assert bayesian_quality(cat, m=0)[0] == pytest.approx(0.8)


def test_popularity_and_recency() -> None:
    cat = Catalog.from_rows(
        [
            rec(1, popularity=0, release=date(2026, 11, 5)),
            rec(2, popularity=99, release=date(2025, 11, 5)),
            rec(3, popularity=9, release=None),
        ]
    )
    np.testing.assert_allclose(popularity_score(cat), [0, 1, np.log1p(9) / np.log1p(99)])
    np.testing.assert_allclose(recency_score(cat, date(2026, 11, 5), 365), [1, 0.5, 0])
    flat = Catalog.from_rows([rec(1, popularity=0)])
    assert popularity_score(flat).tolist() == [0.0]
    assert popularity_score(Catalog.from_rows([])).shape == (0,)


def test_min_max() -> None:
    np.testing.assert_allclose(min_max(np.array([1.0, 3.0, np.nan, 2.0])), [0, 1, 0, 0.5])
    assert min_max(np.array([0.4, 0.4])).tolist() == [1.0, 1.0]
    assert min_max(np.array([np.nan])).tolist() == [0.0]


def test_score_combines_weights() -> None:
    rows = np.array([0, 2])
    sims = np.array([0.2, 0.0, 0.8])
    q, p, r = np.array([0.5, 0, 0.7]), np.array([0.1, 0, 0.3]), np.array([1.0, 0, 0.0])
    neg = np.array([0.5, 0.0, -0.4])
    c = score(
        rows,
        sims=sims,
        neg_sims=neg,
        quality=q,
        pop=p,
        recency=r,
        weights=Weights(sim=0.7, qual=0.15, pop=0.1, rec=0.05, neg=0.2),
    )
    np.testing.assert_allclose(c.sim, [0, 1])
    np.testing.assert_allclose(c.neg, [0.5, 0])  # clipped at 0
    np.testing.assert_allclose(
        c.score, [0.15 * 0.5 + 0.1 * 0.1 + 0.05 - 0.2 * 0.5, 0.7 + 0.15 * 0.7 + 0.1 * 0.3]
    )
    assert set(c.of(0)) == {"sim", "quality", "pop", "recency", "neg", "score"}


def test_cold_start_prefers_quality_and_popularity() -> None:
    rows = np.array([0, 1])
    c = cold_start_components(
        rows, quality=np.array([0.9, 0.5]), pop=np.array([0.9, 0.2]), recency=np.array([0, 1.0])
    )
    assert c.score[0] > c.score[1]
    assert c.sim.tolist() == [0, 0]


# --- catalog ---------------------------------------------------------------------


def test_catalog_snapshot_helpers() -> None:
    cat = Catalog.from_rows(
        [rec(1, genres=("drama", "crime"), cert={"US": " R "}), rec(2, T, genres=())]
    )
    assert len(cat) == 2
    assert cat.primary_genre(0) == "drama"
    assert cat.primary_genre(1) is None
    assert cat.certification_for("us") == ["R", None]
    assert cat.genre_rows["crime"].tolist() == [0]
    assert cat.vector_rows(["tv:2", "movie:9"]).tolist() == [NO_ROW, 0]
