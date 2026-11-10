"""§8.7 MMR ordering, FR-25 caps (property test), relaxation and cap scaling."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from app.recsys.rerank import mmr_rerank, scaled_cap


def run(
    scores: list[float],
    vectors: np.ndarray,
    langs: list[str],
    genres: list[str | None],
    *,
    n: int = 10,
    lam: float = 0.7,
    single: bool = False,
    cap_lang: int = 4,
    cap_genre: int = 3,
):  # type: ignore[no-untyped-def]
    return mmr_rerank(
        np.array(scores, dtype=np.float64),
        vectors.astype(np.float32),
        langs,
        genres,
        n=n,
        lam=lam,
        cap_per_language=cap_lang,
        cap_per_genre=cap_genre,
        single_language=single,
    )


def test_first_pick_is_highest_score_then_diversity_wins() -> None:
    # 0 and 1 are near-duplicates; 2 is orthogonal with a slightly lower score.
    v = np.array([[1, 0], [1, 0.01], [0, 1]], dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    res = run([1.0, 0.95, 0.8], v, ["en"] * 3, ["a", "b", "c"], n=3)
    assert res.order == [0, 2, 1]
    # λ = 1 → pure score order.
    assert run([1.0, 0.95, 0.8], v, ["en"] * 3, ["a", "b", "c"], n=3, lam=1.0).order == [0, 1, 2]


def test_unembedded_items_have_zero_similarity() -> None:
    v = np.array([[1, 0], [0, 0], [1, 0]], dtype=np.float32)
    res = run([1.0, 0.5, 0.9], v, ["en"] * 3, [None] * 3, n=3, lam=0.5)
    # After 0, item 2 has max_cos 1 (0.45 - 0.5) and item 1 has 0 (0.25 - 0) → 1 first.
    assert res.order == [0, 1, 2]


@pytest.mark.parametrize("trial", range(30))
def test_caps_hold_when_enough_diversity(trial: int) -> None:
    rng = random.Random(trial)  # noqa: S311 - test data
    m = 30
    langs = [rng.choice(["en", "hi", "ko", "ja"]) for _ in range(m)]
    genres = [rng.choice(["drama", "comedy", "action", "horror", "romance"]) for _ in range(m)]
    # Guarantee enough of each so caps can be met without relaxation.
    langs[:12] = ["en", "hi", "ko", "ja"] * 3
    genres[:15] = ["drama", "comedy", "action", "horror", "romance"] * 3
    vectors = np.random.default_rng(trial).normal(size=(m, 8))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    res = run([rng.random() for _ in range(m)], vectors, langs, genres)
    assert len(res.order) == 10
    assert len(set(res.order)) == 10
    if not res.relaxations:
        assert max(Counter(langs[i] for i in res.order).values()) <= 4
        assert max(Counter(genres[i] for i in res.order).values()) <= 3
    assert res.language_cap is not None
    assert max(Counter(langs[i] for i in res.order).values()) <= res.language_cap
    assert max(Counter(genres[i] for i in res.order).values()) <= res.genre_cap


def test_caps_skip_top_scored_items_that_break_them() -> None:
    m = 18
    langs = ["en"] * 6 + ["ko"] * 6 + ["ja"] * 6
    genres = [f"g{i}" for i in range(m)]
    res = run([1.0] * 6 + [0.1] * 12, np.zeros((m, 2)), langs, genres)
    picked = Counter(langs[i] for i in res.order)
    assert picked["en"] == 4  # the 5th and 6th English titles outscore the rest but are capped
    assert sum(picked.values()) == 10
    assert res.relaxations == []


def test_single_language_skips_language_cap() -> None:
    m = 12
    res = run([1.0] * m, np.zeros((m, 2)), ["en"] * m, [f"g{i % 4}" for i in range(m)], single=True)
    assert len(res.order) == 10
    assert res.language_cap is None
    assert res.relaxations == []


def test_relaxation_genre_first_then_language() -> None:
    # 2 genres x cap 3 = 6 < 10 → genre cap raised to 5.
    m = 20
    genres = ["a", "b"] * 10
    langs = ["en", "hi", "ko", "ja", "es"] * 4
    res = run([1.0] * m, np.zeros((m, 2)), langs, genres)
    assert len(res.order) == 10
    assert res.relaxations == [{"cap": "genre", "value": 4}, {"cap": "genre", "value": 5}]
    # One language only, multi-language user: genre bump can't help → language raised too.
    res = run([1.0] * m, np.zeros((m, 2)), ["en"] * m, [f"g{i}" for i in range(m)])
    assert len(res.order) == 10
    assert {"cap": "language", "value": 5} in res.relaxations
    assert res.language_cap == 10


def test_fewer_candidates_than_n_and_empty() -> None:
    res = run([0.3, 0.2], np.zeros((2, 2)), ["en", "en"], ["a", "a"])
    assert sorted(res.order) == [0, 1]
    assert run([], np.zeros((0, 2)), [], []).order == []


@pytest.mark.parametrize(
    ("cap", "n", "expected"), [(4, 10, 4), (3, 10, 3), (4, 3, 2), (3, 3, 1), (4, 20, 8), (3, 1, 1)]
)
def test_scaled_cap(cap: int, n: int, expected: int) -> None:
    assert scaled_cap(cap, n) == expected
