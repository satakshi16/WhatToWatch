"""§8.2 profile math: signal weights, time decay, latest-explicit-wins, cold start."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from app.db.models import InteractionKind, TitleType
from app.ingest.build_embeddings import TitleKey, Vectors
from app.recsys.profile import Signal, build_profile, decay, latest_explicit, signal_weight

M = TitleType.MOVIE
NOW = datetime(2026, 11, 5, tzinfo=UTC)
R, NI, SEEN = InteractionKind.RATING, InteractionKind.NOT_INTERESTED, InteractionKind.SEEN


def e(i: int, dim: int = 4) -> Vectors:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


VECTORS: dict[TitleKey, Vectors] = {(1, M): e(0), (2, M): e(1), (3, M): e(2)}


def sig(
    tmdb_id: int, kind: InteractionKind = R, value: float | None = 5, days: float = 0, seq: int = 0
) -> Signal:
    return Signal((tmdb_id, M), kind, value, NOW - timedelta(days=days), seq)


def profile(*signals: Signal, half_life: float = 90) -> tuple[Vectors | None, object]:
    p = build_profile(signals, VECTORS.get, now=NOW, half_life_days=half_life)
    return p.vector, p


@pytest.mark.parametrize(
    ("kind", "value", "weight"),
    [
        (R, 5, 1.0),
        (R, 4, 0.6),
        (R, 3, 0.0),
        (R, 2, -0.5),
        (R, 1, -1.0),
        (NI, None, -0.7),
        (SEEN, None, 0.0),
        (InteractionKind.THUMB, 1, 0.0),  # Phase 2
    ],
)
def test_signal_weights(kind: InteractionKind, value: float | None, weight: float) -> None:
    assert signal_weight(kind, value) == weight


def test_decay_half_life() -> None:
    assert decay(0, 90) == 1.0
    assert decay(90, 90) == pytest.approx(0.5)
    assert decay(180, 90) == pytest.approx(0.25)
    assert decay(-5, 90) == 1.0  # future timestamps count as now


def test_weighted_decayed_sum_is_normalized() -> None:
    # 5-star now (w=1, d=1) on e0 and 4-star 90 days ago (w=0.6, d=0.5) on e1.
    vector, _ = profile(sig(1, value=5), sig(2, value=4, days=90))
    expected = np.array([1.0, 0.3, 0, 0]) / np.linalg.norm([1.0, 0.3])
    assert vector is not None
    np.testing.assert_allclose(vector, expected, rtol=1e-6)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_negative_signals_push_away_and_feed_neg_profile() -> None:
    p = build_profile([sig(1, value=5), sig(2, kind=NI)], VECTORS.get, now=NOW, half_life_days=90)
    assert p.vector is not None
    assert p.vector[0] > 0 > p.vector[1]
    assert p.neg_vector is not None
    np.testing.assert_allclose(p.neg_vector, e(1))


def test_latest_explicit_signal_wins() -> None:
    # Re-rated 5 → 1: only the 1-star counts.
    vector, _ = profile(sig(1, value=5, days=10), sig(1, value=1, days=1))
    assert vector is not None
    np.testing.assert_allclose(vector, -e(0))
    # Rated, then marked not interested later → not_interested wins.
    picks = latest_explicit([sig(2, value=5, days=3), sig(2, kind=NI, days=1)])
    assert picks[(2, M)].kind is NI
    # Same timestamp → higher seq (later insert) wins.
    tie = latest_explicit([sig(3, value=2, seq=2), sig(3, value=5, seq=1)])
    assert tie[(3, M)].value == 2
    # Non-explicit kinds are ignored.
    assert latest_explicit([sig(1, kind=SEEN)]) == {}


def test_only_neutral_ratings_is_cold_start() -> None:
    _, p = profile(sig(1, value=3), sig(2, value=3))
    assert p.cold_start  # type: ignore[attr-defined]
    assert p.signals_used == 0  # type: ignore[attr-defined]


def test_cancelling_signals_is_cold_start() -> None:
    p = build_profile(
        [sig(1, value=5), sig(1 + 100, value=1)],
        {(1, M): e(0), (101, M): e(0)}.get,
        now=NOW,
        half_life_days=90,
    )
    assert p.cold_start


def test_unembedded_titles_are_counted_not_used() -> None:
    p = build_profile([sig(1), sig(99)], VECTORS.get, now=NOW, half_life_days=90)
    assert p.signals_used == 1
    assert p.signals_unembedded == 1
    assert not p.cold_start


def test_liked_lists_latest_ratings_of_four_plus_newest_first() -> None:
    p = build_profile(
        [
            sig(1, value=4, days=5),
            sig(2, value=5, days=1),
            sig(3, value=2),
            sig(99, value=5, days=9),
        ],
        VECTORS.get,
        now=NOW,
        half_life_days=90,
    )
    assert p.liked == ((2, M), (1, M), (99, M))
