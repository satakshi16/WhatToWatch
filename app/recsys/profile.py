"""User profile vector from feedback signals (§8.2).

`profile = normalize( Σ wᵢ · dᵢ · vᵢ )` over the user's signals, with
`wᵢ` from the signal table and `dᵢ = 0.5 ** (age_days / PROFILE_HALF_LIFE_DAYS)`.

Phase 1 signals are `rating` (1-5) and `not_interested`; other kinds (thumb, watchlist,
click, shown-but-ignored) arrive in Phase 2 and are ignored here. Per title only the
**latest explicit signal** counts (a re-rating replaces the earlier rating; a later
`not_interested` replaces an earlier rating and vice versa). Titles without a vector
(not embedded, §8.1 / §13 Q7) cannot shape the profile and are counted as skipped.

If the weighted sum is ~zero (e.g. only 3-star ratings) there is no profile and the
caller falls back to the cold-start ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.db.models import InteractionKind
from app.ingest.build_embeddings import TitleKey, Vectors

# §8.2 signal table (Phase 1 rows).
RATING_WEIGHTS: dict[int, float] = {5: 1.0, 4: 0.6, 3: 0.0, 2: -0.5, 1: -1.0}
NOT_INTERESTED_WEIGHT = -0.7
EXPLICIT_KINDS: frozenset[InteractionKind] = frozenset(
    {InteractionKind.RATING, InteractionKind.NOT_INTERESTED}
)
# Below this norm the weighted sum counts as zero → cold start.
ZERO_NORM = 1e-6
SECONDS_PER_DAY = 86_400.0

VectorLookup = Callable[[TitleKey], Vectors | None]


@dataclass(frozen=True, slots=True)
class Signal:
    """One feedback event (an `interactions` row)."""

    key: TitleKey
    kind: InteractionKind
    value: float | None
    created_at: datetime
    # Insertion order, to break `created_at` ties (interactions.id).
    seq: int = 0


@dataclass(frozen=True, slots=True)
class Profile:
    vector: Vectors | None  # L2-normalized, or None → cold start
    neg_vector: Vectors | None  # normalized sum of negative signals only
    signals_used: int  # titles that contributed (non-zero weight, embedded)
    signals_unembedded: int  # titles with a non-zero weight but no vector
    liked: tuple[TitleKey, ...]  # latest rating ≥ 4, most recent first (for reasons)

    @property
    def cold_start(self) -> bool:
        return self.vector is None


def signal_weight(kind: InteractionKind, value: float | None) -> float:
    """wᵢ from the §8.2 table; 0 for kinds Phase 1 does not use."""
    if kind is InteractionKind.RATING and value is not None:
        return RATING_WEIGHTS.get(round(value), 0.0)
    if kind is InteractionKind.NOT_INTERESTED:
        return NOT_INTERESTED_WEIGHT
    return 0.0


def decay(age_days: float, half_life_days: float) -> float:
    """dᵢ = 0.5 ** (age_days / half_life); future timestamps count as age 0."""
    return float(0.5 ** (max(age_days, 0.0) / half_life_days))


def latest_explicit(signals: Iterable[Signal]) -> dict[TitleKey, Signal]:
    """Per title, the most recent explicit signal (ties → higher `seq`)."""
    latest: dict[TitleKey, Signal] = {}
    for s in signals:
        if s.kind not in EXPLICIT_KINDS:
            continue
        current = latest.get(s.key)
        if current is None or (s.created_at, s.seq) > (current.created_at, current.seq):
            latest[s.key] = s
    return latest


def _normalize(v: Vectors) -> Vectors | None:
    norm = float(np.linalg.norm(v))
    if norm < ZERO_NORM:
        return None
    out: Vectors = (v / norm).astype(np.float32)
    return out


def build_profile(
    signals: Iterable[Signal],
    vector_of: VectorLookup,
    *,
    now: datetime,
    half_life_days: float,
) -> Profile:
    latest = latest_explicit(signals)
    total: Vectors | None = None
    negative: Vectors | None = None
    used = unembedded = 0
    for key, s in latest.items():
        weight = signal_weight(s.kind, s.value)
        if weight == 0.0:
            continue
        vector = vector_of(key)
        if vector is None:
            unembedded += 1
            continue
        age_days = (now - s.created_at).total_seconds() / SECONDS_PER_DAY
        contribution = (weight * decay(age_days, half_life_days) * vector).astype(np.float32)
        total = contribution if total is None else total + contribution
        if weight < 0:
            negative = -contribution if negative is None else negative - contribution
        used += 1
    liked = sorted(
        (
            s
            for s in latest.values()
            if s.kind is InteractionKind.RATING and s.value is not None and s.value >= 4
        ),
        key=lambda s: (s.created_at, s.seq),
        reverse=True,
    )
    return Profile(
        vector=_normalize(total) if total is not None else None,
        neg_vector=_normalize(negative) if negative is not None else None,
        signals_used=used,
        signals_unembedded=unembedded,
        liked=tuple(s.key for s in liked),
    )
