"""User feedback: make titles usable, record interactions, FR-23 exclusions (P1-09).

Used by `POST /onboarding/seeds` and `POST /interactions`:
1. `ensure_titles`: titles not in the catalog are fetched from TMDB and inserted, then
   every title missing from the embedding index is embedded on the fly when it has open
   content (§8.1), so it counts for recommendations immediately.
2. `add_interactions`: append rows to the `interactions` log (never updated).
3. `excluded_title_keys`: the FR-23 hard filter (rating, `seen`, `not_interested`),
   which the recommender (P1-11) applies to every candidate set.
4. Seed sets: each `POST /onboarding/seeds` tags its rows with one `payload.seed_set` id
   and **replaces** the user's seeds. Rows from earlier submissions stay in the log (it is
   append-only) but are "stale": the profile ignores them (`stale_seed_ids`). Legacy rows
   without an id form one set, current until the next submission.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import Interaction, InteractionKind, InteractionSource
from app.ingest.build_embeddings import TitleKey
from app.ingest.ingest_catalog import existing_keys, insert_missing_titles
from app.ingest.providers import TMDBFactory
from app.recsys.embeddings import OnDemandEmbedder

log = get_logger(__name__)

SEED_SET_KEY = "seed_set"

# FR-23: a title with any of these interactions is never recommended again.
EXCLUDING_KINDS: tuple[InteractionKind, ...] = (
    InteractionKind.RATING,
    InteractionKind.SEEN,
    InteractionKind.NOT_INTERESTED,
)


class UnknownTitlesError(Exception):
    """Some titles are neither in the catalog nor on TMDB (404)."""

    def __init__(self, keys: Sequence[TitleKey]) -> None:
        super().__init__(f"{len(keys)} title(s) not found on TMDB")
        self.keys = list(keys)


@dataclass
class EnsureResult:
    # Titles inserted into the catalog by this call (fetched from TMDB).
    inserted: list[TitleKey] = field(default_factory=list)
    # Per title: whether it has a vector in the embedding index (usable for similarity).
    embedded: dict[TitleKey, bool] = field(default_factory=dict)


async def ensure_titles(
    session: Session,
    keys: Sequence[TitleKey],
    *,
    tmdb_factory: TMDBFactory,
    countries: Sequence[str],
    embedder: OnDemandEmbedder | None,
) -> EnsureResult:
    """Make sure every key is in `titles` and, where possible, in the embedding index.

    Raises `UnknownTitlesError` (nothing inserted) if TMDB has no such title, and lets
    `TMDBError` propagate if TMDB cannot be reached. TMDB is only called when some key
    is missing from the catalog. Embedding problems never raise (see `OnDemandEmbedder`).
    """
    keys = list(dict.fromkeys(keys))
    result = EnsureResult()
    if len(existing_keys(session, keys)) < len(keys):
        async with tmdb_factory() as tmdb:
            inserted, not_found = await insert_missing_titles(tmdb, session, keys, countries)
        if not_found:
            raise UnknownTitlesError(not_found)
        result.inserted = inserted
    if embedder is not None:
        result.embedded = await embedder.ensure(session, keys)
    else:
        result.embedded = dict.fromkeys(keys, False)
    return result


def add_interactions(
    session: Session,
    user_id: uuid.UUID,
    keys: Sequence[TitleKey],
    *,
    kind: InteractionKind,
    source: InteractionSource,
    value: float | None = None,
    payload: dict[str, Any] | None = None,
) -> list[Interaction]:
    """Append one interaction per key (caller commits). Titles must exist in `titles`."""
    rows = [
        Interaction(
            user_id=user_id,
            tmdb_id=tmdb_id,
            type=type_,
            kind=kind,
            value=value,
            source=source,
            payload=dict(payload) if payload is not None else None,
        )
        for tmdb_id, type_ in keys
    ]
    session.add_all(rows)
    return rows


def excluded_title_keys(session: Session, user_id: uuid.UUID) -> set[TitleKey]:
    """FR-23: titles the user rated, marked seen or marked not interested."""
    stmt = (
        select(Interaction.tmdb_id, Interaction.type)
        .where(Interaction.user_id == user_id, Interaction.kind.in_(EXCLUDING_KINDS))
        .distinct()
    )
    return {(tmdb_id, type_) for tmdb_id, type_ in session.execute(stmt)}


def _seed_set_of(payload: Any) -> str | None:
    if isinstance(payload, dict):
        value = payload.get(SEED_SET_KEY)
        return str(value) if value is not None else None
    return None


def _split_seed_rows(session: Session, user_id: uuid.UUID) -> tuple[list[Any], list[Any]]:
    """The user's seed rows, split into (current set, stale), each oldest first."""
    stmt = (
        select(Interaction.id, Interaction.tmdb_id, Interaction.type, Interaction.payload)
        .where(
            Interaction.user_id == user_id,
            Interaction.source == InteractionSource.ONBOARDING_SEED,
            Interaction.kind == InteractionKind.RATING,
        )
        # By id, not created_at: ids follow insertion order even if the clock steps back.
        .order_by(Interaction.id)
    )
    rows = list(session.execute(stmt))
    if not rows:
        return [], []
    latest = _seed_set_of(rows[-1].payload)
    current = [r for r in rows if _seed_set_of(r.payload) == latest]
    stale = [r for r in rows if _seed_set_of(r.payload) != latest]
    return current, stale


def current_seed_keys(session: Session, user_id: uuid.UUID) -> list[TitleKey]:
    """The user's current seed titles, in submission order."""
    current, _ = _split_seed_rows(session, user_id)
    return list(dict.fromkeys((r.tmdb_id, r.type) for r in current))


def stale_seed_ids(session: Session, user_id: uuid.UUID) -> set[int]:
    """Interaction ids of seeds replaced by a later submission (ignored by the profile)."""
    _, stale = _split_seed_rows(session, user_id)
    return {r.id for r in stale}
