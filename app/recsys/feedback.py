"""User feedback: make titles usable, record interactions, FR-23 exclusions (P1-09).

Used by `POST /onboarding/seeds` and `POST /interactions`:
1. `ensure_titles`: titles not in the catalog are fetched from TMDB and inserted, then
   every title missing from the embedding index is embedded on the fly when it has open
   content (§8.1), so it counts for recommendations immediately.
2. `add_interactions`: append rows to the `interactions` log (never updated).
3. `excluded_title_keys`: the FR-23 hard filter (rating, `seen`, `not_interested`),
   which the recommender (P1-11) applies to every candidate set.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import Interaction, InteractionKind, InteractionSource
from app.ingest.build_embeddings import TitleKey
from app.ingest.ingest_catalog import existing_keys, insert_missing_titles
from app.ingest.providers import TMDBFactory
from app.recsys.embeddings import OnDemandEmbedder

log = get_logger(__name__)

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
) -> list[Interaction]:
    """Append one interaction per key (caller commits). Titles must exist in `titles`."""
    rows = [
        Interaction(
            user_id=user_id, tmdb_id=tmdb_id, type=type_, kind=kind, value=value, source=source
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
