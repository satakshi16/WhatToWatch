"""`POST /interactions` (P1-09, FR-23, FR-33): rating and not_interested in Phase 1.

Titles not in the catalog are fetched from TMDB and inserted first (the `interactions`
table has a foreign key to `titles`), then embedded on the fly when they have open
content (§8.1). P2-06 adds the other kinds (thumb, seen, watchlist, click).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from fastapi import APIRouter, status
from fastapi.exceptions import RequestValidationError
from sqlalchemy.orm import Session

from app.api.deps import (
    AppSettings,
    CurrentUser,
    DbSession,
    Embedder,
    TMDBClientFactory,
    tmdb_unavailable,
)
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import InteractionKind, InteractionSource, TitleType, User
from app.ingest.providers import TMDBFactory
from app.ingest.tmdb_client import TMDBError
from app.recsys.embeddings import OnDemandEmbedder
from app.recsys.feedback import (
    EnsureResult,
    UnknownTitlesError,
    add_interactions,
    ensure_titles,
)
from app.schemas.feedback import InteractionIn, InteractionOut

router = APIRouter(prefix="/interactions", tags=["interactions"])
log = get_logger(__name__)

TitleKey = tuple[int, TitleType]


def catalog_countries(settings: Settings, user: User) -> list[str]:
    """Countries whose availability is stored for titles fetched on demand."""
    return list(dict.fromkeys([*settings.seed_countries, user.country]))


async def ensure_titles_or_http(
    db: Session,
    keys: Sequence[TitleKey],
    *,
    user: User,
    settings: Settings,
    tmdb_factory: TMDBFactory,
    embedder: OnDemandEmbedder,
    loc_of: Callable[[TitleKey], tuple[str | int, ...]],
) -> EnsureResult:
    """`ensure_titles` with errors mapped to HTTP: unknown title → 422, TMDB down → 503."""
    try:
        return await ensure_titles(
            db,
            keys,
            tmdb_factory=tmdb_factory,
            countries=catalog_countries(settings, user),
            embedder=embedder,
        )
    except UnknownTitlesError as exc:
        raise RequestValidationError(
            [
                {
                    "loc": loc_of(key),
                    "msg": f"no {key[1].value} with TMDB id {key[0]}",
                    "type": "value_error.title_not_found",
                }
                for key in exc.keys
            ]
        ) from exc
    except TMDBError as exc:
        log.warning("title_fetch_failed", error=type(exc).__name__)
        raise tmdb_unavailable() from exc


@router.post("", response_model=InteractionOut, status_code=status.HTTP_201_CREATED)
async def create_interaction(
    body: InteractionIn,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    tmdb_factory: TMDBClientFactory,
    embedder: Embedder,
) -> InteractionOut:
    """Record a 1-5 star rating or a "not interested" for one title.

    Either way the title is never recommended to this user again (FR-23). Ratings are
    append-only: rating a title again adds a new event and the latest one counts.
    """
    key: TitleKey = (body.tmdb_id, body.type)
    ensured = await ensure_titles_or_http(
        db,
        [key],
        user=user,
        settings=settings,
        tmdb_factory=tmdb_factory,
        embedder=embedder,
        loc_of=lambda _: ("body", "tmdb_id"),
    )
    kind = InteractionKind(body.kind)
    (row,) = add_interactions(
        db,
        user.id,
        [key],
        kind=kind,
        source=InteractionSource.BROWSE,
        value=float(body.value) if body.value is not None else None,
    )
    db.commit()
    db.refresh(row)
    log.info(
        "interaction_recorded",
        user_id=str(user.id),
        kind=kind.value,
        added_to_catalog=bool(ensured.inserted),
        embedded=ensured.embedded.get(key, False),
    )
    out = InteractionOut.model_validate(row)
    out.added_to_catalog = key in ensured.inserted
    out.embedded = ensured.embedded.get(key, False)
    return out
