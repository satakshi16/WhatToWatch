"""`GET /recommendations?n=10` (P1-12, FR-20, FR-22, FR-26, FR-29): on-demand list.

Runs `RecommendationService.recommend` (§8) for the current user and maps each result to
the §7 item shape, adding display fields from `titles` (localized title, poster, year,
runtime / seasons, genres) and provider names / logos from the cached provider list.
Onboarding must be finished first (409 otherwise). Admins can pass `?debug=true` to get
score components per item plus the pipeline debug block. `?exclude=movie:123&exclude=tv:45`
leaves titles out for this request ("Refresh" passes the ones already shown).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import select, tuple_

from app.api.deps import AppSettings, CurrentUser, DbSession, TMDBClientFactory
from app.core.logging import get_logger
from app.db.models import OnboardingStep, Title, TitleType
from app.ingest.tmdb_client import logo_url, poster_url
from app.recsys.availability import Offer
from app.recsys.service import (
    PreferencesMissingError,
    RecommendationService,
    RecommendedItem,
)
from app.schemas.recommendations import (
    MAX_RECOMMENDATIONS,
    RecommendationItemOut,
    RecommendationsOut,
    RecommendedProviderOut,
    ScoreComponentsOut,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])
log = get_logger(__name__)

TMDB_WEB_BASE_URL = "https://www.themoviedb.org"
MAX_EXCLUDE = 200
ONBOARDING_INCOMPLETE = "Finish onboarding (preferences and seed titles) to get recommendations."

_PROBLEM: dict[str, Any] = {"content": {"application/problem+json": {}}}


def parse_exclude(values: Sequence[str]) -> set[tuple[int, TitleType]]:
    """`["movie:123", "tv:45"]` → title keys; 422 on anything else."""
    keys: set[tuple[int, TitleType]] = set()
    for value in values:
        type_, _, raw_id = value.partition(":")
        try:
            key = (int(raw_id), TitleType(type_))
        except ValueError:
            key = None
        if key is None or not 0 < key[0] < 2**31:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"exclude entries look like 'movie:123' or 'tv:45', got {value!r}.",
            )
        keys.add(key)
    return keys


def watch_page_url(tmdb_id: int, type_: TitleType, country: str) -> str:
    """TMDB's watch page for a title (fallback when availability has no link)."""
    return f"{TMDB_WEB_BASE_URL}/{type_.value}/{tmdb_id}/watch?locale={country}"


def localized_title(title: Title, languages: Sequence[str], country: str) -> str:
    """Title in the user's first preferred language, else the catalog (English) title.

    `localized_titles` is keyed `<lang>-<COUNTRY>`; the user's country wins over others.
    """
    lang = languages[0] if languages else "en"
    if lang == "en":
        return title.title
    translations = title.localized_titles or {}
    exact = translations.get(f"{lang}-{country}")
    if exact:
        return str(exact)
    for code, text in sorted(translations.items()):
        if code.split("-", 1)[0] == lang and text:
            return str(text)
    return title.title


def _provider(offer: Offer) -> RecommendedProviderOut:
    return RecommendedProviderOut(
        provider_id=offer.provider_id,
        name=offer.provider_name,
        logo_url=logo_url(offer.logo_path),
        offer_type=offer.offer_type,
    )


def _item(
    item: RecommendedItem,
    title: Title,
    *,
    languages: Sequence[str],
    country: str,
    debug: bool,
) -> RecommendationItemOut:
    return RecommendationItemOut(
        rank=item.rank,
        tmdb_id=item.tmdb_id,
        type=item.type,
        title=localized_title(title, languages, country),
        original_title=title.original_title,
        year=title.release_date.year if title.release_date else None,
        language=title.original_language,
        runtime_min=title.runtime,
        seasons=title.seasons if item.type is TitleType.TV else None,
        genres=list(title.genres or []),
        poster_url=poster_url(title.poster_path),
        providers=[_provider(o) for o in item.offers],
        watch_link=item.availability.link or watch_page_url(item.tmdb_id, item.type, country),
        availability_checked_at=item.availability.fetched_at,
        reason=item.reason,
        reason_source=item.reason_source,  # type: ignore[arg-type]
        slot=item.slot,  # type: ignore[arg-type]
        score=round(item.score, 6),
        pool=item.pool if debug else None,
        components=ScoreComponentsOut(**item.components) if debug else None,
    )


@router.get(
    "",
    response_model=RecommendationsOut,
    responses={
        403: {"description": "`debug=true` without admin rights", **_PROBLEM},
        409: {"description": "Onboarding not finished", **_PROBLEM},
    },
)
async def get_recommendations(
    request: Request,
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    tmdb_factory: TMDBClientFactory,
    n: Annotated[
        int | None,
        Query(ge=1, le=MAX_RECOMMENDATIONS, description="List size; defaults to REC_N_DEFAULT."),
    ] = None,
    debug: Annotated[
        bool, Query(description="Admins only: score components and pipeline stats.")
    ] = False,
    exclude: Annotated[
        list[str] | None,
        Query(
            max_length=MAX_EXCLUDE,
            description="Titles to leave out this time, as `movie:<tmdb_id>` / `tv:<tmdb_id>`.",
        ),
    ] = None,
) -> RecommendationsOut:
    """Ranked titles streamable on the user's platforms, with a reason for each.

    Fewer than `n` items come back when not enough titles pass the filters and are
    available on the user's platforms (none at all without platforms in their country).
    """
    if debug and not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Admin privileges required.")
    prefs = user.preferences
    if prefs is None or prefs.onboarding_step is not OnboardingStep.DONE:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=ONBOARDING_INCOMPLETE)
    size = n or min(settings.rec_n_default, MAX_RECOMMENDATIONS)
    skip = parse_exclude(exclude or [])

    service = RecommendationService.from_settings(
        db,
        store=request.app.state.embedder.store,
        tmdb_factory=tmdb_factory,
        settings=settings,
        catalog_cache=request.app.state.catalog_cache,
    )
    try:
        result = await service.recommend(user.id, size, skip=skip)
    except PreferencesMissingError as exc:  # pragma: no cover - checked above
        raise HTTPException(status.HTTP_409_CONFLICT, detail=ONBOARDING_INCOMPLETE) from exc

    keys = [item.key for item in result.items]
    titles: dict[tuple[int, TitleType], Title] = {}
    if keys:
        rows = db.scalars(select(Title).where(tuple_(Title.tmdb_id, Title.type).in_(keys)))
        titles = {(t.tmdb_id, t.type): t for t in rows}

    items: list[RecommendationItemOut] = []
    for item in result.items:
        title = titles.get(item.key)
        if title is None:  # pragma: no cover - deleted between ranking and lookup
            log.warning("recommendation_title_missing", tmdb_id=item.tmdb_id, type=item.type)
            continue
        items.append(
            _item(item, title, languages=prefs.languages, country=user.country, debug=debug)
        )

    return RecommendationsOut(
        n=size,
        cold_start=result.cold_start,
        generated_at=result.generated_at,
        config_hash=result.config_hash,
        model_version=result.model_version,
        items=items,
        debug=result.debug if debug else None,
    )
