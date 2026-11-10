"""`GET /providers?country=` (P1-07, FR-11): streaming platforms available in a country."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import AppSettings, CurrentUser, DbSession, TMDBClientFactory
from app.core.config import Settings
from app.db.models import Provider
from app.ingest.providers import ProvidersUnavailableError, TMDBFactory, get_country_providers
from app.ingest.tmdb_client import logo_url
from app.schemas.common import CountryCode
from app.schemas.preferences import ProviderListOut, ProviderOut

router = APIRouter(prefix="/providers", tags=["providers"])


async def load_country_providers(
    db: Session, country: str, tmdb_factory: TMDBFactory, settings: Settings
) -> list[Provider]:
    """Provider list for `country` (cached, refreshed when stale); 503 if unavailable."""
    try:
        return await get_country_providers(
            db, country, tmdb_factory, ttl=timedelta(days=settings.providers_ttl_days)
        )
    except ProvidersUnavailableError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The provider list for {country} is temporarily unavailable. Try again later.",
            headers={"Retry-After": "60"},
        ) from exc


@router.get("", response_model=ProviderListOut)
async def list_providers(
    user: CurrentUser,
    db: DbSession,
    settings: AppSettings,
    tmdb_factory: TMDBClientFactory,
    country: Annotated[
        CountryCode | None, Query(description="ISO-3166-1 alpha-2; defaults to the user's.")
    ] = None,
) -> ProviderListOut:
    """Union of TMDB's movie + TV provider lists, sorted by TMDB display priority."""
    code = country or user.country
    providers = await load_country_providers(db, code, tmdb_factory, settings)
    return ProviderListOut(
        country=code,
        providers=[
            ProviderOut(
                provider_id=p.provider_id,
                name=p.name,
                logo_url=logo_url(p.logo_path),
                display_priority=p.display_priority,
            )
            for p in providers
        ],
        fetched_at=min((p.fetched_at for p in providers), default=None),
    )
