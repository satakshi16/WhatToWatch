"""Per-country streaming provider list: TMDB fetch + `providers` table cache (P1-04, P1-07).

The list for a country is the union of TMDB's movie and TV provider lists. The API serves
it from the `providers` table and refreshes it from TMDB when it is missing or older than
`PROVIDERS_TTL_DAYS`. If TMDB fails, a stale cached list is served instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.base import utcnow
from app.db.models import Provider, TitleType
from app.ingest.tmdb_client import TMDBClient, TMDBError

log = get_logger(__name__)

TMDBFactory = Callable[[], TMDBClient]


class ProvidersUnavailableError(Exception):
    """No cached provider list for the country and TMDB could not be reached."""

    def __init__(self, country: str) -> None:
        super().__init__(f"Provider list for {country} is unavailable")
        self.country = country


@dataclass(frozen=True, slots=True)
class ProviderRow:
    provider_id: int
    name: str
    logo_path: str | None
    display_priority: int | None


async def fetch_provider_list(tmdb: TMDBClient, country: str) -> list[ProviderRow]:
    """Union of TMDB's movie + TV provider lists for `country` (movie entry wins on overlap)."""
    movie, tv = await asyncio.gather(
        tmdb.provider_list(TitleType.MOVIE, country),
        tmdb.provider_list(TitleType.TV, country),
    )
    merged = {p.provider_id: p for p in [*tv, *movie]}
    return [
        ProviderRow(
            provider_id=p.provider_id,
            name=p.provider_name,
            logo_path=p.logo_path,
            display_priority=p.display_priorities.get(country, p.display_priority),
        )
        for p in merged.values()
    ]


def replace_providers(
    session: Session, country: str, rows: Sequence[ProviderRow], now: datetime | None = None
) -> None:
    """Replace the cached list for `country` (caller commits)."""
    now = now or utcnow()
    session.execute(delete(Provider).where(Provider.country == country))
    session.add_all(
        Provider(
            provider_id=r.provider_id,
            country=country,
            name=r.name,
            logo_path=r.logo_path,
            display_priority=r.display_priority,
            fetched_at=now,
        )
        for r in rows
    )


def _sort_key(p: Provider) -> tuple[bool, int, str, int]:
    # Lowest display priority first; providers without one go last.
    priority = p.display_priority
    return (priority is None, priority or 0, p.name.casefold(), p.provider_id)


def cached_providers(session: Session, country: str) -> list[Provider]:
    rows = session.scalars(select(Provider).where(Provider.country == country)).all()
    return sorted(rows, key=_sort_key)


async def get_country_providers(
    session: Session,
    country: str,
    tmdb_factory: TMDBFactory,
    *,
    ttl: timedelta,
    now: datetime | None = None,
) -> list[Provider]:
    """Cached provider list for `country`, sorted by display priority; refreshed when stale.

    Raises `ProvidersUnavailableError` only if there is no cached list and TMDB fails.
    An empty TMDB list is a valid answer (nothing is cached, so it is re-checked next time).
    """
    now = now or utcnow()
    cached = cached_providers(session, country)
    if cached and min(p.fetched_at for p in cached) > now - ttl:
        return cached
    try:
        async with tmdb_factory() as tmdb:
            rows = await fetch_provider_list(tmdb, country)
    except TMDBError as exc:
        log.warning(
            "providers_refresh_failed",
            country=country,
            error=type(exc).__name__,
            serving_stale=bool(cached),
        )
        if cached:
            return cached
        raise ProvidersUnavailableError(country) from exc
    replace_providers(session, country, rows, now)
    session.commit()
    log.info("providers_refreshed", country=country, providers=len(rows))
    return cached_providers(session, country)
