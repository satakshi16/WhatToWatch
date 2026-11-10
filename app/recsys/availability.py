"""Per-country availability lookups over the `title_availability` cache (P1-10, FR-22, FR-29).

`AvailabilityService.bulk_get(keys, country)` answers from `title_availability` and fetches
TMDB `watch/providers` (JustWatch data) only for titles whose rows for that country are
missing or older than `AVAILABILITY_TTL_DAYS`. Fetches run concurrently under a semaphore
and are capped per call (`AVAILABILITY_MAX_FETCHES_PER_REQUEST`, §8.6), in the order the
keys are given, so a caller passing a scored list spends the budget on its best titles.

A title checked with no offers gets the D-008 marker row, so it is not re-fetched within
the TTL either. A TMDB 404 counts as "no offers". Any other TMDB failure never raises:
stale rows are served (`stale=True`) or, without any rows, the title is reported as not
checked (`fetched_at=None`), which `is_streamable` treats as not streamable.

`is_streamable` is the FR-22 hard filter: at least one offer on one of the user's
platforms with `offer_type` in {flatrate, free, ads}, plus {rent, buy} if `include_rent_buy`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import delete, select, tuple_
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.db.base import utcnow
from app.db.models import (
    NO_PROVIDER_ID,
    OfferType,
    Provider,
    TitleAvailability,
    TitleType,
)
from app.ingest.ingest_catalog import TitleKey, country_availability_rows, existing_keys
from app.ingest.providers import TMDBFactory
from app.ingest.tmdb_client import TMDBClient, TMDBError, TMDBNotFoundError
from app.ingest.tmdb_models import CountryProviders

log = get_logger(__name__)

Clock = Callable[[], datetime]

# FR-22: offer types that count as "streamable" without and with rent/buy.
STREAM_OFFER_TYPES: frozenset[OfferType] = frozenset(
    {OfferType.FLATRATE, OfferType.FREE, OfferType.ADS}
)
RENT_BUY_OFFER_TYPES: frozenset[OfferType] = frozenset({OfferType.RENT, OfferType.BUY})

_CHUNK = 500


def allowed_offer_types(include_rent_buy: bool) -> frozenset[OfferType]:
    return STREAM_OFFER_TYPES | RENT_BUY_OFFER_TYPES if include_rent_buy else STREAM_OFFER_TYPES


@dataclass(frozen=True, slots=True)
class Offer:
    """One provider offer; name/logo come from the cached `providers` list (None if absent)."""

    provider_id: int
    offer_type: OfferType
    provider_name: str | None = None
    logo_path: str | None = None
    display_priority: int | None = None


@dataclass(frozen=True, slots=True)
class Availability:
    """Availability of one title in one country.

    `offers` never contains the D-008 marker. `link` is TMDB's watch page for the title in
    that country. `fetched_at is None` means the title has never been checked there
    (TMDB unreachable or fetch budget used up); `stale` means the rows are older than the
    TTL and could not be refreshed in this call.
    """

    tmdb_id: int
    type: TitleType
    country: str
    offers: tuple[Offer, ...] = ()
    link: str | None = None
    fetched_at: datetime | None = None
    stale: bool = False

    @property
    def key(self) -> TitleKey:
        return self.tmdb_id, self.type

    @property
    def checked(self) -> bool:
        return self.fetched_at is not None


def streamable_offers(
    availability: Availability, user_platforms: Collection[int], include_rent_buy: bool
) -> list[Offer]:
    """Offers that satisfy FR-22 for this user, in display-priority order."""
    allowed = allowed_offer_types(include_rent_buy)
    platforms = set(user_platforms)
    return [
        o for o in availability.offers if o.provider_id in platforms and o.offer_type in allowed
    ]


def is_streamable(
    availability: Availability, user_platforms: Collection[int], include_rent_buy: bool
) -> bool:
    """FR-22: at least one allowed offer on one of the user's platforms in their country."""
    return bool(streamable_offers(availability, user_platforms, include_rent_buy))


@dataclass
class _FetchOutcome:
    key: TitleKey
    offers: CountryProviders | None = None
    ok: bool = False


class AvailabilityService:
    """Cached availability lookups for one DB session (create one per request)."""

    def __init__(
        self,
        session: Session,
        tmdb_factory: TMDBFactory,
        *,
        ttl: timedelta,
        max_fetches: int | None = None,
        concurrency: int = 20,
        clock: Clock = utcnow,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._session = session
        self._tmdb_factory = tmdb_factory
        self._ttl = ttl
        self._max_fetches = max_fetches
        self._concurrency = concurrency
        self._clock = clock
        # TMDB fetches attempted by this service so far (lets callers that walk a list in
        # chunks keep one budget across `bulk_get` calls, §8.6).
        self.fetch_attempts = 0

    @classmethod
    def from_settings(
        cls,
        session: Session,
        tmdb_factory: TMDBFactory,
        settings: Settings,
        *,
        clock: Clock = utcnow,
    ) -> AvailabilityService:
        return cls(
            session,
            tmdb_factory,
            ttl=timedelta(days=settings.availability_ttl_days),
            max_fetches=settings.availability_max_fetches_per_request,
            concurrency=settings.tmdb_max_concurrency,
            clock=clock,
        )

    async def get(self, tmdb_id: int, type_: TitleType, country: str) -> Availability:
        result = await self.bulk_get([(tmdb_id, type_)], country)
        return result[(tmdb_id, type_)]

    async def bulk_get(
        self,
        keys: Sequence[TitleKey],
        country: str,
        *,
        max_fetches: int | None = None,
    ) -> dict[TitleKey, Availability]:
        """Availability for every key in `country` (input order kept, duplicates dropped).

        `max_fetches` overrides the service-wide cap for this call (None = service default;
        a service default of None means no cap). Only titles in the catalog are fetched,
        since `title_availability` rows need a `titles` row.
        """
        country = country.upper()
        keys = list(dict.fromkeys(keys))
        if not keys:
            return {}
        now = self._clock()
        cutoff = now - self._ttl
        cached = self._load(keys, country)

        due = [k for k in keys if k not in cached or _oldest(cached[k]) < cutoff]
        budget = self._max_fetches if max_fetches is None else max_fetches
        fetchable = existing_keys(self._session, due) if due else set()
        to_fetch = [k for k in due if k in fetchable]
        if budget is not None:
            to_fetch = to_fetch[: max(budget, 0)]

        fetched_ok = 0
        self.fetch_attempts += len(to_fetch)
        if to_fetch:
            outcomes = await self._fetch(to_fetch, country)
            fetched_ok = self._store([o for o in outcomes if o.ok], country, now)
            if fetched_ok:
                cached.update(self._load([o.key for o in outcomes if o.ok], country))

        providers = self._providers(country)
        result = {k: _build(k, country, cached.get(k, []), providers, cutoff) for k in keys}
        log.info(
            "availability_lookup",
            country=country,
            titles=len(keys),
            cache_hits=len(keys) - len(due),
            fetched=fetched_ok,
            fetch_failed=len(to_fetch) - fetched_ok,
            skipped=len(due) - len(to_fetch),
        )
        return result

    # --- Internals ---------------------------------------------------------------

    def _load(
        self, keys: Sequence[TitleKey], country: str
    ) -> dict[TitleKey, list[TitleAvailability]]:
        rows: dict[TitleKey, list[TitleAvailability]] = defaultdict(list)
        for start in range(0, len(keys), _CHUNK):
            chunk = keys[start : start + _CHUNK]
            stmt = select(TitleAvailability).where(
                TitleAvailability.country == country,
                tuple_(TitleAvailability.tmdb_id, TitleAvailability.type).in_(chunk),
            )
            for row in self._session.scalars(stmt):
                rows[(row.tmdb_id, row.type)].append(row)
        return dict(rows)

    async def _fetch(self, keys: Sequence[TitleKey], country: str) -> list[_FetchOutcome]:
        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(tmdb: TMDBClient, key: TitleKey) -> _FetchOutcome:
            tmdb_id, type_ = key
            async with semaphore:
                try:
                    providers = await tmdb.watch_providers(type_, tmdb_id)
                except TMDBNotFoundError:
                    return _FetchOutcome(key, None, ok=True)
                except TMDBError as exc:
                    log.warning(
                        "availability_fetch_failed",
                        tmdb_id=tmdb_id,
                        type=type_.value,
                        country=country,
                        error=type(exc).__name__,
                    )
                    return _FetchOutcome(key)
            return _FetchOutcome(key, providers.results.get(country), ok=True)

        try:
            async with self._tmdb_factory() as tmdb:
                return list(await asyncio.gather(*(one(tmdb, k) for k in keys)))
        except TMDBError as exc:  # e.g. no credentials configured
            log.warning("availability_fetch_unavailable", error=type(exc).__name__)
            return [_FetchOutcome(k) for k in keys]

    def _store(self, outcomes: Sequence[_FetchOutcome], country: str, now: datetime) -> int:
        """Replace the rows of every fetched title in `country`; commits once."""
        if not outcomes:
            return 0
        keys = [o.key for o in outcomes]
        for start in range(0, len(keys), _CHUNK):
            self._session.execute(
                delete(TitleAvailability).where(
                    TitleAvailability.country == country,
                    tuple_(TitleAvailability.tmdb_id, TitleAvailability.type).in_(
                        keys[start : start + _CHUNK]
                    ),
                )
            )
        for o in outcomes:
            tmdb_id, type_ = o.key
            self._session.add_all(country_availability_rows(tmdb_id, type_, country, o.offers, now))
        self._session.commit()
        return len(outcomes)

    def _providers(self, country: str) -> dict[int, Provider]:
        stmt = select(Provider).where(Provider.country == country)
        return {p.provider_id: p for p in self._session.scalars(stmt)}


def _oldest(rows: Iterable[TitleAvailability]) -> datetime:
    return min(r.fetched_at for r in rows)


def _offer_sort_key(o: Offer) -> tuple[bool, int, int, str]:
    priority = o.display_priority
    return (priority is None, priority or 0, o.provider_id, o.offer_type.value)


def _build(
    key: TitleKey,
    country: str,
    rows: Sequence[TitleAvailability],
    providers: dict[int, Provider],
    cutoff: datetime,
) -> Availability:
    tmdb_id, type_ = key
    if not rows:
        return Availability(tmdb_id=tmdb_id, type=type_, country=country)
    offers = []
    for r in rows:
        if r.provider_id == NO_PROVIDER_ID:
            continue
        p = providers.get(r.provider_id)
        offers.append(
            Offer(
                provider_id=r.provider_id,
                offer_type=r.offer_type,
                provider_name=p.name if p else None,
                logo_path=p.logo_path if p else None,
                display_priority=p.display_priority if p else None,
            )
        )
    fetched_at = _oldest(rows)
    return Availability(
        tmdb_id=tmdb_id,
        type=type_,
        country=country,
        offers=tuple(sorted(offers, key=_offer_sort_key)),
        link=next((r.link for r in rows if r.link), None),
        fetched_at=fetched_at,
        stale=fetched_at < cutoff,
    )
