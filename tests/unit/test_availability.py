"""P1-10 availability service: TTL cache, D-008 marker rows, fetch budget, concurrency,
TMDB failures, FR-22 offer-type filtering. TMDB is a fake client; the clock is injected."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import NO_PROVIDER_ID, OfferType, Provider, TitleAvailability, TitleType
from app.ingest.tmdb_client import (
    TMDBConfigError,
    TMDBError,
    TMDBNotFoundError,
    TMDBUnavailableError,
)
from app.ingest.tmdb_models import WatchProviders
from app.recsys.availability import (
    Availability,
    AvailabilityService,
    Offer,
    is_streamable,
    streamable_offers,
)
from tests.catalog_factory import add_catalog, tmdb_title

M, T = TitleType.MOVIE, TitleType.TV
TTL = timedelta(days=7)
T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
NETFLIX, PRIME, TUBI, APPLE = 8, 9, 73, 2


def _offer(provider_id: int, name: str = "P", priority: int = 1) -> dict[str, Any]:
    return {
        "provider_id": provider_id,
        "provider_name": name,
        "logo_path": f"/{provider_id}.png",
        "display_priority": priority,
    }


def watch(country: str = "US", link: str | None = None, **offers: list[int]) -> WatchProviders:
    """WatchProviders with one country; `offers` maps offer type → provider IDs."""
    body: dict[str, Any] = {"link": link or f"https://www.themoviedb.org/watch?locale={country}"}
    body.update({kind: [_offer(pid) for pid in ids] for kind, ids in offers.items()})
    return WatchProviders.model_validate({"results": {country: body}})


class FakeTMDB:
    """Async-context-manager TMDB stand-in: `watch_providers` from a dict, records calls."""

    def __init__(self) -> None:
        self.responses: dict[tuple[int, TitleType], WatchProviders | Exception] = {}
        self.calls: list[tuple[int, TitleType]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def watch_providers(self, type_: TitleType, tmdb_id: int) -> WatchProviders:
        self.calls.append((tmdb_id, type_))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0)
            response = self.responses.get((tmdb_id, type_), WatchProviders())
            if isinstance(response, Exception):
                raise response
            return response
        finally:
            self.in_flight -= 1


class Clock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


@pytest.fixture
def session(migrated_engine: Engine) -> Iterator[Session]:
    with Session(migrated_engine) as s:
        add_catalog(s, *(tmdb_title(i, M) for i in range(1, 11)), tmdb_title(1, T))
        s.add_all(
            [
                Provider(provider_id=NETFLIX, country="US", name="Netflix", display_priority=1),
                Provider(provider_id=PRIME, country="US", name="Prime Video", display_priority=2),
                Provider(provider_id=TUBI, country="US", name="Tubi", display_priority=5),
                Provider(provider_id=APPLE, country="US", name="Apple TV", display_priority=9),
            ]
        )
        s.commit()
        yield s


@pytest.fixture
def tmdb() -> FakeTMDB:
    return FakeTMDB()


@pytest.fixture
def clock() -> Clock:
    return Clock()


def service(session: Session, tmdb: FakeTMDB, clock: Clock, **kwargs: Any) -> AvailabilityService:
    return AvailabilityService(
        session,
        lambda: tmdb,  # type: ignore[arg-type, return-value]
        ttl=TTL,
        clock=clock,
        **kwargs,
    )


def rows(session: Session, tmdb_id: int, type_: TitleType = M) -> list[TitleAvailability]:
    stmt = select(TitleAvailability).where(
        TitleAvailability.tmdb_id == tmdb_id, TitleAvailability.type == type_
    )
    return list(session.scalars(stmt))


# --- get / caching -------------------------------------------------------------


async def test_miss_fetches_stores_and_returns_provider_info(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    link = "https://www.themoviedb.org/movie/1/watch?locale=US"
    tmdb.responses[(1, M)] = watch(link=link, flatrate=[PRIME, NETFLIX], rent=[APPLE])

    av = await service(session, tmdb, clock).get(1, M, "US")

    assert tmdb.calls == [(1, M)]
    assert av.checked
    assert not av.stale
    assert av.fetched_at == T0
    assert av.link == link
    # Sorted by the country's display priority; names/logos from `providers`.
    assert [(o.provider_name, o.offer_type) for o in av.offers] == [
        ("Netflix", OfferType.FLATRATE),
        ("Prime Video", OfferType.FLATRATE),
        ("Apple TV", OfferType.RENT),
    ]
    assert {(r.provider_id, r.offer_type) for r in rows(session, 1)} == {
        (NETFLIX, OfferType.FLATRATE),
        (PRIME, OfferType.FLATRATE),
        (APPLE, OfferType.RENT),
    }


async def test_second_lookup_within_ttl_makes_no_network_calls(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = watch(flatrate=[NETFLIX])
    await service(session, tmdb, clock).get(1, M, "US")

    clock.advance(TTL)  # exactly at the TTL is still fresh
    # A new service (new request) still hits the DB cache, not TMDB.
    av = await service(session, tmdb, clock).get(1, M, "US")

    assert tmdb.calls == [(1, M)]
    assert av.fetched_at == T0
    assert [o.provider_id for o in av.offers] == [NETFLIX]


async def test_stale_rows_are_refetched_and_replaced(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = watch(flatrate=[NETFLIX, PRIME])
    svc = service(session, tmdb, clock)
    await svc.get(1, M, "US")

    clock.advance(TTL + timedelta(seconds=1))
    tmdb.responses[(1, M)] = watch(flatrate=[PRIME])  # Netflix dropped it
    av = await svc.get(1, M, "US")

    assert tmdb.calls == [(1, M), (1, M)]
    assert av.fetched_at == clock.now
    assert not av.stale
    assert [o.provider_id for o in av.offers] == [PRIME]
    assert [r.provider_id for r in rows(session, 1)] == [PRIME]


async def test_no_offers_stores_marker_and_is_not_refetched_within_ttl(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = watch(country="IN", flatrate=[NETFLIX])  # nothing in US
    svc = service(session, tmdb, clock)

    av = await svc.get(1, M, "US")
    assert av.checked
    assert av.offers == ()
    assert av.link is None
    marker = rows(session, 1)
    assert [(r.provider_id, r.offer_type, r.country) for r in marker] == [
        (NO_PROVIDER_ID, OfferType.NONE, "US")
    ]

    clock.advance(timedelta(days=6))
    av = await svc.get(1, M, "US")
    assert tmdb.calls == [(1, M)]
    assert av.checked
    assert av.offers == ()

    clock.advance(timedelta(days=2))  # past the TTL → re-checked
    await svc.get(1, M, "US")
    assert tmdb.calls == [(1, M), (1, M)]


async def test_tmdb_404_counts_as_no_offers(session: Session, tmdb: FakeTMDB, clock: Clock) -> None:
    tmdb.responses[(1, M)] = TMDBNotFoundError(404, "/movie/1/watch/providers", "not found")

    av = await service(session, tmdb, clock).get(1, M, "US")

    assert av.checked
    assert av.offers == ()
    assert [r.offer_type for r in rows(session, 1)] == [OfferType.NONE]


async def test_only_requested_country_is_stored_and_country_is_normalized(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = WatchProviders.model_validate(
        {
            "results": {
                "US": {"flatrate": [_offer(NETFLIX)]},
                "IN": {"flatrate": [_offer(PRIME)]},
            }
        }
    )

    av = await service(session, tmdb, clock).get(1, M, "us")

    assert av.country == "US"
    assert {r.country for r in rows(session, 1)} == {"US"}


async def test_unknown_provider_has_no_name(session: Session, tmdb: FakeTMDB, clock: Clock) -> None:
    tmdb.responses[(1, M)] = watch(flatrate=[999])

    av = await service(session, tmdb, clock).get(1, M, "US")

    assert av.offers == (Offer(provider_id=999, offer_type=OfferType.FLATRATE),)


async def test_ingested_rows_are_used_without_fetching(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    session.add(
        TitleAvailability(
            tmdb_id=1,
            type=T,
            country="US",
            provider_id=NETFLIX,
            offer_type=OfferType.FLATRATE,
            link="https://tmdb/tv/1",
            fetched_at=T0 - timedelta(days=3),
        )
    )
    session.commit()

    av = await service(session, tmdb, clock).get(1, T, "US")

    assert tmdb.calls == []
    assert av.link == "https://tmdb/tv/1"
    assert av.fetched_at == T0 - timedelta(days=3)


# --- failures ------------------------------------------------------------------


async def test_tmdb_failure_without_cache_reports_unchecked_and_retries_next_time(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = TMDBUnavailableError(
        "/movie/1/watch/providers", attempts=4, last_status=503
    )
    svc = service(session, tmdb, clock)

    av = await svc.get(1, M, "US")
    assert not av.checked
    assert av.offers == ()
    assert rows(session, 1) == []
    assert not is_streamable(av, [NETFLIX], include_rent_buy=True)

    tmdb.responses[(1, M)] = watch(flatrate=[NETFLIX])
    av = await svc.get(1, M, "US")
    assert av.checked
    assert len(tmdb.calls) == 2


async def test_tmdb_failure_serves_stale_rows(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = watch(flatrate=[NETFLIX])
    svc = service(session, tmdb, clock)
    await svc.get(1, M, "US")

    clock.advance(TTL * 2)
    tmdb.responses[(1, M)] = TMDBUnavailableError(
        "/movie/1/watch/providers", attempts=4, last_status=503
    )
    av = await svc.get(1, M, "US")

    assert av.stale
    assert av.fetched_at == T0
    assert [o.provider_id for o in av.offers] == [NETFLIX]
    assert is_streamable(av, [NETFLIX], include_rent_buy=False)


async def test_tmdb_client_unavailable_degrades_to_cache(session: Session, clock: Clock) -> None:
    def factory() -> Any:
        raise TMDBConfigError("Set TMDB_READ_TOKEN or TMDB_API_KEY")

    svc = AvailabilityService(session, factory, ttl=TTL, clock=clock)
    av = await svc.get(1, M, "US")

    assert not av.checked


async def test_non_tmdb_errors_propagate(session: Session, tmdb: FakeTMDB, clock: Clock) -> None:
    tmdb.responses[(1, M)] = RuntimeError("bug")

    with pytest.raises(RuntimeError):
        await service(session, tmdb, clock).get(1, M, "US")


# --- bulk_get ------------------------------------------------------------------


async def test_bulk_get_fetches_only_missing_and_keeps_order(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(2, M)] = watch(flatrate=[NETFLIX])
    svc = service(session, tmdb, clock)
    await svc.get(2, M, "US")
    tmdb.calls.clear()

    keys = [(3, M), (2, M), (1, T), (3, M)]
    result = await svc.bulk_get(keys, "US")

    assert list(result) == [(3, M), (2, M), (1, T)]
    assert sorted(tmdb.calls) == [(1, T), (3, M)]
    assert all(av.checked for av in result.values())
    assert result[(2, M)].key == (2, M)


async def test_bulk_get_empty(session: Session, tmdb: FakeTMDB, clock: Clock) -> None:
    assert await service(session, tmdb, clock).bulk_get([], "US") == {}
    assert tmdb.calls == []


async def test_fetch_budget_follows_input_order(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    svc = service(session, tmdb, clock, max_fetches=3)
    keys = [(i, M) for i in (5, 4, 3, 2, 1)]

    result = await svc.bulk_get(keys, "US")

    assert tmdb.calls == [(5, M), (4, M), (3, M)]
    assert [result[k].checked for k in keys] == [True, True, True, False, False]

    # Per-call override.
    tmdb.calls.clear()
    await svc.bulk_get(keys, "US", max_fetches=0)
    assert tmdb.calls == []
    await svc.bulk_get(keys, "US", max_fetches=10)
    assert tmdb.calls == [(2, M), (1, M)]


async def test_titles_not_in_catalog_are_not_fetched(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    result = await service(session, tmdb, clock).bulk_get([(999, M), (1, M)], "US")

    assert tmdb.calls == [(1, M)]
    assert not result[(999, M)].checked


async def test_fetches_run_concurrently_under_the_limit(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    keys = [(i, M) for i in range(1, 11)]

    await service(session, tmdb, clock, concurrency=3).bulk_get(keys, "US")

    assert len(tmdb.calls) == 10
    assert 1 < tmdb.max_in_flight <= 3


async def test_one_failure_does_not_block_the_others(
    session: Session, tmdb: FakeTMDB, clock: Clock
) -> None:
    tmdb.responses[(1, M)] = TMDBError("boom")
    tmdb.responses[(2, M)] = watch(flatrate=[NETFLIX])

    result = await service(session, tmdb, clock).bulk_get([(1, M), (2, M)], "US")

    assert not result[(1, M)].checked
    assert [o.provider_id for o in result[(2, M)].offers] == [NETFLIX]


def test_from_settings(session: Session, tmdb: FakeTMDB) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        availability_ttl_days=3,
        availability_max_fetches_per_request=5,
        tmdb_max_concurrency=4,
    )
    svc = AvailabilityService.from_settings(session, lambda: tmdb, settings)  # type: ignore[arg-type, return-value]

    assert svc._ttl == timedelta(days=3)
    assert svc._max_fetches == 5
    assert svc._concurrency == 4


def test_concurrency_must_be_positive(session: Session, tmdb: FakeTMDB) -> None:
    with pytest.raises(ValueError, match="concurrency"):
        AvailabilityService(session, lambda: tmdb, ttl=TTL, concurrency=0)  # type: ignore[arg-type, return-value]


# --- FR-22 filtering -----------------------------------------------------------


def _av(*offers: tuple[int, OfferType]) -> Availability:
    return Availability(
        tmdb_id=1,
        type=M,
        country="US",
        offers=tuple(Offer(pid, kind) for pid, kind in offers),
        fetched_at=T0,
    )


@pytest.mark.parametrize("kind", [OfferType.FLATRATE, OfferType.FREE, OfferType.ADS])
def test_stream_offer_types_always_count(kind: OfferType) -> None:
    av = _av((NETFLIX, kind))
    assert is_streamable(av, [NETFLIX], include_rent_buy=False)
    assert is_streamable(av, [NETFLIX], include_rent_buy=True)


@pytest.mark.parametrize("kind", [OfferType.RENT, OfferType.BUY])
def test_rent_buy_only_counts_when_included(kind: OfferType) -> None:
    av = _av((APPLE, kind))
    assert not is_streamable(av, [APPLE], include_rent_buy=False)
    assert is_streamable(av, [APPLE], include_rent_buy=True)


def test_offer_must_be_on_a_user_platform() -> None:
    av = _av((NETFLIX, OfferType.FLATRATE), (APPLE, OfferType.RENT))
    assert not is_streamable(av, [PRIME], include_rent_buy=True)
    assert not is_streamable(av, [], include_rent_buy=True)


def test_streamable_offers_filters_by_platform_and_type() -> None:
    av = _av(
        (NETFLIX, OfferType.FLATRATE),
        (PRIME, OfferType.RENT),
        (PRIME, OfferType.ADS),
        (APPLE, OfferType.BUY),
    )
    picked = streamable_offers(av, {NETFLIX, PRIME}, include_rent_buy=False)
    assert [(o.provider_id, o.offer_type) for o in picked] == [
        (NETFLIX, OfferType.FLATRATE),
        (PRIME, OfferType.ADS),
    ]


def test_unchecked_and_marker_only_titles_are_not_streamable() -> None:
    unchecked = Availability(tmdb_id=1, type=M, country="US")
    assert not unchecked.checked
    assert not is_streamable(unchecked, [NETFLIX], include_rent_buy=True)
    assert not is_streamable(_av(), [NETFLIX], include_rent_buy=True)
