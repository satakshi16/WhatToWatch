"""Catalog ingest end to end against a mocked TMDB producing a small catalog (P1-04)."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from app.db.models import NO_PROVIDER_ID, OfferType, Provider, Title, TitleAvailability, TitleType
from app.ingest.ingest_catalog import IngestOptions, IngestStats, ingest_catalog
from app.ingest.tmdb_client import TMDBAuthError, TMDBClient

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "tmdb"
TODAY = date(2026, 11, 3)
LANGUAGES = ("en", "hi", "ko", "ja", "es", "fr", "ta", "te", "ml")
PER_PAGE = 5
TOTAL_PAGES = 2
API = r"https://api\.themoviedb\.org/3"


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


MOVIE = fixture("movie_27205.json")
TV = fixture("tv_1429.json")


def fake_id(language: str, sort_by: str, recent: bool, page: int, index: int) -> int:
    """Deterministic IDs. `vote_count.desc` overlaps `popularity.desc` to exercise dedupe."""
    lang = LANGUAGES.index(language) + 1
    strategy = 2 if recent else 1  # vote_count.desc reuses popularity.desc IDs
    return lang * 10_000 + strategy * 1_000 + page * 100 + index


def language_of(tmdb_id: int) -> str:
    return LANGUAGES[tmdb_id // 10_000 - 1]


class FakeTMDB:
    """Discover / details / provider-list handlers; records calls."""

    def __init__(self) -> None:
        self.not_found: set[int] = set()
        self.failing: set[int] = set()
        self.unauthorized = False
        self.discover_calls: list[dict[str, str]] = []
        self.detail_calls: list[tuple[str, int]] = []

    def discover(self, request: httpx.Request, kind: str) -> httpx.Response:
        params = dict(request.url.params)
        self.discover_calls.append({"kind": kind, **params})
        page = int(params["page"])
        language = params["with_original_language"]
        recent = any(k.endswith(".gte") for k in params)
        results = [
            {"id": fake_id(language, params["sort_by"], recent, page, i)} for i in range(PER_PAGE)
        ]
        return httpx.Response(
            200, json={"page": page, "results": results, "total_pages": TOTAL_PAGES}
        )

    def details(self, request: httpx.Request, kind: str, tmdb_id: str) -> httpx.Response:
        id_ = int(tmdb_id)
        self.detail_calls.append((kind, id_))
        if self.unauthorized:
            return httpx.Response(401, json={"status_code": 7, "status_message": "Invalid key"})
        if id_ in self.not_found:
            return httpx.Response(404, json={"status_code": 34, "status_message": "Not found"})
        if id_ in self.failing:
            return httpx.Response(400, json={"status_message": "bad"})
        data = copy.deepcopy(MOVIE if kind == "movie" else TV)
        data["id"] = id_
        data["original_language"] = language_of(id_)
        data["title" if kind == "movie" else "name"] = f"{kind} {id_}"
        if id_ % 2:  # odd IDs: no US offers → marker row
            data["watch/providers"]["results"].pop("US", None)
        return httpx.Response(200, json=data)

    def providers(self, request: httpx.Request, kind: str) -> httpx.Response:
        region = request.url.params["watch_region"]
        shared = {"provider_id": 8, "provider_name": "Netflix", "display_priorities": {region: 2}}
        only = {"provider_id": 100 if kind == "movie" else 200, "provider_name": f"{kind} only"}
        return httpx.Response(200, json={"results": [shared, only]})


@pytest.fixture
def fake() -> Iterator[FakeTMDB]:
    tmdb = FakeTMDB()
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=rf"{API}/discover/(?P<kind>movie|tv)").mock(side_effect=tmdb.discover)
        router.get(url__regex=rf"{API}/watch/providers/(?P<kind>movie|tv)").mock(
            side_effect=tmdb.providers
        )
        # Late-bound so tests can swap `tmdb.details`.
        router.get(url__regex=rf"{API}/(?P<kind>movie|tv)/(?P<tmdb_id>\d+)").mock(
            side_effect=lambda request, **kw: tmdb.details(request, **kw)
        )
        yield tmdb


@pytest.fixture
async def tmdb() -> AsyncIterator[TMDBClient]:
    async def no_sleep(_: float) -> None:
        return None

    async with TMDBClient(api_key="test-key", max_attempts=2, sleep=no_sleep) as client:
        yield client


def options(**overrides: Any) -> IngestOptions:
    values: dict[str, Any] = {
        "languages": LANGUAGES,
        "types": (TitleType.MOVIE, TitleType.TV),
        "countries": ("US",),
        "pages": TOTAL_PAGES,
        "batch_size": 25,
    }
    values.update(overrides)
    return IngestOptions(**values)


def count(session: Session, model: type[Any]) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


# 9 languages x 2 types x (popularity/vote_count shared IDs + recent) x 2 pages x 5.
EXPECTED_TITLES = len(LANGUAGES) * 2 * 2 * TOTAL_PAGES * PER_PAGE


async def test_ingest_builds_catalog(
    fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine
) -> None:
    stats = await ingest_catalog(tmdb, migrated_engine, options(), today=TODAY)

    assert stats.discovered == stats.selected == EXPECTED_TITLES
    assert stats.inserted == EXPECTED_TITLES
    assert (stats.updated, stats.not_found, stats.failed) == (0, 0, 0)
    assert stats.discover_requests == len(LANGUAGES) * 2 * 3 * TOTAL_PAGES

    with Session(migrated_engine) as session:
        assert count(session, Title) == EXPECTED_TITLES
        pairs = set(session.execute(select(Title.original_language, Title.type)).all())
        assert pairs == {(lang, t) for lang in LANGUAGES for t in TitleType}

        title = session.get(Title, (fake_id("ko", "popularity.desc", False, 1, 0), TitleType.TV))
        assert title is not None
        assert title.original_language == "ko"
        assert title.directors
        assert title.genres
        assert title.certification["US"] == "TV-MA"

        marker = session.scalars(
            select(TitleAvailability).where(
                TitleAvailability.tmdb_id == title.tmdb_id + 1,
                TitleAvailability.type == TitleType.TV,
            )
        ).all()
        assert [(m.provider_id, m.offer_type) for m in marker] == [(NO_PROVIDER_ID, OfferType.NONE)]
        offers = session.scalars(
            select(TitleAvailability).where(
                TitleAvailability.tmdb_id == title.tmdb_id, TitleAvailability.type == TitleType.TV
            )
        ).all()
        assert {(o.provider_id, o.offer_type) for o in offers} >= {(15, OfferType.FLATRATE)}

        providers = {
            p.provider_id: p
            for p in session.scalars(select(Provider).where(Provider.country == "US"))
        }
        assert set(providers) == {8, 100, 200}
        assert providers[8].display_priority == 2

    recent = [c for c in fake.discover_calls if "primary_release_date.gte" in c]
    assert recent
    assert recent[0]["primary_release_date.gte"] == "2025-11-03"
    assert recent[0]["primary_release_date.lte"] == "2026-11-03"


async def test_second_run_updates_without_duplicates(
    fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine
) -> None:
    await ingest_catalog(tmdb, migrated_engine, options(), today=TODAY)
    with Session(migrated_engine) as session:
        first = dict(session.execute(select(Title.tmdb_id, Title.updated_at)).all())
        availability = count(session, TitleAvailability)
        providers = count(session, Provider)

    stats = await ingest_catalog(tmdb, migrated_engine, options(), today=TODAY)

    assert (stats.inserted, stats.updated) == (0, EXPECTED_TITLES)
    with Session(migrated_engine) as session:
        assert count(session, Title) == EXPECTED_TITLES
        assert count(session, TitleAvailability) == availability
        assert count(session, Provider) == providers
        second = dict(session.execute(select(Title.tmdb_id, Title.updated_at)).all())
    assert all(second[k] > first[k] for k in first)


async def test_availability_replaced_on_rerun(
    fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine
) -> None:
    opts = options(languages=("en",), types=(TitleType.MOVIE,), pages=1)
    await ingest_catalog(tmdb, migrated_engine, opts, today=TODAY)
    even_id = fake_id("en", "popularity.desc", False, 1, 0)

    # The title loses its US offers: its rows collapse to the marker row.
    original = fake.details

    def no_offers(request: httpx.Request, kind: str, tmdb_id: str) -> httpx.Response:
        response = original(request, kind, tmdb_id)
        data = response.json()
        if response.status_code == 200:
            data["watch/providers"]["results"].pop("US", None)
        return httpx.Response(response.status_code, json=data)

    fake.details = no_offers  # type: ignore[method-assign]
    await ingest_catalog(tmdb, migrated_engine, opts, today=TODAY)

    with Session(migrated_engine) as session:
        rows = session.scalars(
            select(TitleAvailability).where(
                TitleAvailability.tmdb_id == even_id, TitleAvailability.type == TitleType.MOVIE
            )
        ).all()
        assert [(r.provider_id, r.offer_type) for r in rows] == [(NO_PROVIDER_ID, OfferType.NONE)]


async def test_limit_covers_all_languages_and_types(
    fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine
) -> None:
    stats = await ingest_catalog(tmdb, migrated_engine, options(pages=10, limit=40), today=TODAY)

    assert stats.selected == stats.inserted == 40
    # One discover round was enough for 40 titles; no further pages fetched.
    assert {c["page"] for c in fake.discover_calls} == {"1"}
    assert len(fake.detail_calls) == 40
    with Session(migrated_engine) as session:
        assert count(session, Title) == 40
        pairs = set(session.execute(select(Title.original_language, Title.type)).all())
        assert pairs == {(lang, t) for lang in LANGUAGES for t in TitleType}


async def test_missing_and_failing_titles_are_skipped(
    fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine
) -> None:
    gone = fake_id("hi", "popularity.desc", False, 1, 0)
    broken = fake_id("ta", "popularity.desc", False, 1, 1)
    fake.not_found.add(gone)
    fake.failing.add(broken)

    stats = await ingest_catalog(tmdb, migrated_engine, options(pages=1), today=TODAY)

    # Each ID is both a movie and a TV title in the fake.
    assert (stats.not_found, stats.failed) == (2, 2)
    assert stats.inserted == stats.selected - 4
    with Session(migrated_engine) as session:
        ids = set(session.scalars(select(Title.tmdb_id)))
    assert gone not in ids
    assert broken not in ids


async def test_resume_skips_fresh_titles(
    fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine
) -> None:
    await ingest_catalog(tmdb, migrated_engine, options(pages=1, limit=20), today=TODAY)
    fake.detail_calls.clear()

    stats = await ingest_catalog(
        tmdb, migrated_engine, options(pages=1, limit=30, resume=True), today=TODAY
    )

    assert stats.skipped_fresh == 20
    assert stats.inserted == 10
    assert len(fake.detail_calls) == 10


async def test_auth_error_aborts(fake: FakeTMDB, tmdb: TMDBClient, migrated_engine: Engine) -> None:
    fake.unauthorized = True
    with pytest.raises(TMDBAuthError):
        await ingest_catalog(tmdb, migrated_engine, options(pages=1, limit=5), today=TODAY)


# --- CLI (`make ingest`) ------------------------------------------------------------


@pytest.fixture
def cli_settings(settings: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from app.ingest import ingest_catalog as module

    configured = settings.model_copy(update={"tmdb_read_token": SecretStr("test-token")})
    monkeypatch.setattr(module, "get_settings", lambda: configured)
    return configured


def test_main_without_schema_asks_for_migrate(
    cli_settings: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.ingest import ingest_catalog as module

    assert module.main([]) == 2
    assert "run `make migrate` first" in capsys.readouterr().err


def test_main_without_tmdb_credentials(
    settings: Any,
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app.ingest import ingest_catalog as module

    monkeypatch.setattr(module, "get_settings", lambda: settings)
    assert module.main([]) == 2
    assert "TMDB is not configured" in capsys.readouterr().err


def test_main_reports_rejected_credentials(
    cli_settings: Any,
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app.ingest import ingest_catalog as module

    async def rejected(*_args: object) -> IngestStats:
        raise TMDBAuthError(401, "/discover/movie")

    monkeypatch.setattr(module, "ingest_catalog", rejected)
    assert module.main(["--limit", "1"]) == 1
    assert "TMDB rejected the credentials (401)" in capsys.readouterr().out


def test_main_prints_summary(
    cli_settings: Any,
    migrated_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app.ingest import ingest_catalog as module

    seen: list[IngestOptions] = []

    async def fake_ingest(_tmdb: object, _engine: object, options: IngestOptions) -> IngestStats:
        seen.append(options)
        return IngestStats(inserted=3, updated=2, skipped_fresh=1, availability_rows=7, providers=4)

    monkeypatch.setattr(module, "ingest_catalog", fake_ingest)
    assert module.main(["--limit", "5", "--languages", "ko", "--pages", "1"]) == 0
    [options] = seen
    assert (options.limit, options.languages, options.pages) == (5, ("ko",), 1)
    out = capsys.readouterr().out
    assert "Ingested 5 titles (3 new, 2 updated, 1 skipped as fresh" in out
    assert "7 availability rows, 4 providers" in out
