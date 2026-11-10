"""Catalog ingest CLI: TMDB discover + details → `titles`, `title_availability`, `providers`.

Usage:
    python -m app.ingest.ingest_catalog [--languages en hi ...] [--types movie tv]
        [--pages N] [--countries US ...] [--limit N] [--resume [--fresh-hours H]]

Flow (P1-04):
1. Discover: for each language x type, `/discover` sorted by `popularity.desc` and
   `vote_count.desc`, plus releases from the last 12 months, `--pages` pages each.
   Pages are fetched round by round (page 1 of every query, then page 2, ...), so
   `--limit` stops early. Results are deduped and interleaved across language x type,
   so even a small `--limit` covers every language and type.
2. Details: fetched concurrently (the client's semaphore caps in-flight requests) with
   keywords, credits, watch providers, certifications and translations appended.
3. Upsert by PK `(tmdb_id, type)` and replace `title_availability` for the seed
   countries, one commit per batch. Re-running updates rows (and `updated_at`) and
   never duplicates them. `--resume` skips titles updated within `--fresh-hours`, so an
   interrupted run continues where it stopped.
4. Provider lists for the seed countries are refreshed into `providers`.

Titles TMDB no longer has (404) are skipped; other per-title TMDB errors are logged
and counted, and the run continues. Auth/config errors abort the run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from itertools import zip_longest

from sqlalchemy import Engine, delete, inspect, select, tuple_
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.base import utcnow
from app.db.models import NO_PROVIDER_ID, OfferType, Title, TitleAvailability, TitleType
from app.db.session import create_db_engine
from app.ingest.genres import canonical_genres
from app.ingest.providers import fetch_provider_list, replace_providers
from app.ingest.tmdb_client import (
    TMDBAuthError,
    TMDBClient,
    TMDBConfigError,
    TMDBError,
    TMDBNotFoundError,
)
from app.ingest.tmdb_models import CountryProviders, MovieDetails, TVDetails

log = get_logger(__name__)

TitleKey = tuple[int, TitleType]
Details = MovieDetails | TVDetails

DETAILS_LANGUAGE = "en-US"
TOP_CAST = 3
RECENT_DAYS = 365
# Theatrical release: the certification TMDB treats as canonical for movies.
THEATRICAL_RELEASE_TYPE = 3


@dataclass(frozen=True)
class DiscoverQuery:
    """One paged discover query (a language x type x strategy)."""

    type: TitleType
    language: str
    sort_by: str
    recent: bool = False


@dataclass(frozen=True)
class IngestOptions:
    languages: tuple[str, ...]
    types: tuple[TitleType, ...]
    countries: tuple[str, ...]
    pages: int
    limit: int | None = None
    resume: bool = False
    fresh_hours: float = 24.0
    batch_size: int = 100


@dataclass
class IngestStats:
    discover_requests: int = 0
    discovered: int = 0
    selected: int = 0
    skipped_fresh: int = 0
    inserted: int = 0
    updated: int = 0
    not_found: int = 0
    failed: int = 0
    availability_rows: int = 0
    providers: int = 0
    duration_s: float = 0.0
    by_language: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)


# --- Discovery -----------------------------------------------------------------


def discover_queries(options: IngestOptions) -> list[DiscoverQuery]:
    return [
        query
        for language in options.languages
        for type_ in options.types
        for query in (
            DiscoverQuery(type_, language, "popularity.desc"),
            DiscoverQuery(type_, language, "vote_count.desc"),
            DiscoverQuery(type_, language, "popularity.desc", recent=True),
        )
    ]


async def discover_titles(
    tmdb: TMDBClient, options: IngestOptions, stats: IngestStats, *, today: date
) -> list[TitleKey]:
    """Deduped title keys, interleaved across language x type buckets, truncated to `limit`."""
    queries = discover_queries(options)
    buckets: dict[tuple[str, TitleType], list[TitleKey]] = {
        (q.language, q.type): [] for q in queries
    }
    seen: set[TitleKey] = set()
    exhausted: set[DiscoverQuery] = set()
    since = today - timedelta(days=RECENT_DAYS)

    for page in range(1, options.pages + 1):
        active = [q for q in queries if q not in exhausted]
        if not active:
            break
        results = await asyncio.gather(
            *(
                tmdb.discover(
                    q.type,
                    original_language=q.language,
                    sort_by=q.sort_by,
                    date_gte=since if q.recent else None,
                    date_lte=today if q.recent else None,
                    page=page,
                )
                for q in active
            )
        )
        stats.discover_requests += len(active)
        for query, result in zip(active, results, strict=True):
            if page >= result.total_pages or not result.results:
                exhausted.add(query)
            for item in result.results:
                key = (item.id, query.type)
                if key not in seen:
                    seen.add(key)
                    buckets[(query.language, query.type)].append(key)
        log.info("ingest_discover_page", page=page, queries=len(active), unique=len(seen))
        if options.limit is not None and len(seen) >= options.limit:
            break

    stats.discovered = len(seen)
    keys = _interleave(buckets.values())
    return keys[: options.limit] if options.limit is not None else keys


def _interleave(lists: Iterable[list[TitleKey]]) -> list[TitleKey]:
    return [key for group in zip_longest(*lists) for key in group if key is not None]


# --- Mapping TMDB details → rows ----------------------------------------------


def title_key(details: Details) -> TitleKey:
    return details.id, TitleType.MOVIE if isinstance(details, MovieDetails) else TitleType.TV


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def clean_keywords(details: Details) -> list[str]:
    # `str.strip()` also removes Unicode whitespace such as the NBSP TMDB sometimes sends.
    raw = details.keywords.keywords if details.keywords else []
    return _dedupe(k.name.strip() for k in raw)


def directors(details: Details) -> list[str]:
    crew = details.credits.crew if details.credits else []
    if isinstance(details, MovieDetails):
        return _dedupe(c.name for c in crew if c.job == "Director")
    if details.created_by:
        return _dedupe(c.name for c in details.created_by)
    return _dedupe(c.name for c in crew if c.job == "Series Director")


def top_cast(details: Details) -> list[str]:
    cast = sorted(details.credits.cast if details.credits else [], key=lambda c: c.order)
    return _dedupe(c.name for c in cast)[:TOP_CAST]


def runtime(details: Details) -> int | None:
    """Movie minutes, or the average TV episode minutes."""
    if isinstance(details, MovieDetails):
        return details.runtime or None
    run_times = [m for m in details.episode_run_time if m > 0]
    if run_times:
        return round(sum(run_times) / len(run_times))
    last = details.last_episode_to_air
    return (last.runtime or None) if last else None


def certifications(details: Details, countries: Sequence[str]) -> dict[str, str]:
    """Country → certification for `countries`; countries without one are left out."""
    wanted = set(countries)
    out: dict[str, str] = {}
    if isinstance(details, MovieDetails):
        for country in details.release_dates.results if details.release_dates else []:
            if country.iso_3166_1 not in wanted:
                continue
            certs = [
                (r.type == THEATRICAL_RELEASE_TYPE, r.certification.strip())
                for r in country.release_dates
                if r.certification.strip()
            ]
            if certs:
                # Prefer the theatrical certification, else the first one given.
                out[country.iso_3166_1] = max(certs, key=lambda c: c[0])[1]
    else:
        for rating in details.content_ratings.results if details.content_ratings else []:
            if rating.iso_3166_1 in wanted and rating.rating.strip():
                out[rating.iso_3166_1] = rating.rating.strip()
    return out


def localized_titles(details: Details) -> dict[str, str]:
    """`<lang>-<COUNTRY>` → translated title, for translations that have one."""
    translations = details.translations.translations if details.translations else []
    return {
        f"{t.iso_639_1}-{t.iso_3166_1}": t.data.title.strip()
        for t in translations
        if t.data.title and t.data.title.strip()
    }


def _overview(details: Details) -> str | None:
    """English overview, else the original-language translation's overview."""
    if details.overview and details.overview.strip():
        return details.overview.strip()
    translations = details.translations.translations if details.translations else []
    for t in translations:
        if t.iso_639_1 == details.original_language and t.data.overview:
            return t.data.overview.strip()
    return None


def title_values(details: Details, countries: Sequence[str]) -> dict[str, object]:
    """Column values for `titles` (everything except `updated_at`)."""
    if isinstance(details, MovieDetails):
        type_, name, original = TitleType.MOVIE, details.title, details.original_title
        released, seasons = details.release_date, None
    else:
        type_, name, original = TitleType.TV, details.name, details.original_name
        released, seasons = details.first_air_date, details.number_of_seasons
    return {
        "tmdb_id": details.id,
        "type": type_,
        "title": name or original,
        "original_title": original or name,
        "original_language": details.original_language,
        "overview": _overview(details),
        "genres": canonical_genres(details.genre_ids, details.original_language),
        "keywords": clean_keywords(details),
        "directors": directors(details),
        "top_cast": top_cast(details),
        "release_date": released,
        "runtime": runtime(details),
        "seasons": seasons,
        "vote_avg": details.vote_average,
        "vote_count": details.vote_count,
        "popularity": details.popularity,
        "poster_path": details.poster_path,
        "adult": details.adult,
        "certification": certifications(details, countries),
        "localized_titles": localized_titles(details),
    }


def availability_rows(
    details: Details, countries: Sequence[str], fetched_at: datetime
) -> list[TitleAvailability]:
    """Offers per country; a country with no offers gets one `none` marker row (D-008).

    Returns `[]` if watch providers were not appended to the details response.
    """
    if details.watch_providers is None:
        return []
    tmdb_id, type_ = title_key(details)
    rows: list[TitleAvailability] = []
    for country in countries:
        offers = details.watch_providers.results.get(country)
        rows.extend(country_availability_rows(tmdb_id, type_, country, offers, fetched_at))
    return rows


def country_availability_rows(
    tmdb_id: int,
    type_: TitleType,
    country: str,
    offers: CountryProviders | None,
    fetched_at: datetime,
) -> list[TitleAvailability]:
    """`title_availability` rows for one title in one country (D-008 marker if no offers)."""
    pairs = {(p.provider_id, offer): None for offer, p in offers.offers()} if offers else {}
    link = offers.link if offers else None
    if not pairs:
        pairs = {(NO_PROVIDER_ID, OfferType.NONE): None}
    return [
        TitleAvailability(
            tmdb_id=tmdb_id,
            type=type_,
            country=country,
            provider_id=provider_id,
            offer_type=offer_type,
            link=link,
            fetched_at=fetched_at,
        )
        for provider_id, offer_type in pairs
    ]


# --- Persistence ---------------------------------------------------------------


def fresh_keys(session: Session, keys: Sequence[TitleKey], since: datetime) -> set[TitleKey]:
    """Keys among `keys` whose title row was updated at or after `since`."""
    found: set[TitleKey] = set()
    for chunk in _chunks(keys, 500):
        stmt = select(Title.tmdb_id, Title.type).where(
            tuple_(Title.tmdb_id, Title.type).in_(chunk), Title.updated_at >= since
        )
        found.update((tmdb_id, type_) for tmdb_id, type_ in session.execute(stmt))
    return found


def upsert_batch(
    session: Session,
    batch: Sequence[Details],
    countries: Sequence[str],
    stats: IngestStats,
) -> None:
    """Upsert titles and replace their availability for `countries`; commits once."""
    now = utcnow()
    values = [title_values(d, countries) for d in batch]
    keys = [title_key(d) for d in batch]
    existing = {
        (tmdb_id, type_)
        for tmdb_id, type_ in session.execute(
            select(Title.tmdb_id, Title.type).where(tuple_(Title.tmdb_id, Title.type).in_(keys))
        )
    }
    for key, row in zip(keys, values, strict=True):
        # Explicit `updated_at` so a re-ingest bumps it even when nothing else changed.
        session.merge(Title(**row, updated_at=now))
        if key in existing:
            stats.updated += 1
        else:
            stats.inserted += 1
        language = str(row["original_language"])
        stats.by_language[language] = stats.by_language.get(language, 0) + 1
        stats.by_type[key[1].value] = stats.by_type.get(key[1].value, 0) + 1
    session.flush()

    session.execute(
        delete(TitleAvailability).where(
            tuple_(TitleAvailability.tmdb_id, TitleAvailability.type).in_(keys),
            TitleAvailability.country.in_(countries),
        )
    )
    for details in batch:
        rows = availability_rows(details, countries, now)
        session.add_all(rows)
        stats.availability_rows += len(rows)
    session.commit()


async def refresh_providers(
    tmdb: TMDBClient, session: Session, countries: Sequence[str], stats: IngestStats
) -> None:
    """Replace `providers` for each country with the union of TMDB's movie + TV lists."""
    for country in countries:
        rows = await fetch_provider_list(tmdb, country)
        replace_providers(session, country, rows)
        session.commit()
        stats.providers += len(rows)
        log.info("ingest_providers", country=country, providers=len(rows))


def _chunks[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --- Orchestration -------------------------------------------------------------


async def _fetch_details(tmdb: TMDBClient, key: TitleKey) -> Details | BaseException:
    tmdb_id, type_ = key
    try:
        return await tmdb.details(type_, tmdb_id, language=DETAILS_LANGUAGE)
    except (TMDBAuthError, TMDBConfigError):
        raise
    except TMDBError as exc:
        return exc


def existing_keys(session: Session, keys: Sequence[TitleKey]) -> set[TitleKey]:
    """Keys among `keys` that have a `titles` row."""
    found: set[TitleKey] = set()
    for chunk in _chunks(list(dict.fromkeys(keys)), 500):
        stmt = select(Title.tmdb_id, Title.type).where(tuple_(Title.tmdb_id, Title.type).in_(chunk))
        found.update((tmdb_id, type_) for tmdb_id, type_ in session.execute(stmt))
    return found


async def insert_missing_titles(
    tmdb: TMDBClient,
    session: Session,
    keys: Sequence[TitleKey],
    countries: Sequence[str],
) -> tuple[list[TitleKey], list[TitleKey]]:
    """Fetch-on-demand (§8.1): insert titles among `keys` that are not in the catalog.

    Details (with availability for `countries`) are fetched concurrently and upserted
    like an ingest batch (one commit). Returns `(inserted, not_found)`. All or nothing:
    if TMDB does not know some title (404), nothing is inserted and `inserted` is empty;
    any other TMDB error is raised, also with nothing inserted.
    """
    have = existing_keys(session, keys)
    missing = [k for k in dict.fromkeys(keys) if k not in have]
    if not missing:
        return [], []
    results = await asyncio.gather(*(_fetch_details(tmdb, k) for k in missing))
    batch: list[Details] = []
    inserted: list[TitleKey] = []
    not_found: list[TitleKey] = []
    for key, result in zip(missing, results, strict=True):
        if isinstance(result, TMDBNotFoundError):
            not_found.append(key)
        elif isinstance(result, BaseException):
            raise result
        else:
            batch.append(result)
            inserted.append(key)
    if not_found:
        return [], not_found
    if batch:
        upsert_batch(session, batch, countries, IngestStats())
    log.info("titles_fetched_on_demand", inserted=len(inserted), not_found=len(not_found))
    return inserted, not_found


async def ingest_catalog(
    tmdb: TMDBClient,
    engine: Engine,
    options: IngestOptions,
    *,
    today: date | None = None,
) -> IngestStats:
    """Run the full ingest. Returns summary stats."""
    started = time.monotonic()
    stats = IngestStats()
    today = today or utcnow().date()
    log.info("ingest_start", **{k: v for k, v in asdict(options).items()})

    keys = await discover_titles(tmdb, options, stats, today=today)
    stats.selected = len(keys)

    with Session(engine, expire_on_commit=False) as session:
        if options.resume and keys:
            fresh = fresh_keys(session, keys, utcnow() - timedelta(hours=options.fresh_hours))
            stats.skipped_fresh = len(fresh)
            keys = [k for k in keys if k not in fresh]

        done = 0
        for batch_keys in _chunks(keys, options.batch_size):
            results = await asyncio.gather(*(_fetch_details(tmdb, k) for k in batch_keys))
            batch: list[Details] = []
            for key, result in zip(batch_keys, results, strict=True):
                if isinstance(result, TMDBNotFoundError):
                    stats.not_found += 1
                elif isinstance(result, BaseException):
                    stats.failed += 1
                    log.warning(
                        "ingest_details_failed",
                        tmdb_id=key[0],
                        type=key[1].value,
                        error=type(result).__name__,
                        detail=str(result),
                    )
                else:
                    batch.append(result)
            if batch:
                upsert_batch(session, batch, options.countries, stats)
            done += len(batch_keys)
            log.info(
                "ingest_progress",
                done=done,
                total=len(keys),
                inserted=stats.inserted,
                updated=stats.updated,
                not_found=stats.not_found,
                failed=stats.failed,
            )

        await refresh_providers(tmdb, session, options.countries, stats)

    stats.duration_s = round(time.monotonic() - started, 2)
    log.info("ingest_done", **asdict(stats))
    return stats


# --- CLI -----------------------------------------------------------------------


def _csv_args(values: Sequence[str] | None) -> list[str]:
    """Accept both `--x a b` and `--x a,b`."""
    return [v.strip() for raw in values or [] for v in raw.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest.ingest_catalog",
        description="Ingest the TMDB catalog into the local database.",
    )
    parser.add_argument(
        "--languages", nargs="+", help="original languages (default: SEED_LANGUAGES)"
    )
    parser.add_argument(
        "--types", nargs="+", choices=[t.value for t in TitleType], help="default: movie tv"
    )
    parser.add_argument(
        "--pages",
        type=int,
        help="discover pages per language x type x strategy (default: INGEST_PAGES_PER_LANGUAGE)",
    )
    parser.add_argument("--countries", nargs="+", help="availability countries (SEED_COUNTRIES)")
    parser.add_argument("--limit", type=int, help="ingest at most N titles (smoke runs)")
    parser.add_argument(
        "--resume", action="store_true", help="skip titles updated within --fresh-hours"
    )
    parser.add_argument("--fresh-hours", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, default=100, help="titles per DB commit")
    return parser


def options_from_args(args: argparse.Namespace, settings: Settings) -> IngestOptions:
    for name in ("pages", "limit", "batch_size"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.fresh_hours < 0:
        raise SystemExit("--fresh-hours must be >= 0")
    languages = _csv_args(args.languages) or settings.seed_languages
    countries = [c.upper() for c in _csv_args(args.countries)] or settings.seed_countries
    for country in countries:
        if len(country) != 2 or not country.isalpha():
            raise SystemExit(f"Invalid country code: {country!r}")
    types = [TitleType(t) for t in _csv_args(args.types)] or list(TitleType)
    return IngestOptions(
        languages=tuple(dict.fromkeys(lang.lower() for lang in languages)),
        types=tuple(dict.fromkeys(types)),
        countries=tuple(dict.fromkeys(countries)),
        pages=args.pages or settings.ingest_pages_per_language,
        limit=args.limit,
        resume=args.resume,
        fresh_hours=args.fresh_hours,
        batch_size=args.batch_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    options = options_from_args(build_parser().parse_args(argv), settings)

    engine = create_db_engine(settings.database_url)
    try:
        if not inspect(engine).has_table(Title.__tablename__):
            print("Database schema missing: run `make migrate` first.", file=sys.stderr)
            return 2
        tmdb = TMDBClient.from_settings(settings)
    except TMDBConfigError as exc:
        print(f"TMDB is not configured: {exc}", file=sys.stderr)
        return 2

    async def run() -> IngestStats:
        async with tmdb:
            return await ingest_catalog(tmdb, engine, options)

    try:
        stats = asyncio.run(run())
    except TMDBAuthError:
        print("TMDB rejected the credentials (401). Check TMDB_READ_TOKEN / TMDB_API_KEY.")
        return 1
    finally:
        engine.dispose()

    print(
        f"Ingested {stats.inserted + stats.updated} titles "
        f"({stats.inserted} new, {stats.updated} updated, {stats.skipped_fresh} skipped as fresh, "
        f"{stats.not_found} not found, {stats.failed} failed); "
        f"{stats.availability_rows} availability rows, {stats.providers} providers "
        f"in {stats.duration_s}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
