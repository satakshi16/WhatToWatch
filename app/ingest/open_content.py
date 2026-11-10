"""Open content ingest CLI: Wikidata + Wikipedia → `title_open_content` (P1-04b, D-011).

Usage:
    python -m app.ingest.open_content [--types movie tv] [--limit N]
        [--resume [--fresh-hours H]] [--batch-size N]

Flow:
1. Select catalog titles (most popular first, interleaved across language x type so a
   small `--limit` covers every language). `--resume` skips titles whose open content
   was fetched within `--fresh-hours`.
2. Wikidata SPARQL, ≤ 200 TMDB IDs per query (`VALUES`), joined by P4947 (TMDB movie
   ID) or P4983 (TMDB TV series ID): label, original language (P364 → P218), genres
   (P136), directors (P57), cast (P161 + P1545 ordinal), publication date (P577) and
   Wikipedia sitelinks.
3. Wikipedia: the enwiki article, else the article in the title's (Wikidata) original
   language; the lead summary comes from the REST summary endpoint.
4. Upsert one row per checked title (also for "no Wikidata match", so `--resume` and
   coverage stats see it), one commit per batch.

Provenance (D-011): TMDB data is used only to pick and order which titles to look up
(IDs as join keys, popularity/language for ordering and coverage reporting). Nothing
from TMDB is written to `title_open_content`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from itertools import zip_longest

from sqlalchemy import Engine, inspect, select, tuple_
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.base import utcnow
from app.db.models import WIKIPEDIA_LICENSE, Title, TitleOpenContent, TitleType
from app.db.session import create_db_engine
from app.ingest.wikimedia_client import (
    WikimediaClient,
    WikimediaConfigError,
    WikimediaError,
    WikiSummary,
    wiki_site,
)

log = get_logger(__name__)

TitleKey = tuple[int, TitleType]

# Wikidata external-ID properties holding TMDB IDs.
TMDB_ID_PROPERTY: dict[TitleType, str] = {TitleType.MOVIE: "P4947", TitleType.TV: "P4983"}
MAX_SPARQL_IDS = 200
TOP_CAST = 3
ENTITY_PREFIX = "http://www.wikidata.org/entity/"

STATUS_SUMMARY = "summary"
STATUS_NO_WIKIPEDIA = "no_wikipedia"
STATUS_NO_WIKIDATA = "no_wikidata"
STATUS_UNCHECKED = "unchecked"


# --- SPARQL --------------------------------------------------------------------


def wiki_languages_for(seed_languages: Sequence[str]) -> tuple[str, ...]:
    """Wikipedias we read summaries from: English first, then the seed languages."""
    return tuple(dict.fromkeys(["en", *(lang.lower() for lang in seed_languages)]))


def label_languages_for(seed_languages: Sequence[str]) -> tuple[str, ...]:
    """Label fallback order for the Wikidata label service."""
    return tuple(dict.fromkeys(["en", "mul", *(lang.lower() for lang in seed_languages)]))


def sparql_query(
    type_: TitleType,
    tmdb_ids: Sequence[int],
    *,
    wiki_languages: Sequence[str],
    label_languages: Sequence[str],
) -> str:
    """One query per batch; each UNION branch tags its rows with `?kind`."""
    if not 0 < len(tmdb_ids) <= MAX_SPARQL_IDS:
        raise ValueError(f"1..{MAX_SPARQL_IDS} TMDB IDs per query, got {len(tmdb_ids)}")
    values = " ".join(f'"{int(i)}"' for i in tmdb_ids)
    sites = " ".join(f"<https://{lang}.wikipedia.org/>" for lang in wiki_languages)
    labels = ",".join(label_languages)
    prop = TMDB_ID_PROPERTY[type_]
    return f"""SELECT ?tmdb ?item ?itemLabel ?kind ?value ?valueLabel ?text ?ord WHERE {{
  VALUES ?tmdb {{ {values} }}
  ?item wdt:{prop} ?tmdb .
  {{ BIND("item" AS ?kind) }}
  UNION {{ ?item wdt:P136 ?value . BIND("genre" AS ?kind) }}
  UNION {{ ?item wdt:P57 ?value . BIND("director" AS ?kind) }}
  UNION {{
    ?item p:P161 ?castStatement . ?castStatement ps:P161 ?value .
    OPTIONAL {{ ?castStatement pq:P1545 ?ord }}
    BIND("cast" AS ?kind)
  }}
  UNION {{ ?item wdt:P364 ?value . ?value wdt:P218 ?text . BIND("language" AS ?kind) }}
  UNION {{ ?item wdt:P577 ?text . BIND("date" AS ?kind) }}
  UNION {{
    VALUES ?value {{ {sites} }}
    ?article schema:about ?item ; schema:isPartOf ?value ; schema:name ?text .
    BIND("sitelink" AS ?kind)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{labels}" . }}
}}"""


# --- Parsing -------------------------------------------------------------------


def qid_of(uri: str) -> str | None:
    if uri.startswith(ENTITY_PREFIX):
        qid = uri.removeprefix(ENTITY_PREFIX)
        return qid if qid[:1] == "Q" and qid[1:].isdigit() else None
    return None


def qid_number(qid: str) -> int:
    return int(qid[1:])


def _real_label(label: str | None, qid: str | None) -> str | None:
    """The label service falls back to the QID when no label exists."""
    label = (label or "").strip()
    return label if label and label != qid else None


def _ordinal(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value.isdigit() else None


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat((value or "")[:10])
    except ValueError:
        return None


def _site_of(url: str) -> str | None:
    """`https://en.wikipedia.org/` → `enwiki`."""
    prefix, suffix = "https://", ".wikipedia.org/"
    if url.startswith(prefix) and url.endswith(suffix):
        lang = url[len(prefix) : -len(suffix)]
        return wiki_site(lang) if lang else None
    return None


@dataclass
class WikidataRecord:
    """Structured fields for one Wikidata item matched to a TMDB ID."""

    tmdb_id: int
    qid: str
    label: str | None = None
    # (QID of the entity, label) pairs, ordered when finalised.
    genres: dict[str, str] = field(default_factory=dict)
    directors: dict[str, str] = field(default_factory=dict)
    # actor QID → (ordinal or None, label)
    cast: dict[str, tuple[int | None, str]] = field(default_factory=dict)
    languages: dict[str, str] = field(default_factory=dict)  # language QID → ISO 639-1
    dates: set[date] = field(default_factory=set)
    sitelinks: dict[str, str] = field(default_factory=dict)  # site → article title

    def ordered_labels(self, entries: dict[str, str]) -> list[str]:
        return list(dict.fromkeys(entries[q] for q in sorted(entries, key=qid_number)))

    @property
    def genre_labels(self) -> list[str]:
        return self.ordered_labels(self.genres)

    @property
    def director_labels(self) -> list[str]:
        return self.ordered_labels(self.directors)

    @property
    def top_cast(self) -> list[str]:
        """≤ 3 actors: P1545 ordinal first (ascending), then by QID; deterministic."""

        def key(item: tuple[str, tuple[int | None, str]]) -> tuple[int, int, int]:
            qid, (ordinal, _) = item
            return (0, ordinal, qid_number(qid)) if ordinal is not None else (1, 0, qid_number(qid))

        names = [label for _, (_, label) in sorted(self.cast.items(), key=key)]
        return list(dict.fromkeys(names))[:TOP_CAST]

    @property
    def language_codes(self) -> list[str]:
        return list(
            dict.fromkeys(self.languages[q].lower() for q in sorted(self.languages, key=qid_number))
        )

    @property
    def original_language(self) -> str | None:
        codes = self.language_codes
        return codes[0] if codes else None

    @property
    def publication_date(self) -> date | None:
        return min(self.dates) if self.dates else None


def parse_bindings(rows: Iterable[dict[str, str]]) -> dict[int, WikidataRecord]:
    """SPARQL rows → one record per TMDB ID (lowest QID wins if several items match)."""
    by_item: dict[tuple[int, str], WikidataRecord] = {}
    for row in rows:
        qid = qid_of(row.get("item", ""))
        tmdb = row.get("tmdb", "").strip()
        if qid is None or not tmdb.isdigit():
            continue
        record = by_item.setdefault((int(tmdb), qid), WikidataRecord(int(tmdb), qid))
        record.label = record.label or _real_label(row.get("itemLabel"), qid)
        kind = row.get("kind")
        value = row.get("value", "")
        value_qid = qid_of(value)
        value_label = _real_label(row.get("valueLabel"), value_qid)
        text = (row.get("text") or "").strip()
        if kind in ("genre", "director") and value_qid and value_label:
            target = record.genres if kind == "genre" else record.directors
            target[value_qid] = value_label
        elif kind == "cast" and value_qid and value_label:
            ordinal = _ordinal(row.get("ord"))
            previous = record.cast.get(value_qid)
            if previous is None or (ordinal is not None and (previous[0] or 10**9) > ordinal):
                record.cast[value_qid] = (ordinal, value_label)
        elif kind == "language" and value_qid and text:
            record.languages[value_qid] = text
        elif kind == "date" and (parsed := _parse_date(text)):
            record.dates.add(parsed)
        elif kind == "sitelink" and (site := _site_of(value)) and text:
            record.sitelinks[site] = text

    chosen: dict[int, WikidataRecord] = {}
    for (tmdb_id, _qid), record in sorted(by_item.items(), key=lambda kv: qid_number(kv[0][1])):
        chosen.setdefault(tmdb_id, record)
    return chosen


def choose_article(record: WikidataRecord, wiki_languages: Sequence[str]) -> tuple[str, str] | None:
    """enwiki if present, else the Wikipedia in the title's (Wikidata) original language."""
    if "enwiki" in record.sitelinks:
        return "enwiki", record.sitelinks["enwiki"]
    allowed = set(wiki_languages)
    for code in record.language_codes:
        site = wiki_site(code)
        if code in allowed and site in record.sitelinks:
            return site, record.sitelinks[site]
    return None


def open_content_values(
    key: TitleKey,
    record: WikidataRecord | None,
    summary: WikiSummary | None,
    fetched_at: datetime,
) -> dict[str, object]:
    """Column values for `title_open_content`. Built only from Wikimedia data."""
    tmdb_id, type_ = key
    values: dict[str, object] = {
        "tmdb_id": tmdb_id,
        "type": type_,
        "wikidata_qid": None,
        "label": None,
        "original_language": None,
        "genres": [],
        "directors": [],
        "top_cast": [],
        "publication_date": None,
        "wiki_site": None,
        "wiki_title": None,
        "wiki_revision_id": None,
        "summary": None,
        "summary_license": None,
        "fetched_at": fetched_at,
    }
    if record is None:
        return values
    values.update(
        wikidata_qid=record.qid,
        label=record.label,
        original_language=record.original_language,
        genres=record.genre_labels,
        directors=record.director_labels,
        top_cast=record.top_cast,
        publication_date=record.publication_date,
    )
    if summary is not None:
        values.update(wiki_site=summary.site, wiki_title=summary.title)
        if summary.usable:
            values.update(
                wiki_revision_id=summary.revision_id,
                summary=summary.extract,
                summary_license=WIKIPEDIA_LICENSE,
            )
    return values


# --- Fetching ------------------------------------------------------------------


@dataclass
class FetchResult:
    rows: list[dict[str, object]] = field(default_factory=list)
    failed: list[TitleKey] = field(default_factory=list)


async def _summary_or_error(
    wm: WikimediaClient, site: str, title: str
) -> WikiSummary | WikimediaError | None:
    try:
        return await wm.page_summary(site, title)
    except WikimediaConfigError:
        raise
    except WikimediaError as exc:
        return exc


async def fetch_open_content(
    wm: WikimediaClient,
    keys: Sequence[TitleKey],
    *,
    wiki_languages: Sequence[str],
    label_languages: Sequence[str],
    fetched_at: datetime | None = None,
) -> FetchResult:
    """Look up `keys` on Wikidata + Wikipedia. Keys that failed are not given a row."""
    fetched_at = fetched_at or utcnow()
    result = FetchResult()
    by_type: dict[TitleType, list[int]] = defaultdict(list)
    for tmdb_id, type_ in keys:
        by_type[type_].append(tmdb_id)

    for type_, ids in by_type.items():
        for chunk in _chunks(ids, MAX_SPARQL_IDS):
            query = sparql_query(
                type_, chunk, wiki_languages=wiki_languages, label_languages=label_languages
            )
            try:
                records = parse_bindings(await wm.sparql(query))
            except WikimediaConfigError:
                raise
            except WikimediaError as exc:
                log.warning(
                    "open_content_sparql_failed", type=type_.value, ids=len(chunk), error=str(exc)
                )
                result.failed.extend((i, type_) for i in chunk)
                continue

            articles: dict[int, tuple[str, str]] = {}
            for i in chunk:
                record = records.get(i)
                article = choose_article(record, wiki_languages) if record else None
                if article:
                    articles[i] = article
            summaries = await asyncio.gather(
                *(_summary_or_error(wm, site, title) for site, title in articles.values())
            )
            fetched = dict(zip(articles, summaries, strict=True))
            for tmdb_id in chunk:
                summary = fetched.get(tmdb_id)
                if isinstance(summary, WikimediaError):
                    log.warning(
                        "open_content_summary_failed",
                        tmdb_id=tmdb_id,
                        type=type_.value,
                        error=str(summary),
                    )
                    result.failed.append((tmdb_id, type_))
                    continue
                result.rows.append(
                    open_content_values((tmdb_id, type_), records.get(tmdb_id), summary, fetched_at)
                )
    return result


# --- Persistence + reporting ---------------------------------------------------


def row_status(row: dict[str, object] | TitleOpenContent | None) -> str:
    if row is None:
        return STATUS_UNCHECKED
    get = row.get if isinstance(row, dict) else lambda k: getattr(row, k)
    if not get("wikidata_qid"):
        return STATUS_NO_WIKIDATA
    summary = get("summary")
    return STATUS_SUMMARY if isinstance(summary, str) and summary.strip() else STATUS_NO_WIKIPEDIA


def upsert_open_content(session: Session, rows: Sequence[dict[str, object]]) -> None:
    for row in rows:
        session.merge(TitleOpenContent(**row))
    session.commit()


def select_keys(session: Session, types: Sequence[TitleType], limit: int | None) -> list[TitleKey]:
    """Catalog titles, most popular first, interleaved across language x type buckets."""
    stmt = (
        select(Title.tmdb_id, Title.type, Title.original_language)
        .where(Title.type.in_(types))
        .order_by(Title.popularity.desc(), Title.tmdb_id, Title.type)
    )
    buckets: dict[tuple[str, TitleType], list[TitleKey]] = {}
    for tmdb_id, type_, language in session.execute(stmt):
        buckets.setdefault((language, type_), []).append((tmdb_id, type_))
    keys = [k for group in zip_longest(*buckets.values()) for k in group if k is not None]
    return keys[:limit] if limit is not None else keys


def fresh_keys(session: Session, keys: Sequence[TitleKey], since: datetime) -> set[TitleKey]:
    found: set[TitleKey] = set()
    for chunk in _chunks(keys, 500):
        stmt = select(TitleOpenContent.tmdb_id, TitleOpenContent.type).where(
            tuple_(TitleOpenContent.tmdb_id, TitleOpenContent.type).in_(chunk),
            TitleOpenContent.fetched_at >= since,
        )
        found.update((tmdb_id, type_) for tmdb_id, type_ in session.execute(stmt))
    return found


@dataclass(frozen=True)
class CoverageRow:
    language: str
    type: str
    total: int
    wikidata: int
    summary: int
    unchecked: int

    @property
    def coverage(self) -> float:
        return self.summary / self.total if self.total else 0.0


def coverage_report(session: Session) -> list[CoverageRow]:
    """Open-content coverage of the whole catalog by (TMDB) language x type.

    TMDB's language is used for grouping only; this report is printed, never written
    into the embedding files.
    """
    stmt = select(
        Title.original_language,
        Title.type,
        TitleOpenContent.wikidata_qid,
        TitleOpenContent.summary,
        TitleOpenContent.tmdb_id,
    ).outerjoin(
        TitleOpenContent,
        (TitleOpenContent.tmdb_id == Title.tmdb_id) & (TitleOpenContent.type == Title.type),
    )
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for language, type_, qid, summary, checked in session.execute(stmt):
        bucket = counts[(language, type_.value)]
        bucket["total"] += 1
        if checked is None:
            bucket["unchecked"] += 1
        if qid:
            bucket["wikidata"] += 1
        if summary and summary.strip():
            bucket["summary"] += 1
    return [
        CoverageRow(lang, type_, c["total"], c["wikidata"], c["summary"], c["unchecked"])
        for (lang, type_), c in sorted(counts.items())
    ]


def format_coverage(rows: Sequence[CoverageRow]) -> str:
    lines = [
        f"{'lang':<6}{'type':<7}{'titles':>7}{'wikidata':>10}{'summary':>9}"
        f"{'unchecked':>11}{'coverage':>10}"
    ]
    for r in rows:
        lines.append(
            f"{r.language:<6}{r.type:<7}{r.total:>7}{r.wikidata:>10}{r.summary:>9}"
            f"{r.unchecked:>11}{r.coverage:>9.0%} "
        )
    total = sum(r.total for r in rows)
    summary = sum(r.summary for r in rows)
    lines.append(
        f"{'all':<13}{total:>7}{sum(r.wikidata for r in rows):>10}{summary:>9}"
        f"{sum(r.unchecked for r in rows):>11}{(summary / total if total else 0):>9.0%} "
    )
    return "\n".join(line.rstrip() for line in lines)


# --- Orchestration -------------------------------------------------------------


@dataclass(frozen=True)
class OpenContentOptions:
    types: tuple[TitleType, ...] = tuple(TitleType)
    limit: int | None = None
    resume: bool = False
    fresh_hours: float = 24.0
    batch_size: int = MAX_SPARQL_IDS


@dataclass
class OpenContentStats:
    selected: int = 0
    skipped_fresh: int = 0
    checked: int = 0
    matched: int = 0
    no_wikidata: int = 0
    no_wikipedia: int = 0
    with_summary: int = 0
    failed: int = 0
    duration_s: float = 0.0
    # "<lang>/<type>" → status → count (TMDB language, for reporting only)
    by_language_type: dict[str, dict[str, int]] = field(default_factory=dict)


async def ingest_open_content(
    wm: WikimediaClient,
    engine: Engine,
    options: OpenContentOptions,
    *,
    wiki_languages: Sequence[str],
    label_languages: Sequence[str],
) -> OpenContentStats:
    started = time.monotonic()
    stats = OpenContentStats()
    log.info("open_content_start", **asdict(options))
    with Session(engine, expire_on_commit=False) as session:
        keys = select_keys(session, options.types, options.limit)
        stats.selected = len(keys)
        if options.resume and keys:
            fresh = fresh_keys(session, keys, utcnow() - timedelta(hours=options.fresh_hours))
            stats.skipped_fresh = len(fresh)
            keys = [k for k in keys if k not in fresh]
        languages: dict[TitleKey, str] = {
            (tmdb_id, type_): language
            for tmdb_id, type_, language in session.execute(
                select(Title.tmdb_id, Title.type, Title.original_language)
            )
        }

        done = 0
        for batch in _chunks(keys, options.batch_size):
            result = await fetch_open_content(
                wm, batch, wiki_languages=wiki_languages, label_languages=label_languages
            )
            if result.rows:
                upsert_open_content(session, result.rows)
            for row in result.rows:
                status = row_status(row)
                stats.checked += 1
                stats.matched += status != STATUS_NO_WIKIDATA
                stats.no_wikidata += status == STATUS_NO_WIKIDATA
                stats.no_wikipedia += status == STATUS_NO_WIKIPEDIA
                stats.with_summary += status == STATUS_SUMMARY
                _count(stats, languages, row, status)
            stats.failed += len(result.failed)
            done += len(batch)
            log.info(
                "open_content_progress",
                done=done,
                total=len(keys),
                matched=stats.matched,
                with_summary=stats.with_summary,
                failed=stats.failed,
            )
    stats.duration_s = round(time.monotonic() - started, 2)
    log.info("open_content_done", **asdict(stats))
    return stats


def _count(
    stats: OpenContentStats,
    languages: dict[TitleKey, str],
    row: dict[str, object],
    status: str,
) -> None:
    type_, tmdb_id = row["type"], row["tmdb_id"]
    assert isinstance(type_, TitleType)  # noqa: S101 - type narrowing
    assert isinstance(tmdb_id, int)  # noqa: S101 - type narrowing
    label = f"{languages.get((tmdb_id, type_), '?')}/{type_.value}"
    bucket = stats.by_language_type.setdefault(label, {})
    bucket[status] = bucket.get(status, 0) + 1


def _chunks[T](items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --- CLI -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest.open_content",
        description="Fetch open-licensed item content (Wikidata + Wikipedia) for catalog titles.",
    )
    parser.add_argument(
        "--types", nargs="+", choices=[t.value for t in TitleType], help="default: movie tv"
    )
    parser.add_argument("--limit", type=int, help="look up at most N titles (smoke runs)")
    parser.add_argument(
        "--resume", action="store_true", help="skip titles fetched within --fresh-hours"
    )
    parser.add_argument("--fresh-hours", type=float, default=24.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=MAX_SPARQL_IDS,
        help=f"titles per SPARQL query and DB commit (max {MAX_SPARQL_IDS})",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> OpenContentOptions:
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if not 1 <= args.batch_size <= MAX_SPARQL_IDS:
        raise SystemExit(f"--batch-size must be between 1 and {MAX_SPARQL_IDS}")
    if args.fresh_hours < 0:
        raise SystemExit("--fresh-hours must be >= 0")
    types = [TitleType(t) for raw in args.types or [] for t in raw.split(",") if t]
    return OpenContentOptions(
        types=tuple(dict.fromkeys(types)) or tuple(TitleType),
        limit=args.limit,
        resume=args.resume,
        fresh_hours=args.fresh_hours,
        batch_size=args.batch_size,
    )


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    options = options_from_args(build_parser().parse_args(argv))
    try:
        wm = WikimediaClient.from_settings(settings)
    except WikimediaConfigError as exc:
        print(f"Wikimedia is not configured: {exc}", file=sys.stderr)
        return 2

    engine = create_db_engine(settings.database_url)
    try:
        if not inspect(engine).has_table(TitleOpenContent.__tablename__):
            print("Database schema missing: run `make migrate` first.", file=sys.stderr)
            asyncio.run(wm.aclose())
            return 2

        async def run() -> OpenContentStats:
            async with wm:
                return await ingest_open_content(
                    wm,
                    engine,
                    options,
                    wiki_languages=wiki_languages_for(settings.seed_languages),
                    label_languages=label_languages_for(settings.seed_languages),
                )

        stats = asyncio.run(run())
        with Session(engine) as session:
            report = coverage_report(session)
    finally:
        engine.dispose()

    print(
        f"Checked {stats.checked} titles ({stats.with_summary} with a Wikipedia summary, "
        f"{stats.no_wikipedia} on Wikidata without a usable article, {stats.no_wikidata} "
        f"not on Wikidata, {stats.skipped_fresh} skipped as fresh, {stats.failed} failed) "
        f"in {stats.duration_s}s.\n"
    )
    print("Open-content coverage of the catalog (language = TMDB original language):")
    print(format_coverage(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
