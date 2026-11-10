"""`GET /titles/search` (P1-08, FR-12): search-as-you-type over TMDB movies + series.

TMDB matches the query against titles in the requested `language` (plus original
titles), so a Korean-language search for "squid" does not find Squid Game. When the
user's first preferred language is not English we therefore search in that language
**and** in English, and interleave the two result lists (deduplicated). A title shows
in the user's language when TMDB has it, else in English (TMDB itself would fall back to
the original title). Results are cached in-process per (query, language, include_adult).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from itertools import chain, zip_longest
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, DbSession, TMDBClientFactory
from app.core.cache import TTLCache
from app.core.config import Settings
from app.core.logging import get_logger
from app.db.models import Title, TitleType, UserPreferences
from app.db.models.user import DEFAULT_LANGUAGES
from app.ingest.providers import TMDBFactory
from app.ingest.tmdb_client import TMDBClient, TMDBError, poster_url
from app.ingest.tmdb_models import MediaListItem
from app.schemas.titles import SearchType, TitleSearchOut, TitleSearchResult

router = APIRouter(prefix="/titles", tags=["titles"])
log = get_logger(__name__)

ENGLISH = "en-US"
MIN_QUERY_LENGTH = 2
MAX_QUERY_LENGTH = 100
DEFAULT_LIMIT = 10
MAX_LIMIT = 20  # one TMDB page

# (normalized query, TMDB language, include_adult) -> movie/TV items of page 1, in TMDB order.
SearchKey = tuple[str, str, bool]
SearchCache = TTLCache[SearchKey, tuple[MediaListItem, ...]]

SearchQuery = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH
    ),
]


def new_search_cache(settings: Settings) -> SearchCache:
    return TTLCache(
        ttl_seconds=settings.title_search_cache_ttl_seconds,
        max_entries=settings.title_search_cache_max_entries,
    )


def get_search_cache(request: Request) -> SearchCache:
    """The app-wide cache created in `create_app` (one per app, so tests are isolated)."""
    cache: SearchCache = request.app.state.title_search_cache
    return cache


def normalize_query(q: str) -> str:
    """Cache key form: whitespace collapsed, case-folded (TMDB search is case-insensitive)."""
    return re.sub(r"\s+", " ", q).strip().casefold()


def search_languages(preferred: Sequence[str]) -> list[str]:
    """TMDB `language` values to search: the user's first language, then English."""
    first = (preferred or DEFAULT_LANGUAGES)[0]
    return [ENGLISH] if first == "en" else [first, ENGLISH]


def _localized(item: MediaListItem, language: str) -> bool:
    """Whether TMDB returned a title in `language` (it falls back to the original title).

    A title equal to the original only counts when the original is in that language.
    """
    base = language.split("-", 1)[0]
    return item.display_title != item.display_original_title or item.original_language == base


def merge_results(
    per_language: Sequence[tuple[str, Sequence[MediaListItem]]],
) -> list[MediaListItem]:
    """Interleave `(language, results)` lists (priority order), dropping repeats of a title.

    For a title found in several lists, the record from the highest-priority list that
    has a title in its language is used (so an untranslated title shows in English, not
    in its original script); if none has one, the highest-priority record is used.
    """
    records: dict[tuple[str | None, int], list[tuple[str, MediaListItem]]] = {}
    for language, items in per_language:
        for item in items:
            records.setdefault((item.media_type, item.id), []).append((language, item))
    merged: list[MediaListItem] = []
    seen: set[tuple[str | None, int]] = set()
    for item in chain.from_iterable(zip_longest(*(items for _, items in per_language))):
        if item is None:
            continue
        key = (item.media_type, item.id)
        if key not in seen:
            seen.add(key)
            found = records[key]
            merged.append(next((r for lang, r in found if _localized(r, lang)), found[0][1]))
    return merged


async def _fetch(tmdb: TMDBClient, cache: SearchCache, key: SearchKey) -> tuple[MediaListItem, ...]:
    query, language, include_adult = key
    page = await tmdb.search_multi(query, language, include_adult=include_adult)
    items = tuple(item for item in page.results if item.title_type is not None)  # no people
    cache.set(key, items)
    return items


async def search_titles(
    tmdb_factory: TMDBFactory,
    cache: SearchCache,
    query: str,
    languages: Sequence[str],
    *,
    include_adult: bool,
) -> list[MediaListItem]:
    """Merged movie/TV results for `query`; raises `TMDBError` only if every language fails.

    A TMDB client is opened only for languages that are not cached.
    """
    keys: list[SearchKey] = [(query, lang, include_adult) for lang in languages]
    found: dict[SearchKey, tuple[MediaListItem, ...]] = {}
    for key in keys:
        if (cached := cache.get(key)) is not None:
            found[key] = cached
    missing = [key for key in keys if key not in found]
    errors: list[TMDBError] = []
    if missing:
        try:
            async with tmdb_factory() as tmdb:
                outcomes = await asyncio.gather(
                    *(_fetch(tmdb, cache, key) for key in missing), return_exceptions=True
                )
        except TMDBError as exc:  # e.g. no credentials configured
            outcomes = [exc] * len(missing)
        for key, outcome in zip(missing, outcomes, strict=True):
            if isinstance(outcome, TMDBError):
                log.warning("title_search_failed", language=key[1], error=type(outcome).__name__)
                errors.append(outcome)
            elif isinstance(outcome, BaseException):
                raise outcome
            else:
                found[key] = outcome
    if not found:
        raise errors[0]
    return merge_results([(key[1], found[key]) for key in keys if key in found])


def _in_catalog(db: Session, items: Iterable[MediaListItem]) -> set[tuple[TitleType, int]]:
    ids = {item.id for item in items}
    if not ids:
        return set()
    rows = db.execute(select(Title.type, Title.tmdb_id).where(Title.tmdb_id.in_(ids)))
    return {(type_, tmdb_id) for type_, tmdb_id in rows}


@router.get("/search", response_model=TitleSearchOut)
async def search(
    user: CurrentUser,
    db: DbSession,
    tmdb_factory: TMDBClientFactory,
    cache: Annotated[SearchCache, Depends(get_search_cache)],
    q: Annotated[SearchQuery, Query(description="Search text, 2 to 100 characters.")],
    type_: Annotated[SearchType, Query(alias="type")] = "all",
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> TitleSearchOut:
    """Movies and series matching `q` (people are left out), in TMDB relevance order.

    Titles are localized to the user's first preferred language, with English as fallback.
    Adult titles are included only if the user allows them.
    """
    prefs = db.get(UserPreferences, user.id)
    languages = search_languages(prefs.languages if prefs else DEFAULT_LANGUAGES)
    include_adult = bool(prefs and prefs.allow_adult)
    query = normalize_query(q)
    try:
        items = await search_titles(
            tmdb_factory, cache, query, languages, include_adult=include_adult
        )
    except TMDBError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Title search is temporarily unavailable. Try again later.",
            headers={"Retry-After": "60"},
        ) from exc

    picked = [
        item
        for item in items
        if (type_ == "all" or item.media_type == type_) and (include_adult or not item.adult)
    ][:limit]
    catalog = _in_catalog(db, picked)
    results = []
    for item in picked:
        title_type = item.title_type
        assert title_type is not None  # noqa: S101 - people were filtered out
        results.append(
            TitleSearchResult(
                tmdb_id=item.id,
                type=title_type,
                title=item.display_title or item.display_original_title,
                original_title=item.display_original_title,
                year=item.year,
                poster_url=poster_url(item.poster_path),
                in_catalog=(title_type, item.id) in catalog,
            )
        )
    log.info("title_search", results=len(results), languages=languages, query_len=len(query))
    return TitleSearchOut(query=query, type=type_, languages=languages, results=results)
