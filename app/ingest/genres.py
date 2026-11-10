"""Canonical genre slugs, merging TMDB's separate movie and TV genre lists (§13).

TMDB uses different genre IDs for movies and TV, and some TV genres are
compound ("Sci-Fi & Fantasy"). Everything we store (`titles.genres`) and
filter on (`user_preferences.avoid_genres`) uses the slugs defined here.
IDs shared by both lists (e.g. 16 Animation, 35 Comedy) mean the same thing,
so one merged map is enough.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

ANIME = "anime"

# TMDB genre ID -> canonical slugs, in priority order.
TMDB_GENRE_MAP: Mapping[int, tuple[str, ...]] = {
    # Movie genres
    28: ("action",),
    12: ("adventure",),
    16: ("animation",),
    35: ("comedy",),
    80: ("crime",),
    99: ("documentary",),
    18: ("drama",),
    10751: ("family",),
    14: ("fantasy",),
    36: ("history",),
    27: ("horror",),
    10402: ("music",),
    9648: ("mystery",),
    10749: ("romance",),
    878: ("science-fiction",),
    10770: ("tv-movie",),
    53: ("thriller",),
    10752: ("war",),
    37: ("western",),
    # TV-only genres
    10759: ("action", "adventure"),  # Action & Adventure
    10762: ("kids",),
    10763: ("news",),
    10764: ("reality",),
    10765: ("science-fiction", "fantasy"),  # Sci-Fi & Fantasy
    10766: ("soap",),
    10767: ("talk",),
    10768: ("war", "politics"),  # War & Politics
}

_ANIMATION_ID = 16
_ANIME_LANGUAGES = frozenset({"ja"})

CANONICAL_GENRES: frozenset[str] = frozenset(
    {slug for slugs in TMDB_GENRE_MAP.values() for slug in slugs} | {ANIME}
)


def canonical_genres(genre_ids: Iterable[int], original_language: str | None = None) -> list[str]:
    """Map TMDB genre IDs (movie or TV) to de-duplicated canonical slugs.

    Order follows TMDB's order, so the first slug is the primary genre.
    Japanese-language animation is also tagged `anime`, right after `animation`.
    Unknown IDs are skipped.
    """
    is_anime_language = (original_language or "").lower() in _ANIME_LANGUAGES
    result: list[str] = []
    for genre_id in genre_ids:
        slugs = TMDB_GENRE_MAP.get(genre_id, ())
        if genre_id == _ANIMATION_ID and is_anime_language:
            slugs = (*slugs, ANIME)
        for slug in slugs:
            if slug not in result:
                result.append(slug)
    return result
