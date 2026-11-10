"""Static choices and display helpers for the Streamlit UI (P1-13).

Kept free of `app.*` imports; tests check these lists stay in step with the backend
(canonical genre slugs, eras, ISO country codes).
"""

from __future__ import annotations

import importlib.resources
from functools import lru_cache

# Canonical genre slugs (`app/ingest/genres.py`) → labels.
GENRES: dict[str, str] = {
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "anime": "Anime",
    "comedy": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "family": "Family",
    "fantasy": "Fantasy",
    "history": "History",
    "horror": "Horror",
    "kids": "Kids",
    "music": "Music",
    "mystery": "Mystery",
    "news": "News",
    "politics": "Politics",
    "reality": "Reality",
    "romance": "Romance",
    "science-fiction": "Science fiction",
    "soap": "Soap",
    "talk": "Talk",
    "thriller": "Thriller",
    "tv-movie": "TV movie",
    "war": "War",
    "western": "Western",
}

# TMDB `original_language` codes (ISO 639-1; `cn` is TMDB's code for Cantonese).
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ko": "Korean",
    "ja": "Japanese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese (Mandarin)",
    "cn": "Cantonese",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "bn": "Bengali",
    "mr": "Marathi",
    "pa": "Punjabi",
    "tr": "Turkish",
    "th": "Thai",
    "id": "Indonesian",
    "tl": "Tagalog",
    "ru": "Russian",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
}

ERAS: dict[str, str] = {
    "any": "Any year",
    "last_2y": "Last 2 years",
    "last_10y": "Last 10 years",
    "classics_pre_2000": "Classics (before 2000)",
}

CONTENT_TYPES: dict[str, str] = {"movie": "Movies", "tv": "Series"}
TYPE_LABELS: dict[str, str] = {"movie": "Movie", "tv": "Series"}
SEARCH_TYPES: dict[str, str] = {"all": "All", "movie": "Movies", "tv": "Series"}


@lru_cache
def countries() -> dict[str, str]:
    """ISO-3166-1 alpha-2 code → name, sorted by name (from the `tzdata` package)."""
    text = (importlib.resources.files("tzdata") / "zoneinfo" / "iso3166.tab").read_text("utf-8")
    rows = [line.split("\t") for line in text.splitlines() if line and not line.startswith("#")]
    return dict(sorted(((r[0].upper(), r[1].strip()) for r in rows), key=lambda kv: kv[1]))


def language_label(code: str | None) -> str:
    if not code:
        return "Unknown language"
    return LANGUAGES.get(code, code.upper())


def genre_label(slug: str) -> str:
    return GENRES.get(slug, slug.replace("-", " ").capitalize())


def runtime_label(minutes: int | None) -> str | None:
    if not minutes:
        return None
    hours, mins = divmod(minutes, 60)
    if not hours:
        return f"{mins}m"
    return f"{hours}h {mins:02d}m" if mins else f"{hours}h"


def seasons_label(seasons: int | None) -> str | None:
    if not seasons:
        return None
    return f"{seasons} season{'s' if seasons != 1 else ''}"


def meta_line(item: dict[str, object]) -> str:
    """`2010 · Movie · English · 2h 28m · Science fiction, Action` (unknown parts skipped)."""
    type_ = str(item.get("type", ""))
    runtime = item.get("runtime_min")
    seasons = item.get("seasons")
    language = item.get("language")
    genres = item.get("genres") or []
    parts = [
        str(item["year"]) if item.get("year") else None,
        TYPE_LABELS.get(type_, type_),
        language_label(language if isinstance(language, str) else None),
        seasons_label(seasons if isinstance(seasons, int) else None)
        if type_ == "tv"
        else runtime_label(runtime if isinstance(runtime, int) else None),
        ", ".join(genre_label(str(g)) for g in genres) if isinstance(genres, list) else None,
    ]
    return " · ".join(p for p in parts if p)
