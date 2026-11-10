"""Template reasons (§8.8, FR-26 Phase 1; LLM reasons arrive in P2).

- With a profile: the (up to) 2 positively rated titles (latest rating ≥ 4) most similar
  to the pick → "Because you liked {X} and {Y}" / "Because you liked {X}".
- Cold start, or a pick / liked titles without vectors: "Popular in {language} on
  {platform}" ("Popular in {language}" if no platform name is known).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.ingest.build_embeddings import Vectors

REASON_SOURCE = "template"

# English display names for common ISO 639-1 codes; unknown codes are shown upper-cased.
LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "bn": "Bengali",
    "cn": "Cantonese",  # TMDB uses "cn" for Cantonese
    "da": "Danish",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code.lower(), code.upper())


def because_you_liked(names: Sequence[str]) -> str:
    if len(names) >= 2:
        return f"Because you liked {names[0]} and {names[1]}"
    return f"Because you liked {names[0]}"


def popular_in(language: str, platform: str | None) -> str:
    base = f"Popular in {language_name(language)}"
    return f"{base} on {platform}" if platform else base


def most_similar_liked(
    vector: Vectors | None, liked_vectors: Vectors, liked_names: Sequence[str], k: int = 2
) -> list[str]:
    """Names of the `k` liked titles closest to `vector` (ties → earlier in `liked_names`)."""
    if vector is None or not liked_names or not np.any(vector):
        return []
    sims = liked_vectors @ vector
    order = np.lexsort((np.arange(len(liked_names)), -sims))[:k]
    return [liked_names[int(i)] for i in order]


def template_reason(
    *,
    vector: Vectors | None,
    liked_vectors: Vectors,
    liked_names: Sequence[str],
    language: str,
    platform: str | None,
    cold_start: bool,
) -> str:
    if not cold_start:
        names = most_similar_liked(vector, liked_vectors, liked_names)
        if names:
            return because_you_liked(names)
    return popular_in(language, platform)
