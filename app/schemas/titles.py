"""Title search models (P1-08, FR-12)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.db.models.enums import TitleType

SearchType = Literal["movie", "tv", "all"]


class TitleSearchResult(BaseModel):
    tmdb_id: int
    type: TitleType
    title: str  # localized (user's first language, else English, else TMDB's fallback)
    original_title: str
    year: int | None
    poster_url: str | None
    in_catalog: bool


class TitleSearchOut(BaseModel):
    query: str
    type: SearchType
    # TMDB `language` values searched, in priority order (e.g. `["ko", "en-US"]`).
    languages: list[str]
    results: list[TitleSearchResult]
