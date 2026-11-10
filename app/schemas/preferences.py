"""Preferences and provider-list models (P1-07, FR-10, FR-11)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
)

from app.db.models.enums import Era, OnboardingStep, TitleType
from app.db.models.user import DEFAULT_LANGUAGES
from app.ingest.genres import CANONICAL_GENRES

MAX_LANGUAGES = 20
MAX_PLATFORMS = 50
MAX_RUNTIME_MINUTES = 1000


def _normalize(value: object) -> object:
    return value.strip().lower() if isinstance(value, str) else value


# ISO 639-1 style codes as used by TMDB `original_language` (any TMDB language is allowed).
LanguageCode = Annotated[str, BeforeValidator(_normalize), StringConstraints(pattern=r"^[a-z]{2}$")]


def _dedupe[T](values: list[T]) -> list[T]:
    """Drop repeats, keeping first-seen order (the first language is the preferred one)."""
    return list(dict.fromkeys(values))


def _or_default_languages(values: list[str]) -> list[str]:
    return values or list(DEFAULT_LANGUAGES)


def _genre_slug(value: str) -> str:
    if value not in CANONICAL_GENRES:
        raise ValueError("must be a canonical genre slug")
    return value


GenreSlug = Annotated[
    str, BeforeValidator(_normalize), StringConstraints(max_length=50), AfterValidator(_genre_slug)
]
ProviderId = Annotated[int, Field(gt=0)]


class PreferencesIn(BaseModel):
    """`PUT /me/preferences`: replaces all preferences. Omitted fields take their defaults."""

    model_config = ConfigDict(extra="forbid")

    languages: Annotated[
        list[LanguageCode],
        Field(
            max_length=MAX_LANGUAGES,
            description="Preferred first. Omitted or `[]` means the default (English).",
        ),
        AfterValidator(_dedupe),
        AfterValidator(_or_default_languages),
    ] = Field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    platforms: Annotated[
        list[ProviderId],
        Field(max_length=MAX_PLATFORMS, description="TMDB provider IDs in the user's country."),
        AfterValidator(_dedupe),
    ] = Field(default_factory=list)
    content_types: Annotated[
        list[TitleType], Field(min_length=1, max_length=2), AfterValidator(_dedupe)
    ] = Field(default_factory=lambda: [TitleType.MOVIE, TitleType.TV])
    era: Era = Era.ANY
    avoid_genres: Annotated[
        list[GenreSlug], Field(max_length=len(CANONICAL_GENRES)), AfterValidator(_dedupe)
    ] = Field(default_factory=list)
    max_runtime: Annotated[int, Field(gt=0, le=MAX_RUNTIME_MINUTES)] | None = None
    include_rent_buy: bool = False
    allow_adult: bool = False


class PreferencesOut(BaseModel):
    languages: list[str]
    platforms: list[int]
    content_types: list[TitleType]
    era: Era
    avoid_genres: list[str]
    max_runtime: int | None
    include_rent_buy: bool
    allow_adult: bool
    onboarding_step: OnboardingStep
    updated_at: datetime | None = Field(description="`null` until preferences are first saved.")


class ProviderOut(BaseModel):
    provider_id: int
    name: str
    logo_url: str | None
    display_priority: int | None


class ProviderListOut(BaseModel):
    country: str
    providers: list[ProviderOut]
    fetched_at: datetime | None = Field(description="Oldest cache timestamp; `null` if empty.")
