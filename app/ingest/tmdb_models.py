"""Pydantic models for the parts of TMDB v3 responses we use.

Unknown fields are ignored, so TMDB adding fields never breaks parsing.
TMDB sends `""` for missing dates; those become `None`.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from app.db.models.enums import OfferType, TitleType


def _empty_to_none(value: Any) -> Any:
    return None if value == "" else value


def _rename_key(data: Any, src: str, dst: str) -> Any:
    """Copy `data[src]` to `data[dst]` when only `src` is present (movie/TV naming)."""
    if isinstance(data, dict) and src in data and dst not in data:
        return {**data, dst: data[src]}
    return data


OptionalDate = Annotated[date | None, BeforeValidator(_empty_to_none)]

MediaType = Literal["movie", "tv", "person"]


class TMDBModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


# --- Shared pieces -------------------------------------------------------------


class Genre(TMDBModel):
    id: int
    name: str


class Keyword(TMDBModel):
    id: int
    name: str


class Keywords(TMDBModel):
    # Movies return `keywords`, TV returns `results`.
    keywords: list[Keyword] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _tv_results(cls, data: Any) -> Any:
        return _rename_key(data, "results", "keywords")


class CastMember(TMDBModel):
    id: int
    name: str
    character: str | None = None
    order: int = 0


class CrewMember(TMDBModel):
    id: int
    name: str
    job: str | None = None
    department: str | None = None


class Credits(TMDBModel):
    cast: list[CastMember] = Field(default_factory=list)
    crew: list[CrewMember] = Field(default_factory=list)


class Creator(TMDBModel):
    id: int
    name: str


# --- Watch providers (JustWatch data) -----------------------------------------


class ProviderOffer(TMDBModel):
    provider_id: int
    provider_name: str
    logo_path: str | None = None
    display_priority: int = 0


class CountryProviders(TMDBModel):
    link: str | None = None
    flatrate: list[ProviderOffer] = Field(default_factory=list)
    free: list[ProviderOffer] = Field(default_factory=list)
    ads: list[ProviderOffer] = Field(default_factory=list)
    rent: list[ProviderOffer] = Field(default_factory=list)
    buy: list[ProviderOffer] = Field(default_factory=list)

    def offers(self) -> list[tuple[OfferType, ProviderOffer]]:
        """All offers as `(offer_type, provider)` pairs."""
        groups = (
            (OfferType.FLATRATE, self.flatrate),
            (OfferType.FREE, self.free),
            (OfferType.ADS, self.ads),
            (OfferType.RENT, self.rent),
            (OfferType.BUY, self.buy),
        )
        return [(offer_type, p) for offer_type, providers in groups for p in providers]


class WatchProviders(TMDBModel):
    """Per-country offers, keyed by ISO-3166-1 alpha-2 code."""

    results: dict[str, CountryProviders] = Field(default_factory=dict)


class Provider(TMDBModel):
    """Entry of `/watch/providers/{movie|tv}` (provider catalog for a region)."""

    provider_id: int
    provider_name: str
    logo_path: str | None = None
    display_priority: int = 0
    display_priorities: dict[str, int] = Field(default_factory=dict)


class ProviderList(TMDBModel):
    results: list[Provider] = Field(default_factory=list)


# --- Certifications ------------------------------------------------------------


class ReleaseDate(TMDBModel):
    certification: str = ""
    type: int | None = None
    release_date: str | None = None


class CountryReleaseDates(TMDBModel):
    iso_3166_1: str
    release_dates: list[ReleaseDate] = Field(default_factory=list)


class ReleaseDates(TMDBModel):
    """Movie certifications (`append_to_response=release_dates`)."""

    results: list[CountryReleaseDates] = Field(default_factory=list)


class ContentRating(TMDBModel):
    iso_3166_1: str
    rating: str = ""


class ContentRatings(TMDBModel):
    """TV certifications (`append_to_response=content_ratings`)."""

    results: list[ContentRating] = Field(default_factory=list)


# --- Translations --------------------------------------------------------------


OptionalText = Annotated[str | None, BeforeValidator(_empty_to_none)]


class TranslationData(TMDBModel):
    # Movies use `title`, TV uses `name`. TMDB sends "" when there is no translation.
    title: OptionalText = None
    overview: OptionalText = None

    @model_validator(mode="before")
    @classmethod
    def _tv_name(cls, data: Any) -> Any:
        return _rename_key(data, "name", "title")


class Translation(TMDBModel):
    iso_3166_1: str
    iso_639_1: str
    english_name: str | None = None
    data: TranslationData = Field(default_factory=TranslationData)


class Translations(TMDBModel):
    translations: list[Translation] = Field(default_factory=list)


# --- Details -------------------------------------------------------------------


class _DetailsBase(TMDBModel):
    id: int
    original_language: str
    overview: str | None = None
    genres: list[Genre] = Field(default_factory=list)
    vote_average: float | None = None
    vote_count: int = 0
    popularity: float = 0.0
    poster_path: str | None = None
    adult: bool = False
    # Appended sub-resources (present only if requested via append_to_response).
    keywords: Keywords | None = None
    credits: Credits | None = None
    watch_providers: WatchProviders | None = Field(default=None, validation_alias="watch/providers")
    translations: Translations | None = None

    @property
    def genre_ids(self) -> list[int]:
        return [g.id for g in self.genres]


class MovieDetails(_DetailsBase):
    title: str
    original_title: str
    release_date: OptionalDate = None
    runtime: int | None = None
    release_dates: ReleaseDates | None = None


class EpisodeSummary(TMDBModel):
    air_date: OptionalDate = None
    season_number: int | None = None
    episode_number: int | None = None
    runtime: int | None = None


class TVDetails(_DetailsBase):
    name: str
    original_name: str
    first_air_date: OptionalDate = None
    episode_run_time: list[int] = Field(default_factory=list)
    number_of_seasons: int | None = None
    # Fallback for runtime: `episode_run_time` is often empty.
    last_episode_to_air: EpisodeSummary | None = None
    created_by: list[Creator] = Field(default_factory=list)
    content_ratings: ContentRatings | None = None


# --- Lists (search / discover / trending) --------------------------------------


class MediaListItem(TMDBModel):
    """One result row of search, discover or trending; movie, TV or person.

    `media_type` is set by TMDB for search/trending and filled in by the client
    for discover (where the type is implied by the endpoint).
    """

    id: int
    media_type: MediaType | None = None
    title: str | None = None  # movie
    name: str | None = None  # tv / person
    original_title: str | None = None  # movie
    original_name: str | None = None  # tv
    original_language: str | None = None
    overview: str | None = None
    release_date: OptionalDate = None  # movie
    first_air_date: OptionalDate = None  # tv
    genre_ids: list[int] = Field(default_factory=list)
    poster_path: str | None = None
    popularity: float = 0.0
    vote_average: float | None = None
    vote_count: int = 0
    adult: bool = False

    @property
    def title_type(self) -> TitleType | None:
        """`movie` / `tv`, or `None` for people."""
        if self.media_type in ("movie", "tv"):
            return TitleType(self.media_type)
        return None

    @property
    def display_title(self) -> str:
        return self.title or self.name or ""

    @property
    def display_original_title(self) -> str:
        return self.original_title or self.original_name or self.display_title

    @property
    def released_on(self) -> date | None:
        return self.release_date or self.first_air_date

    @property
    def year(self) -> int | None:
        d = self.released_on
        return d.year if d else None


class Page(TMDBModel):
    page: int = 1
    results: list[MediaListItem] = Field(default_factory=list)
    total_pages: int = 0
    total_results: int = 0
