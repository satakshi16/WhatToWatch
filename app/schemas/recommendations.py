"""`GET /recommendations` response models (P1-12, FR-26, FR-29; §7 item shape)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import AfterValidator, BaseModel, Field

from app.db.models.enums import OfferType, TitleType

MAX_RECOMMENDATIONS = 20


def _absolute_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("must be an absolute http(s) URL")
    return value


AbsoluteUrl = Annotated[str, AfterValidator(_absolute_url)]


class RecommendedProviderOut(BaseModel):
    """A streamable offer on one of the user's platforms (FR-22)."""

    provider_id: int
    name: str | None
    logo_url: AbsoluteUrl | None
    offer_type: OfferType


class ScoreComponentsOut(BaseModel):
    sim: float
    quality: float
    pop: float
    recency: float
    neg: float
    score: float


class RecommendationItemOut(BaseModel):
    """§7 recommendation item (shared by on-demand and, from P2-05, weekly lists)."""

    rank: int
    tmdb_id: int
    type: TitleType
    title: str = Field(description="Localized to the user's first language when available.")
    original_title: str
    year: int | None
    language: str | None
    runtime_min: int | None
    seasons: int | None
    genres: list[str]
    poster_url: AbsoluteUrl | None
    providers: list[RecommendedProviderOut]
    watch_link: AbsoluteUrl
    availability_checked_at: datetime | None
    reason: str
    reason_source: Literal["template", "llm"]
    slot: Literal["core", "fresh", "wildcard"]
    score: float
    # Admin `?debug=true` only.
    pool: str | None = None
    components: ScoreComponentsOut | None = None


class AttributionOut(BaseModel):
    """FR-29 / §5.7: data providers to credit wherever these results are shown."""

    availability: Literal["JustWatch"] = "JustWatch"
    metadata: Literal["TMDB"] = "TMDB"


class RecommendationsOut(BaseModel):
    n: int = Field(description="Requested list size; fewer items mean too few titles qualified.")
    cold_start: bool
    generated_at: datetime
    config_hash: str
    model_version: str | None
    items: list[RecommendationItemOut]
    attribution: AttributionOut = Field(default_factory=AttributionOut)
    debug: dict[str, Any] | None = Field(
        default=None, description="Admin `?debug=true` only: pipeline counts, caps, timings."
    )
