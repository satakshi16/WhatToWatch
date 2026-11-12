"""Onboarding seeds and interactions models (P1-09, FR-13, FR-23, FR-33)."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db.models.enums import InteractionKind, InteractionSource, OnboardingStep, TitleType

MIN_SEEDS = 3
MAX_SEEDS = 10
# Raw list cap (before deduplication), so a request can't carry an unbounded list.
MAX_SEED_ITEMS = 2 * MAX_SEEDS

TmdbId = Annotated[int, Field(gt=0, lt=2**31)]
Stars = Annotated[int, Field(ge=1, le=5)]


class TitleRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tmdb_id: TmdbId
    type: TitleType


class SeedsIn(BaseModel):
    """3-10 distinct titles the user likes; repeats are dropped (first kept) before counting."""

    model_config = ConfigDict(extra="forbid")

    seeds: Annotated[list[TitleRef], Field(max_length=MAX_SEED_ITEMS)]

    @field_validator("seeds", mode="after")
    @classmethod
    def _dedupe_and_count(cls, seeds: list[TitleRef]) -> list[TitleRef]:
        distinct = list(dict.fromkeys(seeds))
        if not MIN_SEEDS <= len(distinct) <= MAX_SEEDS:
            raise ValueError(
                f"needs {MIN_SEEDS} to {MAX_SEEDS} distinct titles, got {len(distinct)}"
            )
        return distinct


class SeedOut(BaseModel):
    tmdb_id: int
    type: TitleType
    title: str
    original_title: str
    # Fetched from TMDB and inserted into the catalog by this request.
    added_to_catalog: bool
    # Has a vector in the embedding index, so it shapes recommendations right away.
    embedded: bool


class SeedsOut(BaseModel):
    onboarding_step: OnboardingStep
    seeds: list[SeedOut]


class CurrentSeedOut(BaseModel):
    tmdb_id: int
    type: TitleType
    title: str
    original_title: str
    year: int | None


class CurrentSeedsOut(BaseModel):
    """The user's current seed set (empty before onboarding)."""

    seeds: list[CurrentSeedOut]


class InteractionIn(BaseModel):
    """Phase 1 kinds: `rating` (`value` = 1-5 stars) and `not_interested` (no value)."""

    model_config = ConfigDict(extra="forbid")

    tmdb_id: TmdbId
    type: TitleType
    kind: Literal["rating", "not_interested"]
    value: Stars | None = None

    @model_validator(mode="after")
    def _value_matches_kind(self) -> InteractionIn:
        if self.kind == "rating" and self.value is None:
            raise ValueError("a rating needs a value from 1 to 5")
        if self.kind == "not_interested" and self.value is not None:
            raise ValueError("not_interested takes no value")
        return self


class InteractionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tmdb_id: int
    type: TitleType
    kind: InteractionKind
    value: float | None
    source: InteractionSource
    created_at: datetime
    # Fetched from TMDB and inserted into the catalog by this request.
    added_to_catalog: bool = False
    # Has a vector in the embedding index (counts for the profile vector).
    embedded: bool = False
