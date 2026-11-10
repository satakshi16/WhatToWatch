"""Catalog aggregate: `titles`, `title_availability`, `providers` (§6)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, json_dict, json_list, str_enum, utcnow
from app.db.models.enums import OfferType, TitleType

# Provider ID used by the "checked, none available" availability marker row.
NO_PROVIDER_ID = 0


class Title(Base):
    """One TMDB movie or series. Natural key is (tmdb_id, type): IDs overlap across types."""

    __tablename__ = "titles"
    __table_args__ = (Index("ix_titles_original_language_type", "original_language", "type"),)

    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    type: Mapped[TitleType] = mapped_column(str_enum(TitleType, "title_type"), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    original_title: Mapped[str] = mapped_column(String(500))
    original_language: Mapped[str] = mapped_column(String(16))
    overview: Mapped[str | None] = mapped_column(Text)
    # Canonical genre slugs; first = primary genre.
    genres: Mapped[list[str]] = mapped_column(json_list(), default=list)
    keywords: Mapped[list[str]] = mapped_column(json_list(), default=list)
    directors: Mapped[list[str]] = mapped_column(json_list(), default=list)
    top_cast: Mapped[list[str]] = mapped_column(json_list(), default=list)
    release_date: Mapped[date | None] = mapped_column(Date)
    # Movies: minutes; TV: average episode minutes.
    runtime: Mapped[int | None] = mapped_column(Integer)
    seasons: Mapped[int | None] = mapped_column(Integer)
    vote_avg: Mapped[float | None] = mapped_column(Float)
    vote_count: Mapped[int] = mapped_column(Integer, default=0)
    popularity: Mapped[float] = mapped_column(Float, default=0.0)
    poster_path: Mapped[str | None] = mapped_column(String(255))
    adult: Mapped[bool] = mapped_column(Boolean, default=False)
    # Country (alpha-2) -> certification, for seeded countries.
    certification: Mapped[dict[str, Any]] = mapped_column(json_dict(), default=dict)
    # Language/country code -> localized title.
    localized_titles: Mapped[dict[str, Any]] = mapped_column(json_dict(), default=dict)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class TitleAvailability(Base):
    """Cached TMDB watch-provider offer (JustWatch data). Refreshed ≤ AVAILABILITY_TTL_DAYS.

    A title checked with no offers in a country gets one marker row with
    `provider_id = NO_PROVIDER_ID` and `offer_type = none`, to avoid re-fetching.
    """

    __tablename__ = "title_availability"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tmdb_id", "type"],
            ["titles.tmdb_id", "titles.type"],
            ondelete="CASCADE",
            name="fk_title_availability_title",
        ),
        CheckConstraint(
            f"(provider_id = {NO_PROVIDER_ID}) = (offer_type = '{OfferType.NONE.value}')",
            name="none_marker",
        ),
        CheckConstraint("length(country) = 2 AND country = upper(country)", name="country_alpha2"),
        Index("ix_title_availability_country_title", "country", "tmdb_id", "type"),
    )

    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    type: Mapped[TitleType] = mapped_column(str_enum(TitleType, "title_type"), primary_key=True)
    country: Mapped[str] = mapped_column(String(2), primary_key=True)
    provider_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    offer_type: Mapped[OfferType] = mapped_column(
        str_enum(OfferType, "offer_type"), primary_key=True
    )
    link: Mapped[str | None] = mapped_column(String(1000))
    fetched_at: Mapped[datetime] = mapped_column(default=utcnow)


class Provider(Base):
    """Cached TMDB watch-provider list per country (union of movie + tv lists)."""

    __tablename__ = "providers"
    __table_args__ = (
        CheckConstraint("length(country) = 2 AND country = upper(country)", name="country_alpha2"),
    )

    provider_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    country: Mapped[str] = mapped_column(String(2), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    logo_path: Mapped[str | None] = mapped_column(String(255))
    display_priority: Mapped[int | None] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(default=utcnow)
