"""User aggregate: `users`, `user_preferences`, `user_platforms` (§6)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.base import Base, json_dict, json_list, str_enum, utcnow
from app.db.models.enums import Era, OnboardingStep, TitleType, UserStatus

COUNTRY_CHECK = "length({col}) = 2 AND {col} = upper({col})"


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("email = lower(email)", name="email_lowercase"),
        CheckConstraint(COUNTRY_CHECK.format(col="country"), name="country_alpha2"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(2))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[UserStatus] = mapped_column(
        str_enum(UserStatus, "user_status"), default=UserStatus.ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_seen_at: Mapped[datetime | None]

    preferences: Mapped[UserPreferences | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )
    platforms: Mapped[list[UserPlatform]] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True
    )

    @validates("email")
    def _normalize_email(self, _key: str, value: str) -> str:
        return value.strip().lower()

    @validates("country")
    def _normalize_country(self, _key: str, value: str) -> str:
        return value.strip().upper()


# Preferred languages when the user hasn't chosen any (P1-07).
DEFAULT_LANGUAGES: tuple[str, ...] = ("en",)


def _default_languages() -> list[str]:
    return list(DEFAULT_LANGUAGES)


def _default_content_types() -> list[str]:
    return [TitleType.MOVIE.value, TitleType.TV.value]


class UserPreferences(Base):
    __tablename__ = "user_preferences"
    __table_args__ = (
        CheckConstraint("max_runtime IS NULL OR max_runtime > 0", name="max_runtime_positive"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    # ISO-639-1 codes as used by TMDB `original_language`.
    languages: Mapped[list[str]] = mapped_column(json_list(), default=_default_languages)
    # Subset of TitleType values.
    content_types: Mapped[list[str]] = mapped_column(json_list(), default=_default_content_types)
    era: Mapped[Era] = mapped_column(str_enum(Era, "era"), default=Era.ANY)
    # Canonical genre slugs (P1-03).
    avoid_genres: Mapped[list[str]] = mapped_column(json_list(), default=list)
    max_runtime: Mapped[int | None] = mapped_column(Integer)
    include_rent_buy: Mapped[bool] = mapped_column(Boolean, default=False)
    allow_adult: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarding_step: Mapped[OnboardingStep] = mapped_column(
        str_enum(OnboardingStep, "onboarding_step"), default=OnboardingStep.PREFERENCES
    )
    onboarding_state: Mapped[dict[str, Any]] = mapped_column(json_dict(), default=dict)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    user: Mapped[User] = relationship(back_populates="preferences")


class UserPlatform(Base):
    """A streaming provider the user subscribes to, in a given country (TMDB provider IDs)."""

    __tablename__ = "user_platforms"
    __table_args__ = (CheckConstraint(COUNTRY_CHECK.format(col="country"), name="country_alpha2"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    provider_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    country: Mapped[str] = mapped_column(String(2), primary_key=True)

    user: Mapped[User] = relationship(back_populates="platforms")
