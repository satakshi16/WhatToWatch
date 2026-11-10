"""Feedback aggregate: append-only `interactions` log and the `ratings` view (§6)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    event,
)
from sqlalchemy.orm import Mapped, Mapper, mapped_column

from app.db.base import Base, ViewBase, portable_json, str_enum, utcnow
from app.db.models.enums import InteractionKind, InteractionSource, TitleType

RATING_MIN = 1
RATING_MAX = 5


class Interaction(Base):
    """One user feedback event. **Append-only**: corrections are new rows, never updates.

    `value` holds the numeric signal (rating 1-5; thumb +1/-1 in P2); `payload` holds
    structured extras (e.g. clicked provider, refine request). See D-007.
    """

    __tablename__ = "interactions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tmdb_id", "type"],
            ["titles.tmdb_id", "titles.type"],
            name="fk_interactions_title",
        ),
        CheckConstraint(
            f"kind <> '{InteractionKind.RATING.value}' OR "
            f"(value IS NOT NULL AND value >= {RATING_MIN} AND value <= {RATING_MAX})",
            name="rating_range",
        ),
        Index("ix_interactions_user_id_created_at", "user_id", "created_at"),
    )

    # Integer PK (not UUID) gives a stable insertion order to break `created_at` ties.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    tmdb_id: Mapped[int] = mapped_column(Integer)
    type: Mapped[TitleType] = mapped_column(str_enum(TitleType, "title_type"))
    kind: Mapped[InteractionKind] = mapped_column(str_enum(InteractionKind, "interaction_kind"))
    value: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict[str, Any] | None] = mapped_column(portable_json())
    source: Mapped[InteractionSource] = mapped_column(
        str_enum(InteractionSource, "interaction_source")
    )
    # FK to rec_batches is added with that table in Phase 2.
    batch_id: Mapped[uuid.UUID | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


@event.listens_for(Interaction, "before_update")
def _forbid_interaction_update(_mapper: Mapper[Any], _conn: Any, target: Interaction) -> None:
    raise ValueError(f"interactions are append-only (id={target.id}); insert a new row instead")


class Rating(ViewBase):
    """Read-only `ratings` view: latest `rating` interaction per (user, title).

    Created in Alembic revision 0001; ties on `created_at` resolve to the higher `id`.
    """

    __tablename__ = "ratings"

    user_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    type: Mapped[TitleType] = mapped_column(str_enum(TitleType, "title_type"), primary_key=True)
    rating: Mapped[float] = mapped_column(Float)
    rated_at: Mapped[datetime]
