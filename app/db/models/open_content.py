"""Open-licensed item content: `title_open_content` (P1-04b, D-011).

The **only** source of item content for the ML layer: Wikidata (CC0) structured
fields and Wikipedia (CC BY-SA 4.0) lead summaries, joined to `titles` by TMDB ID.
Nothing in this table may be copied from TMDB fields; `tmdb_id` / `type` are join keys.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, json_list, str_enum, utcnow
from app.db.models.enums import TitleType

WIKIPEDIA_LICENSE = "CC BY-SA 4.0"


class TitleOpenContent(Base):
    """One row per title checked against Wikidata; `wikidata_qid IS NULL` = no match."""

    __tablename__ = "title_open_content"
    __table_args__ = (
        ForeignKeyConstraint(
            ["tmdb_id", "type"],
            ["titles.tmdb_id", "titles.type"],
            ondelete="CASCADE",
            name="fk_title_open_content_title",
        ),
        CheckConstraint(
            "summary IS NULL OR (summary_license IS NOT NULL AND wiki_title IS NOT NULL)",
            name="summary_attributed",
        ),
    )

    tmdb_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    type: Mapped[TitleType] = mapped_column(str_enum(TitleType, "title_type"), primary_key=True)
    wikidata_qid: Mapped[str | None] = mapped_column(String(16))
    label: Mapped[str | None] = mapped_column(String(500))
    # ISO 639-1 code from Wikidata P364 → P218.
    original_language: Mapped[str | None] = mapped_column(String(16))
    genres: Mapped[list[str]] = mapped_column(json_list(), default=list)
    directors: Mapped[list[str]] = mapped_column(json_list(), default=list)
    top_cast: Mapped[list[str]] = mapped_column(json_list(), default=list)
    publication_date: Mapped[date | None] = mapped_column(Date)
    # Wikimedia database name, e.g. `enwiki`.
    wiki_site: Mapped[str | None] = mapped_column(String(32))
    wiki_title: Mapped[str | None] = mapped_column(String(500))
    wiki_revision_id: Mapped[int | None] = mapped_column(BigInteger)
    summary: Mapped[str | None] = mapped_column(Text)
    summary_license: Mapped[str | None] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(default=utcnow)

    @property
    def has_open_content(self) -> bool:
        """True if the title can be embedded (§8.1: needs a Wikipedia summary)."""
        return bool(self.summary and self.summary.strip())
