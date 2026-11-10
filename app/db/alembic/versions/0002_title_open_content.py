"""`title_open_content`: Wikidata + Wikipedia item content (P1-04b, D-011).

Self-contained on purpose (no `app.*` imports) so it never changes when models do.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "title_open_content",
        sa.Column("tmdb_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("type", sa.String(length=5), nullable=False),
        sa.Column("wikidata_qid", sa.String(length=16), nullable=True),
        sa.Column("label", sa.String(length=500), nullable=True),
        sa.Column("original_language", sa.String(length=16), nullable=True),
        sa.Column("genres", JSON, nullable=False),
        sa.Column("directors", JSON, nullable=False),
        sa.Column("top_cast", JSON, nullable=False),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("wiki_site", sa.String(length=32), nullable=True),
        sa.Column("wiki_title", sa.String(length=500), nullable=True),
        sa.Column("wiki_revision_id", sa.BigInteger(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_license", sa.String(length=32), nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("type IN ('movie', 'tv')", name="ck_title_open_content_title_type"),
        sa.CheckConstraint(
            "summary IS NULL OR (summary_license IS NOT NULL AND wiki_title IS NOT NULL)",
            name="ck_title_open_content_summary_attributed",
        ),
        sa.ForeignKeyConstraint(
            ["tmdb_id", "type"],
            ["titles.tmdb_id", "titles.type"],
            name="fk_title_open_content_title",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("tmdb_id", "type", name="pk_title_open_content"),
    )


def downgrade() -> None:
    op.drop_table("title_open_content")
