"""Initial schema: Phase 1 tables + `ratings` view (P1-02).

Self-contained on purpose (no `app.*` imports) so it never changes when models do.

Revision ID: 0001
Revises:
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TS = sa.DateTime(timezone=True)

TITLE_TYPES = ("movie", "tv")
USER_STATUSES = ("active", "paused", "deleted")
ERAS = ("last_2y", "last_10y", "any", "classics_pre_2000")
ONBOARDING_STEPS = ("preferences", "seeds", "quick_rate", "done")
OFFER_TYPES = ("flatrate", "free", "ads", "rent", "buy", "none")
INTERACTION_KINDS = (
    "rating",
    "thumb",
    "seen",
    "not_interested",
    "watchlist",
    "click",
    "refine",
)
INTERACTION_SOURCES = ("onboarding_seed", "quick_rate", "weekly", "browse", "agent")


def _enum(values: tuple[str, ...]) -> sa.String:
    return sa.String(length=max(len(v) for v in values))


def _enum_check(table: str, column: str, name: str, values: tuple[str, ...]) -> sa.CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in values)
    return sa.CheckConstraint(f"{column} IN ({allowed})", name=f"ck_{table}_{name}")


def _country_check(table: str) -> sa.CheckConstraint:
    return sa.CheckConstraint(
        "length(country) = 2 AND country = upper(country)", name=f"ck_{table}_country_alpha2"
    )


RATINGS_VIEW = """
CREATE VIEW ratings AS
SELECT user_id, tmdb_id, type, value AS rating, created_at AS rated_at
FROM (
    SELECT user_id, tmdb_id, type, value, created_at,
           ROW_NUMBER() OVER (
               PARTITION BY user_id, tmdb_id, type
               ORDER BY created_at DESC, id DESC
           ) AS rn
    FROM interactions
    WHERE kind = 'rating'
) latest
WHERE rn = 1
"""


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("timezone", sa.String(length=64), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False),
        sa.Column("status", _enum(USER_STATUSES), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("last_seen_at", TS, nullable=True),
        sa.CheckConstraint("email = lower(email)", name="ck_users_email_lowercase"),
        _country_check("users"),
        _enum_check("users", "status", "user_status", USER_STATUSES),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    op.create_table(
        "user_preferences",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("languages", JSON, nullable=False),
        sa.Column("content_types", JSON, nullable=False),
        sa.Column("era", _enum(ERAS), nullable=False),
        sa.Column("avoid_genres", JSON, nullable=False),
        sa.Column("max_runtime", sa.Integer(), nullable=True),
        sa.Column("include_rent_buy", sa.Boolean(), nullable=False),
        sa.Column("allow_adult", sa.Boolean(), nullable=False),
        sa.Column("onboarding_step", _enum(ONBOARDING_STEPS), nullable=False),
        sa.Column("onboarding_state", JSON, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
        sa.CheckConstraint(
            "max_runtime IS NULL OR max_runtime > 0",
            name="ck_user_preferences_max_runtime_positive",
        ),
        _enum_check("user_preferences", "era", "era", ERAS),
        _enum_check("user_preferences", "onboarding_step", "onboarding_step", ONBOARDING_STEPS),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_preferences_user_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", name="pk_user_preferences"),
    )

    op.create_table(
        "user_platforms",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("provider_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        _country_check("user_platforms"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_user_platforms_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("user_id", "provider_id", "country", name="pk_user_platforms"),
    )

    op.create_table(
        "providers",
        sa.Column("provider_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("logo_path", sa.String(length=255), nullable=True),
        sa.Column("display_priority", sa.Integer(), nullable=True),
        sa.Column("fetched_at", TS, nullable=False),
        _country_check("providers"),
        sa.PrimaryKeyConstraint("provider_id", "country", name="pk_providers"),
    )

    op.create_table(
        "titles",
        sa.Column("tmdb_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("type", _enum(TITLE_TYPES), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("original_title", sa.String(length=500), nullable=False),
        sa.Column("original_language", sa.String(length=16), nullable=False),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("genres", JSON, nullable=False),
        sa.Column("keywords", JSON, nullable=False),
        sa.Column("directors", JSON, nullable=False),
        sa.Column("top_cast", JSON, nullable=False),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column("runtime", sa.Integer(), nullable=True),
        sa.Column("seasons", sa.Integer(), nullable=True),
        sa.Column("vote_avg", sa.Float(), nullable=True),
        sa.Column("vote_count", sa.Integer(), nullable=False),
        sa.Column("popularity", sa.Float(), nullable=False),
        sa.Column("poster_path", sa.String(length=255), nullable=True),
        sa.Column("adult", sa.Boolean(), nullable=False),
        sa.Column("certification", JSON, nullable=False),
        sa.Column("localized_titles", JSON, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
        _enum_check("titles", "type", "title_type", TITLE_TYPES),
        sa.PrimaryKeyConstraint("tmdb_id", "type", name="pk_titles"),
    )
    op.create_index("ix_titles_original_language_type", "titles", ["original_language", "type"])

    op.create_table(
        "title_availability",
        sa.Column("tmdb_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("type", _enum(TITLE_TYPES), nullable=False),
        sa.Column("country", sa.String(length=2), nullable=False),
        sa.Column("provider_id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("offer_type", _enum(OFFER_TYPES), nullable=False),
        sa.Column("link", sa.String(length=1000), nullable=True),
        sa.Column("fetched_at", TS, nullable=False),
        sa.CheckConstraint(
            "(provider_id = 0) = (offer_type = 'none')", name="ck_title_availability_none_marker"
        ),
        _country_check("title_availability"),
        _enum_check("title_availability", "type", "title_type", TITLE_TYPES),
        _enum_check("title_availability", "offer_type", "offer_type", OFFER_TYPES),
        sa.ForeignKeyConstraint(
            ["tmdb_id", "type"],
            ["titles.tmdb_id", "titles.type"],
            name="fk_title_availability_title",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "tmdb_id", "type", "country", "provider_id", "offer_type", name="pk_title_availability"
        ),
    )
    op.create_index(
        "ix_title_availability_country_title",
        "title_availability",
        ["country", "tmdb_id", "type"],
    )

    op.create_table(
        "interactions",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("tmdb_id", sa.Integer(), nullable=False),
        sa.Column("type", _enum(TITLE_TYPES), nullable=False),
        sa.Column("kind", _enum(INTERACTION_KINDS), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("payload", JSON, nullable=True),
        sa.Column("source", _enum(INTERACTION_SOURCES), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", TS, nullable=False),
        sa.CheckConstraint(
            "kind <> 'rating' OR (value IS NOT NULL AND value >= 1 AND value <= 5)",
            name="ck_interactions_rating_range",
        ),
        _enum_check("interactions", "type", "title_type", TITLE_TYPES),
        _enum_check("interactions", "kind", "interaction_kind", INTERACTION_KINDS),
        _enum_check("interactions", "source", "interaction_source", INTERACTION_SOURCES),
        sa.ForeignKeyConstraint(
            ["tmdb_id", "type"], ["titles.tmdb_id", "titles.type"], name="fk_interactions_title"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_interactions_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_interactions"),
    )
    op.create_index("ix_interactions_user_id_created_at", "interactions", ["user_id", "created_at"])

    op.execute(RATINGS_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS ratings")
    op.drop_index("ix_interactions_user_id_created_at", table_name="interactions")
    op.drop_table("interactions")
    op.drop_index("ix_title_availability_country_title", table_name="title_availability")
    op.drop_table("title_availability")
    op.drop_index("ix_titles_original_language_type", table_name="titles")
    op.drop_table("titles")
    op.drop_table("providers")
    op.drop_table("user_platforms")
    op.drop_table("user_preferences")
    op.drop_table("users")
