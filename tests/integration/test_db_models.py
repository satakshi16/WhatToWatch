from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from app.db.models import (
    NO_PROVIDER_ID,
    Interaction,
    InteractionKind,
    InteractionSource,
    OfferType,
    Rating,
    Title,
    TitleAvailability,
    TitleType,
    User,
    UserPlatform,
    UserPreferences,
)
from app.db.session import get_db

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_user(session: Session, email: str = "Priya@Example.com") -> User:
    user = User(
        email=email, password_hash="x", display_name="Priya", country="in", timezone="Asia/Kolkata"
    )
    session.add(user)
    session.flush()
    return user


def make_title(session: Session, tmdb_id: int = 27205, type_: TitleType = TitleType.MOVIE) -> Title:
    title = Title(
        tmdb_id=tmdb_id,
        type=type_,
        title="Inception",
        original_title="Inception",
        original_language="en",
        genres=["science-fiction", "action"],
    )
    session.add(title)
    session.flush()
    return title


def add_interaction(
    session: Session,
    user: User,
    title: Title,
    kind: InteractionKind = InteractionKind.RATING,
    value: float | None = None,
    at: datetime = T0,
) -> Interaction:
    row = Interaction(
        user_id=user.id,
        tmdb_id=title.tmdb_id,
        type=title.type,
        kind=kind,
        value=value,
        source=InteractionSource.BROWSE,
        created_at=at,
    )
    session.add(row)
    session.flush()
    return row


def ratings(session: Session) -> list[tuple[uuid.UUID, int, TitleType, float]]:
    rows = session.scalars(select(Rating).order_by(Rating.tmdb_id, Rating.type)).all()
    return [(r.user_id, r.tmdb_id, r.type, r.rating) for r in rows]


# --- ratings view ---------------------------------------------------------------------


def test_ratings_view_returns_latest_rating_per_user_and_title(db_session: Session) -> None:
    user = make_user(db_session)
    title = make_title(db_session)
    add_interaction(db_session, user, title, value=2, at=T0)
    add_interaction(db_session, user, title, value=5, at=T0 + timedelta(days=2))
    add_interaction(db_session, user, title, value=3, at=T0 + timedelta(days=1))
    db_session.commit()

    rows = db_session.scalars(select(Rating)).all()
    assert len(rows) == 1
    assert rows[0].rating == 5
    assert rows[0].rated_at == T0 + timedelta(days=2)
    assert rows[0].rated_at.tzinfo is not None


def test_ratings_view_ignores_non_rating_kinds(db_session: Session) -> None:
    user = make_user(db_session)
    title = make_title(db_session)
    add_interaction(db_session, user, title, value=4, at=T0)
    add_interaction(
        db_session, user, title, kind=InteractionKind.NOT_INTERESTED, at=T0 + timedelta(days=1)
    )
    other = make_title(db_session, tmdb_id=1)
    add_interaction(db_session, user, other, kind=InteractionKind.SEEN)
    db_session.commit()

    assert ratings(db_session) == [(user.id, title.tmdb_id, TitleType.MOVIE, 4)]


def test_ratings_view_separates_users_and_title_types(db_session: Session) -> None:
    priya = make_user(db_session)
    marcus = make_user(db_session, email="marcus@example.com")
    movie = make_title(db_session, tmdb_id=100, type_=TitleType.MOVIE)
    show = make_title(db_session, tmdb_id=100, type_=TitleType.TV)
    add_interaction(db_session, priya, movie, value=5)
    add_interaction(db_session, priya, show, value=1)
    add_interaction(db_session, marcus, movie, value=3)
    db_session.commit()

    got = set(ratings(db_session))
    assert got == {
        (priya.id, 100, TitleType.MOVIE, 5),
        (priya.id, 100, TitleType.TV, 1),
        (marcus.id, 100, TitleType.MOVIE, 3),
    }


def test_ratings_view_breaks_timestamp_ties_by_insertion_order(db_session: Session) -> None:
    user = make_user(db_session)
    title = make_title(db_session)
    add_interaction(db_session, user, title, value=1, at=T0)
    add_interaction(db_session, user, title, value=4, at=T0)
    db_session.commit()

    assert ratings(db_session) == [(user.id, title.tmdb_id, TitleType.MOVIE, 4)]


# --- constraints & model behaviour -----------------------------------------------------


@pytest.mark.parametrize("value", [0, 5.5, None])
def test_rating_value_must_be_between_1_and_5(db_session: Session, value: float | None) -> None:
    user = make_user(db_session)
    title = make_title(db_session)
    with pytest.raises(IntegrityError):
        add_interaction(db_session, user, title, value=value)


def test_foreign_keys_are_enforced_on_sqlite(db_session: Session) -> None:
    user = make_user(db_session)
    db_session.add(
        Interaction(
            user_id=user.id,
            tmdb_id=999,
            type=TitleType.MOVIE,
            kind=InteractionKind.NOT_INTERESTED,
            source=InteractionSource.BROWSE,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_enum_columns_are_check_constrained(db_session: Session) -> None:
    make_title(db_session)
    with pytest.raises(IntegrityError):
        db_session.execute(
            text(
                "INSERT INTO title_availability "
                "(tmdb_id, type, country, provider_id, offer_type, fetched_at) "
                "VALUES (27205, 'movie', 'US', 8, 'stream', '2026-01-01')"
            )
        )


def test_email_and_country_are_normalized(db_session: Session) -> None:
    user = make_user(db_session, email="  Priya@Example.COM ")
    db_session.commit()
    assert user.email == "priya@example.com"
    assert user.country == "IN"
    with pytest.raises(IntegrityError):
        db_session.execute(
            text(
                "INSERT INTO users (id, email, password_hash, display_name, country, timezone, "
                "is_admin, status, created_at) VALUES ('00000000000000000000000000000001', "
                "'Upper@Example.com', 'x', 'U', 'US', 'UTC', 0, 'active', '2026-01-01')"
            )
        )


def test_defaults_and_mutable_json_columns(db_session: Session) -> None:
    user = make_user(db_session)
    prefs = UserPreferences(user_id=user.id)
    db_session.add(prefs)
    db_session.commit()
    assert prefs.content_types == ["movie", "tv"]
    assert prefs.languages == ["en"]  # DEFAULT_LANGUAGES (P1-07)

    prefs.languages.append("hi")  # in-place mutation must be tracked
    db_session.commit()
    db_session.expire_all()
    assert db_session.get(UserPreferences, user.id).languages == ["en", "hi"]  # type: ignore[union-attr]


def test_datetimes_are_utc_aware_and_naive_rejected(db_session: Session) -> None:
    user = make_user(db_session)
    db_session.commit()
    db_session.expire_all()
    loaded = db_session.get(User, user.id)
    assert loaded is not None
    assert loaded.created_at.tzinfo is not None

    loaded.last_seen_at = datetime(2026, 1, 1)
    with pytest.raises(StatementError, match="naive datetime"):
        db_session.flush()


def test_interactions_are_append_only(db_session: Session) -> None:
    user = make_user(db_session)
    title = make_title(db_session)
    row = add_interaction(db_session, user, title, value=3)
    row.value = 4
    with pytest.raises(ValueError, match="append-only"):
        db_session.flush()


def test_availability_none_marker_requires_provider_zero(db_session: Session) -> None:
    title = make_title(db_session)
    db_session.add(
        TitleAvailability(
            tmdb_id=title.tmdb_id,
            type=title.type,
            country="US",
            provider_id=NO_PROVIDER_ID,
            offer_type=OfferType.NONE,
        )
    )
    db_session.flush()
    db_session.add(
        TitleAvailability(
            tmdb_id=title.tmdb_id,
            type=title.type,
            country="US",
            provider_id=8,
            offer_type=OfferType.NONE,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_user_cascades_to_owned_rows(db_session: Session, migrated_engine: Engine) -> None:
    user = make_user(db_session)
    title = make_title(db_session)
    db_session.add(UserPreferences(user_id=user.id))
    db_session.add(UserPlatform(user_id=user.id, provider_id=8, country="IN"))
    add_interaction(db_session, user, title, value=5)
    db_session.commit()

    db_session.execute(text("DELETE FROM users"))  # DB-level ON DELETE CASCADE
    db_session.commit()
    with migrated_engine.connect() as conn:
        for table in ("user_preferences", "user_platforms", "interactions"):
            assert conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0  # noqa: S608
        assert conn.execute(text("SELECT count(*) FROM titles")).scalar_one() == 1


# --- get_db dependency -------------------------------------------------------------------


def test_get_db_yields_session_bound_to_configured_database(
    app: FastAPI, migrated_engine: Engine
) -> None:
    @app.get("/_test/users")
    def count_users(db: Session = Depends(get_db)) -> dict[str, int]:  # noqa: B008
        return {"users": len(db.scalars(select(User)).all())}

    with Session(migrated_engine) as session:
        make_user(session)
        session.commit()

    client = TestClient(app)
    assert client.get("/_test/users").json() == {"users": 1}
