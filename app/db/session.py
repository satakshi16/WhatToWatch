"""Database engine, session factory and the `get_db` FastAPI dependency."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


def _ensure_sqlite_dir(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() == "sqlite" and url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)


def _sqlite_enable_foreign_keys(dbapi_conn: Any, _record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_engine(database_url: str) -> Engine:
    """New engine; on SQLite, creates the parent dir and enforces foreign keys."""
    _ensure_sqlite_dir(database_url)
    engine = create_engine(database_url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _sqlite_enable_foreign_keys)
    return engine


@lru_cache
def get_engine(database_url: str) -> Engine:
    """Process-wide engine per URL."""
    return create_db_engine(database_url)


@lru_cache
def get_sessionmaker(database_url: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> Iterator[Session]:
    """Request-scoped session. Endpoints commit explicitly; uncommitted work is rolled back."""
    with get_sessionmaker(settings.database_url)() as session:
        yield session


def check_database(engine: Engine) -> None:
    """Raise if the database is unreachable."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
