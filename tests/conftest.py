from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import create_db_engine
from app.main import create_app

# >= 32 bytes, so PyJWT's short-HMAC-key warning (an error under pytest) stays quiet.
TEST_JWT_KEY = "test-signing-key-0123456789abcdef-0123456789"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        log_level="INFO",
        jwt_secret_keys=TEST_JWT_KEY,
        # Never the real ./data index: tests that need one build it here.
        embeddings_dir=tmp_path / "embeddings",
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


REPO_ROOT = Path(__file__).resolve().parents[1]


def alembic_config(database_url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    cfg.attributes["skip_logging_config"] = True
    return cfg


@pytest.fixture
def migrated_engine(settings: Settings) -> Iterator[Engine]:
    """Fresh SQLite DB at `settings.database_url`, migrated to head."""
    command.upgrade(alembic_config(settings.database_url), "head")
    engine = create_db_engine(settings.database_url)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(migrated_engine: Engine) -> Iterator[Session]:
    with Session(migrated_engine, expire_on_commit=False) as session:
        yield session
