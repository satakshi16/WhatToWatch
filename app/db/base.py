"""Declarative base, portable column types and enum helpers (§6 portability rules).

- UUIDs use SQLAlchemy `Uuid` (CHAR(32) on SQLite, native UUID on Postgres).
- Datetimes are stored as UTC and always returned timezone-aware (`UTCDateTime`).
- List/dict columns use `JSON` (JSONB on Postgres); no `ARRAY` (D-005).
- Enums are stored as strings with a CHECK constraint (`str_enum`).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, DateTime, Dialect, Enum, MetaData, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator, TypeEngine

# Deterministic constraint names: required for Alembic batch mode on SQLite and
# keeps migrations identical across SQLite and Postgres.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """`DateTime(timezone=True)` that rejects naive input and always returns aware UTC.

    SQLite has no timezone support and hands back naive datetimes; values are
    normalised to UTC on write, so tagging them as UTC on read is lossless.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC datetimes")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def portable_json() -> TypeEngine[Any]:
    """JSON everywhere, JSONB on Postgres. A new instance per column (see `json_list`)."""
    return JSON().with_variant(JSONB(), "postgresql")


# `Mutable*.as_mutable()` binds to the *type instance*: each column needs its own,
# otherwise list and dict coercion leak onto each other's columns.
def json_list() -> TypeEngine[Any]:
    """JSON array column whose in-place mutations (append, ...) are change-tracked."""
    return MutableList.as_mutable(portable_json())


def json_dict() -> TypeEngine[Any]:
    """JSON object column whose in-place mutations are change-tracked."""
    return MutableDict.as_mutable(portable_json())


def str_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """String column + CHECK constraint (`ck_<table>_<name>`) holding the enum *values*."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: dict[Any, Any] = {  # noqa: RUF012 - SQLAlchemy class config
        uuid.UUID: Uuid(),
        datetime: UTCDateTime(),
    }


class ViewBase(DeclarativeBase):
    """Read-only mappings over SQL views.

    Separate metadata so `create_all` and Alembic autogenerate never try to create
    them as tables; the views themselves are created in migrations.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map: dict[Any, Any] = Base.type_annotation_map
