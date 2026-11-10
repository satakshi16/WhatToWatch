"""Profile models (P1-06)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.db.models.enums import UserStatus
from app.schemas.common import CountryCode, DisplayName, Timezone


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: str
    country: str
    timezone: str
    is_admin: bool
    status: UserStatus
    created_at: datetime
    last_seen_at: datetime | None


class UserUpdate(BaseModel):
    """`PATCH /me`: only the fields sent are changed; none of them may be null."""

    model_config = ConfigDict(extra="forbid")

    display_name: DisplayName | None = None
    country: CountryCode | None = None
    timezone: Timezone | None = None

    @model_validator(mode="after")
    def _no_nulls(self) -> UserUpdate:
        nulls = [name for name in self.model_fields_set if getattr(self, name) is None]
        if nulls:
            raise ValueError(f"fields cannot be null: {', '.join(sorted(nulls))}")
        return self


class UserUpdateOut(UserOut):
    """`PATCH /me` response: the profile plus platforms dropped by a country change (P1-07)."""

    removed_platforms: list[int] = Field(
        default_factory=list,
        description="Provider IDs removed because they are not available in the new country.",
    )
