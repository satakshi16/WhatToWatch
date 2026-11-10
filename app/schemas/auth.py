"""Auth request/response models (P1-06)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.schemas.common import CountryCode, DisplayName, Timezone
from app.schemas.user import UserOut

# Upper bound keeps Argon2 work bounded for absurd inputs.
PASSWORD_MIN = 8
PASSWORD_MAX = 256


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: EmailStr = Field(max_length=320)
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)
    display_name: DisplayName
    country: CountryCode
    timezone: Timezone | None = None
    age_confirmed: bool = Field(description="User confirms they are 16 or older.")

    @field_validator("email", mode="after")
    @classmethod
    def _lower_email(cls, value: str) -> str:
        return value.lower()

    @field_validator("age_confirmed", mode="after")
    @classmethod
    def _must_confirm_age(cls, value: bool) -> bool:
        if not value:
            raise ValueError("you must be 16 or older to register")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Plain string: a malformed email just fails login with the generic error.
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth2 token type, not a secret
    expires_in: int = Field(description="Access token lifetime in seconds.")


class RegisterResponse(TokenResponse):
    user: UserOut
