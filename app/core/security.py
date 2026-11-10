"""Password hashing (Argon2id) and JWT encode/decode (P1-06, NFR-5).

Tokens are HS256 JWTs signed with the first key of `JWT_SECRET_KEYS`. Each token's
header carries a `kid` derived from its key, so a new key can be prepended later
and tokens signed with an older key still verify until that key is removed.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from pydantic import SecretStr

from app.core.config import Settings

ALGORITHM = "HS256"
REQUIRED_CLAIMS = ["sub", "exp", "iat", "jti", "type"]

# argon2-cffi defaults to Argon2id with the RFC 9106 low-memory profile.
_hasher = PasswordHasher()
# Verified against when the user doesn't exist, so login timing doesn't reveal accounts.
_DUMMY_HASH = _hasher.hash("dummy-password-for-timing")


class TokenType(StrEnum):
    ACCESS = "access"
    REFRESH = "refresh"


class InvalidTokenError(Exception):
    """Token is malformed, expired, signed with an unknown key or of the wrong type."""


@dataclass(frozen=True, slots=True)
class TokenClaims:
    sub: uuid.UUID
    jti: str
    type: TokenType
    iat: datetime
    exp: datetime


# --- Passwords ---


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    """Constant-ish time check; with `password_hash=None` it still does the work, then fails."""
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


# --- JWT ---


def key_id(key: SecretStr) -> str:
    """Stable, non-reversible identifier for a signing key."""
    return hashlib.sha256(key.get_secret_value().encode()).hexdigest()[:16]


def _ttl(settings: Settings, token_type: TokenType) -> timedelta:
    if token_type is TokenType.ACCESS:
        return timedelta(minutes=settings.access_token_ttl_min)
    return timedelta(days=settings.refresh_token_ttl_days)


def create_token(
    settings: Settings,
    user_id: uuid.UUID,
    token_type: TokenType,
    *,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now(UTC)
    key = settings.jwt_secret_keys[0]
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int((now + _ttl(settings, token_type)).timestamp()),
        "jti": uuid.uuid4().hex,
        "type": token_type.value,
    }
    return jwt.encode(
        payload, key.get_secret_value(), algorithm=ALGORITHM, headers={"kid": key_id(key)}
    )


def decode_token(settings: Settings, token: str, expected_type: TokenType) -> TokenClaims:
    """Verify signature, expiry, required claims and token type; raise `InvalidTokenError`."""
    try:
        kid = jwt.get_unverified_header(token).get("kid")
        key = next((k for k in settings.jwt_secret_keys if key_id(k) == kid), None)
        if key is None:
            raise InvalidTokenError("unknown signing key")
        payload = jwt.decode(
            token,
            key.get_secret_value(),
            algorithms=[ALGORITHM],
            options={"require": REQUIRED_CLAIMS},
        )
        if payload["type"] != expected_type.value:
            raise InvalidTokenError("wrong token type")
        return TokenClaims(
            sub=uuid.UUID(payload["sub"]),
            jti=str(payload["jti"]),
            type=TokenType(payload["type"]),
            iat=datetime.fromtimestamp(payload["iat"], UTC),
            exp=datetime.fromtimestamp(payload["exp"], UTC),
        )
    except InvalidTokenError:
        raise
    except (jwt.PyJWTError, ValueError, TypeError, KeyError) as exc:
        raise InvalidTokenError(type(exc).__name__) from exc
