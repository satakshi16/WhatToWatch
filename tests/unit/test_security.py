from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.core.security import (
    InvalidTokenError,
    TokenType,
    create_token,
    decode_token,
    hash_password,
    key_id,
    verify_password,
)

KEY_A = "a" * 40
KEY_B = "b" * 40


def _settings(*keys: str) -> Settings:
    return Settings(_env_file=None, jwt_secret_keys=",".join(keys))  # type: ignore[call-arg]


def test_hash_is_argon2id_and_verifies() -> None:
    h = hash_password("correct horse")
    assert h.startswith("$argon2id$")
    assert "correct horse" not in h
    assert verify_password("correct horse", h)
    assert not verify_password("wrong horse", h)


def test_verify_handles_missing_and_garbage_hash() -> None:
    assert not verify_password("anything", None)
    assert not verify_password("anything", "not-a-hash")


def test_access_token_claims() -> None:
    s = _settings(KEY_A)
    uid = uuid.uuid4()
    now = datetime.now(UTC).replace(microsecond=0)
    token = create_token(s, uid, TokenType.ACCESS, now=now)
    assert jwt.get_unverified_header(token) == {
        "alg": "HS256",
        "typ": "JWT",
        "kid": key_id(SecretStr(KEY_A)),
    }
    claims = decode_token(s, token, TokenType.ACCESS)
    assert claims.sub == uid
    assert claims.type is TokenType.ACCESS
    assert claims.exp - claims.iat == timedelta(minutes=30)
    assert len(claims.jti) == 32


def test_refresh_token_lives_7_days_and_is_not_an_access_token() -> None:
    s = _settings(KEY_A)
    token = create_token(s, uuid.uuid4(), TokenType.REFRESH)
    claims = decode_token(s, token, TokenType.REFRESH)
    assert claims.exp - claims.iat == timedelta(days=7)
    with pytest.raises(InvalidTokenError):
        decode_token(s, token, TokenType.ACCESS)


def test_jti_is_unique() -> None:
    s = _settings(KEY_A)
    uid = uuid.uuid4()
    t1, t2 = (create_token(s, uid, TokenType.ACCESS) for _ in range(2))
    assert decode_token(s, t1, TokenType.ACCESS).jti != decode_token(s, t2, TokenType.ACCESS).jti


def test_expired_token_rejected() -> None:
    s = _settings(KEY_A)
    token = create_token(
        s, uuid.uuid4(), TokenType.ACCESS, now=datetime.now(UTC) - timedelta(hours=1)
    )
    with pytest.raises(InvalidTokenError):
        decode_token(s, token, TokenType.ACCESS)


def test_key_rotation_old_tokens_verify_until_key_removed() -> None:
    old = _settings(KEY_A)
    token = create_token(old, uuid.uuid4(), TokenType.ACCESS)
    rotated = _settings(KEY_B, KEY_A)
    decode_token(rotated, token, TokenType.ACCESS)
    new_token = create_token(rotated, uuid.uuid4(), TokenType.ACCESS)
    assert jwt.get_unverified_header(new_token)["kid"] == key_id(SecretStr(KEY_B))
    with pytest.raises(InvalidTokenError):
        decode_token(_settings(KEY_B), token, TokenType.ACCESS)


@pytest.mark.parametrize(
    "token",
    [
        "garbage",
        "",
        # alg=none, no signature
        jwt.encode({"sub": str(uuid.uuid4())}, key=None, algorithm="none"),
    ],
)
def test_malformed_tokens_rejected(token: str) -> None:
    with pytest.raises(InvalidTokenError):
        decode_token(_settings(KEY_A), token, TokenType.ACCESS)


def test_tampered_or_missing_claims_rejected() -> None:
    s = _settings(KEY_A)
    headers = {"kid": key_id(SecretStr(KEY_A))}
    no_jti = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": 2**33, "iat": 0, "type": "access"},
        KEY_A,
        algorithm="HS256",
        headers=headers,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(s, no_jti, TokenType.ACCESS)
    bad_sub = jwt.encode(
        {"sub": "not-a-uuid", "exp": 2**33, "iat": 0, "jti": "x", "type": "access"},
        KEY_A,
        algorithm="HS256",
        headers=headers,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(s, bad_sub, TokenType.ACCESS)
    forged = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": 2**33, "iat": 0, "jti": "x", "type": "access"},
        KEY_B,
        algorithm="HS256",
        headers=headers,
    )
    with pytest.raises(InvalidTokenError):
        decode_token(s, forged, TokenType.ACCESS)
