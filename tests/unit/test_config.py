from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import AppEnv, Settings


def test_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.app_env is AppEnv.LOCAL
    assert s.seed_countries == ["US"]
    assert len(s.seed_languages) == 9
    assert s.adult_certifications == {"US": ["NC-17"]}


def test_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    monkeypatch.setenv("JWT_SECRET_KEYS", "new,old")
    monkeypatch.setenv("SEED_COUNTRIES", "us,in")
    monkeypatch.setenv("ADULT_CERTIFICATIONS", '{"US": ["NC-17", "X"]}')
    monkeypatch.setenv("log_level", "debug")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.cors_origins == ["http://a.test", "http://b.test"]
    assert [k.get_secret_value() for k in s.jwt_secret_keys] == ["new", "old"]
    assert s.seed_countries == ["US", "IN"]
    assert s.adult_certifications["US"] == ["NC-17", "X"]
    assert s.log_level == "DEBUG"


def test_secrets_not_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TMDB_API_KEY", "super-secret-key")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert "super-secret-key" not in repr(s)


def test_env_example_lists_every_setting_and_loads() -> None:
    example = Path(__file__).resolve().parents[2] / ".env.example"
    keys = {
        line.split("=", 1)[0].strip().lower()
        for line in example.read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert set(Settings.model_fields) <= keys
    s = Settings(_env_file=example)  # type: ignore[call-arg]
    assert s.api_base_url == "http://localhost:8000"  # inline comment stripped


def test_empty_jwt_keys_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET_KEYS", " , ")
    with pytest.raises(ValueError, match="JWT_SECRET_KEYS"):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("env", ["dev", "prod"])
@pytest.mark.parametrize("key", ["change-me", "short-key"])
def test_weak_jwt_keys_rejected_outside_local(
    monkeypatch: pytest.MonkeyPatch, env: str, key: str
) -> None:
    monkeypatch.setenv("APP_ENV", env)
    monkeypatch.setenv("JWT_SECRET_KEYS", f"{'k' * 40},{key}")
    with pytest.raises(ValueError, match="JWT_SECRET_KEYS"):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_strong_jwt_keys_accepted_in_prod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SECRET_KEYS", "k" * 32)
    assert Settings(_env_file=None).app_env is AppEnv.PROD  # type: ignore[call-arg]
