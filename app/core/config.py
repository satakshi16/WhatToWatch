"""Application settings — the single source of configuration (§5.2, §12).

Values are loaded from environment variables and an optional `.env` file.
Every key here must also appear in `.env.example`.
"""

from __future__ import annotations

import json
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class AppEnv(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    PROD = "prod"


class VectorBackend(StrEnum):
    NUMPY = "numpy"
    PGVECTOR = "pgvector"


CommaList = Annotated[list[str], NoDecode]

# RFC 7518 §3.2: HS256 keys must be at least as long as the hash output.
MIN_JWT_KEY_BYTES = 32
PLACEHOLDER_JWT_KEY = "change-me"


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---
    app_env: AppEnv = AppEnv.LOCAL
    log_level: str = "INFO"
    api_base_url: str = "http://localhost:8000"
    cors_origins: CommaList = Field(default_factory=lambda: ["http://localhost:8501"])

    # --- Database ---
    database_url: str = "sqlite:///./data/whattowatch.db"

    # --- Auth ---
    jwt_secret_keys: Annotated[list[SecretStr], NoDecode] = Field(
        default_factory=lambda: [SecretStr("change-me")]
    )
    access_token_ttl_min: int = Field(default=30, gt=0)
    refresh_token_ttl_days: int = Field(default=7, gt=0)

    # --- TMDB ---
    tmdb_api_key: SecretStr | None = None
    tmdb_read_token: SecretStr | None = None
    tmdb_max_concurrency: int = Field(default=20, gt=0)
    tmdb_timeout_seconds: float = Field(default=10.0, gt=0)
    # Total tries per request (first try + retries) on 5xx / 429 / network errors.
    tmdb_max_attempts: int = Field(default=4, gt=0)

    # --- Catalog / ingest ---
    seed_languages: CommaList = Field(
        default_factory=lambda: ["en", "hi", "ko", "ja", "es", "fr", "ta", "te", "ml"]
    )
    seed_countries: CommaList = Field(default_factory=lambda: ["US"])
    default_country: str = "US"
    ingest_pages_per_language: int = Field(default=10, gt=0)
    # `GET /providers` refreshes a country's cached provider list after this many days (P1-07).
    providers_ttl_days: int = Field(default=7, gt=0)
    # `GET /titles/search` keeps TMDB search results in memory this long (P1-08).
    title_search_cache_ttl_seconds: int = Field(default=300, ge=0)
    title_search_cache_max_entries: int = Field(default=1000, gt=0)

    # --- Open content: Wikidata + Wikipedia (P1-04b, D-011) ---
    # Contact for Wikimedia's User-Agent policy (email or URL). Required by open-content.
    wikimedia_contact: str | None = None
    wikimedia_concurrency: int = Field(default=4, gt=0)
    wikimedia_timeout_seconds: float = Field(default=30.0, gt=0)
    wikimedia_max_attempts: int = Field(default=4, gt=0)

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-small"
    embeddings_dir: Path = Path("./data")
    embed_batch_size: int = Field(default=32, gt=0)
    onehot_enabled: bool = False
    onehot_weight: float = Field(default=0.15, ge=0)
    onehot_genre_vocab: int = Field(default=40, gt=0)
    vector_backend: VectorBackend = VectorBackend.NUMPY

    # --- Recommender tunables (hashed into config_hash) ---
    rec_n_default: int = Field(default=10, gt=0)
    candidates_similar: int = Field(default=500, gt=0)
    candidates_trending: int = Field(default=100, ge=0)
    w_sim: float = Field(default=0.70, ge=0)
    # D-017 (§13 Q6): quality / popularity come from TMDB, which D-011 keeps out of the
    # ML layer, so they are off until TMDB grants permission (spec defaults: 0.15 / 0.10).
    w_qual: float = Field(default=0.0, ge=0)
    w_pop: float = Field(default=0.0, ge=0)
    w_rec: float = Field(default=0.05, ge=0)
    # Penalty on cosine similarity to the negative profile (§8.2); 0 = off (Phase 1 default).
    w_neg: float = Field(default=0.0, ge=0)
    bayes_m: int = Field(default=200, ge=0)
    recency_half_life_days: int = Field(default=365, gt=0)
    profile_half_life_days: int = Field(default=90, gt=0)
    mmr_lambda: float = Field(default=0.7, ge=0, le=1)
    cap_per_language: int = Field(default=4, gt=0)
    cap_per_genre: int = Field(default=3, gt=0)
    # §8.4: `max_runtime` filters movies only unless this is set (TV runtime = episode length).
    max_runtime_applies_to_tv: bool = False

    # --- Availability ---
    availability_ttl_days: int = Field(default=7, gt=0)
    availability_buffer: int = Field(default=3, ge=0)
    availability_max_fetches_per_request: int = Field(default=60, ge=0)
    adult_certifications: dict[str, list[str]] = Field(default_factory=lambda: {"US": ["NC-17"]})

    @field_validator("cors_origins", "seed_languages", "seed_countries", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("jwt_secret_keys", mode="before")
    @classmethod
    def _parse_keys(cls, value: Any) -> Any:
        value = _split_csv(value)
        if isinstance(value, list) and not value:
            raise ValueError("JWT_SECRET_KEYS must contain at least one key")
        return value

    @field_validator("seed_countries", "default_country", mode="after")
    @classmethod
    def _upper_country(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [v.upper() for v in value]
        return value.upper() if isinstance(value, str) else value

    @field_validator("adult_certifications", mode="before")
    @classmethod
    def _parse_json(cls, value: Any) -> Any:
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("wikimedia_contact", mode="after")
    @classmethod
    def _blank_contact_is_none(cls, value: str | None) -> str | None:
        value = value.strip() if value else None
        if value and any(c in value for c in "\r\n()"):
            raise ValueError("WIKIMEDIA_CONTACT must be a single line without parentheses")
        return value or None

    @model_validator(mode="after")
    def _strong_jwt_keys_outside_local(self) -> Settings:
        if self.app_env is AppEnv.LOCAL:
            return self
        for key in self.jwt_secret_keys:
            raw = key.get_secret_value()
            if raw == PLACEHOLDER_JWT_KEY or len(raw.encode()) < MIN_JWT_KEY_BYTES:
                raise ValueError(
                    f"JWT_SECRET_KEYS: every key must be >= {MIN_JWT_KEY_BYTES} bytes "
                    f"and not the placeholder when APP_ENV={self.app_env.value}"
                )
        return self

    @field_validator("log_level", mode="after")
    @classmethod
    def _upper_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()
