"""Pure helpers of the image secret check (P2-01, `make image-scan`)."""

from __future__ import annotations

from scripts import check_image_secrets as cis

HS = "hashed_secret"  # detect-secrets report field


def test_parse_env_handles_comments_and_quotes() -> None:
    env = cis.parse_env(
        '# header\nA=1  # trailing\nB="quoted # not a comment"\nC=http://x#frag\n\nNOEQUALS\n'
    )
    assert env == {"A": "1", "B": "quoted # not a comment", "C": "http://x#frag"}


def test_secret_values_picks_real_secrets_only() -> None:
    env = {
        "JWT_SECRET_KEYS": "aaaaaaaaaaaaaaaa,bbbbbbbbbbbbbbbb",  # pragma: allowlist secret
        "TMDB_READ_TOKEN": "eyJhbGciOiJIUzI1NiJ9.token",
        "TMDB_API_KEY": "<enter your api key>",
        "POSTGRES_PASSWORD": "whattowatch",  # pragma: allowlist secret
        "LANGSMITH_API_KEY": "",
        "SHORT_TOKEN": "abc",
        "DATABASE_URL": "postgresql://u:p@h/db",
        "CHANGE_SECRET": "change-me",  # pragma: allowlist secret
    }
    assert cis.secret_values(env) == {
        "aaaaaaaaaaaaaaaa": "JWT_SECRET_KEYS",
        "bbbbbbbbbbbbbbbb": "JWT_SECRET_KEYS",
        "eyJhbGciOiJIUzI1NiJ9.token": "TMDB_READ_TOKEN",
    }


def test_find_leaks_names_key_not_value() -> None:
    leaks = cis.find_leaks("ENV X=leakedvalue123", {"leakedvalue123": "X_TOKEN"}, "history")
    assert leaks == ["history: contains the value of X_TOKEN"]
    assert "leakedvalue123" not in leaks[0]
    assert cis.find_leaks("clean", {"leakedvalue123": "X_TOKEN"}, "history") == []


def test_is_env_file() -> None:
    assert cis.is_env_file(".env")
    assert cis.is_env_file("app/.env")
    assert cis.is_env_file("app/.env.local")
    assert not cis.is_env_file("app/.env.example")
    assert not cis.is_env_file("usr/lib/python3/site-packages/dotenv/main.py")
    assert not cis.is_env_file("app/environment.py")


def test_new_findings_ignores_baselined_entries() -> None:
    baseline = {"results": {"scripts/seed_demo.py": [{HS: "h1", "type": "Secret"}]}}
    scan = {
        "results": {
            "./scripts/seed_demo.py": [
                {HS: "h1", "type": "Secret Keyword"},
                {HS: "h2", "type": "Secret Keyword"},
            ],
            "app/core/config.py": [{HS: "h3", "type": "Hex High Entropy String"}],
        }
    }
    assert cis.new_findings(scan, baseline) == [
        "detect-secrets: Secret Keyword in /app/scripts/seed_demo.py",
        "detect-secrets: Hex High Entropy String in /app/app/core/config.py",
    ]
    assert cis.new_findings({"results": {}}, baseline) == []
