"""Static checks on the Docker stack (P2-01, NFR-21): services, healthchecks, no secrets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVICES = {"api", "worker", "postgres", "redis", "ui"}


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return data


def test_compose_has_every_service_with_a_healthcheck(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert set(services) == SERVICES
    for name in SERVICES:
        assert "healthcheck" in services[name], name


def test_postgres_is_pgvector_pg16(compose: dict[str, Any]) -> None:
    assert compose["services"]["postgres"]["image"] == "pgvector/pgvector:pg16"


def test_api_and_worker_share_the_app_image(compose: dict[str, Any]) -> None:
    api, worker = compose["services"]["api"], compose["services"]["worker"]
    assert api["image"] == worker["image"]
    assert api["build"]["target"] == worker["build"]["target"] == "app"
    assert compose["services"]["ui"]["build"]["target"] == "ui"


def test_model_cache_volume_mounted_where_hf_looks(compose: dict[str, Any]) -> None:
    assert "model-cache" in compose["volumes"]
    for name in ("api", "worker"):
        svc = compose["services"][name]
        assert "model-cache:/models" in svc["volumes"]
        assert svc["environment"]["HF_HOME"] == "/models"


def test_api_runs_migrations_before_serving(compose: dict[str, Any]) -> None:
    cmd = " ".join(compose["services"]["api"]["command"])
    assert cmd.index("alembic upgrade head") < cmd.index("uvicorn")


def test_containers_use_postgres_and_internal_urls(compose: dict[str, Any]) -> None:
    api_env = compose["services"]["api"]["environment"]
    assert api_env["DATABASE_URL"].startswith("postgresql+psycopg://")
    assert "@postgres:5432/" in api_env["DATABASE_URL"]
    assert compose["services"]["ui"]["environment"]["API_BASE_URL"] == "http://api:8000"


def test_secrets_only_at_runtime(compose: dict[str, Any]) -> None:
    """Config arrives via env_file at runtime, never as build args."""
    for name, svc in compose["services"].items():
        build = svc.get("build") or {}
        assert "args" not in build, name
        assert "secrets" not in build, name
    assert compose["services"]["api"]["env_file"] == ".env"
    assert "env_file" not in compose["services"]["ui"]  # the UI needs no secrets


def test_published_ports_bind_to_localhost(compose: dict[str, Any]) -> None:
    for name, svc in compose["services"].items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), name


def test_dockerfile_has_no_secret_env_or_args() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert re.search(r"^FROM .* AS app$", dockerfile, re.MULTILINE)
    assert re.search(r"^FROM .* AS ui$", dockerfile, re.MULTILINE)
    for line in dockerfile.splitlines():
        if re.match(r"^\s*(ENV|ARG)\b", line):
            assert not re.search(r"SECRET|TOKEN|PASSWORD|API_KEY", line, re.IGNORECASE), line
    assert not re.search(r"^\s*COPY\s+(\.|\./)\s", dockerfile, re.MULTILINE)  # no `COPY . .`
    assert ".env" not in re.findall(r"^\s*COPY\s+(.*)$", dockerfile, re.MULTILINE)


def test_dockerignore_excludes_env_and_local_data() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {".env", ".env.*", "!.env.example", "data", ".git", ".venv*"} <= patterns


def test_compose_only_stack_vars_documented_in_env_example() -> None:
    example = (ROOT / ".env.example").read_text()
    compose_text = (ROOT / "docker-compose.yml").read_text()
    for var in set(re.findall(r"\$\{([A-Z_]+):-", compose_text)):
        assert re.search(rf"^{var}=", example, re.MULTILINE), var
