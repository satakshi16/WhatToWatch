from __future__ import annotations

import json
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _json_lines(text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_generates_request_id_when_absent(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert re.fullmatch(r"[0-9a-f]{32}", resp.headers["X-Request-ID"])


def test_echoes_incoming_request_id(client: TestClient) -> None:
    resp = client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize("bad", [b"has space", b"x" * 200, "injecté".encode(), b"a\tb"])
def test_replaces_unsafe_incoming_request_id(client: TestClient, bad: bytes) -> None:
    resp = client.get("/api/v1/health", headers={"X-Request-ID": bad})  # type: ignore[dict-item]
    assert re.fullmatch(r"[0-9a-f]{32}", resp.headers["X-Request-ID"])


def test_request_id_on_error_responses(client: TestClient) -> None:
    assert "X-Request-ID" in client.get("/api/v1/nope").headers


def test_request_id_on_unhandled_500(app: FastAPI, client: TestClient) -> None:
    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("kaboom")

    resp = client.get("/boom", headers={"X-Request-ID": "req-500"})
    assert resp.status_code == 500
    assert resp.headers["X-Request-ID"] == "req-500"


def test_logs_are_json_lines_with_request_id(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    client.get("/api/v1/health", headers={"X-Request-ID": "log-me"})
    lines = _json_lines(capsys.readouterr().out)
    access = [ln for ln in lines if ln.get("event") == "request"]
    assert access, lines
    entry = access[-1]
    assert entry["request_id"] == "log-me"
    assert entry["status"] == 200
    assert entry["path"] == "/api/v1/health"
    assert {"timestamp", "level", "duration_ms"} <= entry.keys()
