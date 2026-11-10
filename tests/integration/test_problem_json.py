from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.testclient import TestClient

PROBLEM = "application/problem+json"


def _assert_problem(body: dict[str, object], status: int, instance: str) -> None:
    assert body["type"] == "about:blank"
    assert body["status"] == status
    assert body["instance"] == instance
    assert isinstance(body["title"], str)
    assert body["title"]


def test_404_is_problem_json(client: TestClient) -> None:
    resp = client.get("/api/v1/does-not-exist")
    assert resp.status_code == 404
    assert resp.headers["content-type"] == PROBLEM
    _assert_problem(resp.json(), 404, "/api/v1/does-not-exist")
    assert resp.json()["title"] == "Not Found"


def test_405_is_problem_json(client: TestClient) -> None:
    resp = client.post("/api/v1/health")
    assert resp.status_code == 405
    assert resp.headers["content-type"] == PROBLEM
    assert "allow" in {k.lower() for k in resp.headers}


def test_422_is_problem_json_with_errors(app: FastAPI, client: TestClient) -> None:
    @app.get("/items")
    def items(n: int = Query(ge=1, le=20)) -> dict[str, int]:
        return {"n": n}

    resp = client.get("/items", params={"n": "secret-value"})
    assert resp.status_code == 422
    assert resp.headers["content-type"] == PROBLEM
    body = resp.json()
    _assert_problem(body, 422, "/items")
    assert body["errors"][0]["loc"] == ["query", "n"]
    assert {"loc", "msg", "type"} == set(body["errors"][0])
    assert "secret-value" not in resp.text  # raw input is never echoed


def test_http_exception_detail_preserved(app: FastAPI, client: TestClient) -> None:
    @app.get("/conflict")
    def conflict() -> None:
        raise HTTPException(status_code=409, detail="Email already registered.")

    resp = client.get("/conflict")
    assert resp.headers["content-type"] == PROBLEM
    assert resp.json()["detail"] == "Email already registered."
    assert resp.json()["title"] == "Conflict"


def test_unhandled_exception_is_problem_json_without_leaking(
    app: FastAPI, client: TestClient
) -> None:
    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("database password is hunter2")

    resp = client.get("/boom")
    assert resp.status_code == 500
    assert resp.headers["content-type"] == PROBLEM
    _assert_problem(resp.json(), 500, "/boom")
    assert "hunter2" not in resp.text
