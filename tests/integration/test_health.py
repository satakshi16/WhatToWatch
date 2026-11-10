from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingest.build_embeddings import EmbeddingIndex, save_index
from app.main import create_app


def test_health_returns_200_json(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"status": "ok"}


def _write_index(directory: Path) -> None:
    vectors = np.eye(2, 4, dtype=np.float32)
    save_index(directory, EmbeddingIndex(ids=["movie:1", "tv:1"], vectors=vectors, meta={}))


def test_ready_ok_when_database_reachable_and_embeddings_loaded(settings: Settings) -> None:
    _write_index(settings.embeddings_dir)
    app = create_app(settings)
    # Not loaded before startup...
    assert app.state.embedder.store.loaded is False
    with TestClient(app, raise_server_exceptions=False) as client:
        # ...loaded by the startup hook, without any request touching the index.
        assert app.state.embedder.store.loaded is True
        resp = client.get("/api/v1/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "checks": {"database": "ok", "embeddings": "ok"}}


def test_ready_503_when_embedding_index_missing(client: TestClient) -> None:
    resp = client.get("/api/v1/ready")
    assert resp.status_code == 503
    assert resp.headers["content-type"] == "application/problem+json"
    assert resp.json()["checks"] == {"database": "ok", "embeddings": "unavailable"}


def test_ready_503_problem_json_when_database_unreachable(tmp_path: Path) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        database_url=f"sqlite:///{blocker / 'sub' / 'db.sqlite'}",
        embeddings_dir=tmp_path / "embeddings",
    )
    _write_index(settings.embeddings_dir)
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        resp = client.get("/api/v1/ready")
    assert resp.status_code == 503
    assert resp.headers["content-type"] == "application/problem+json"
    body = resp.json()
    assert body["status"] == 503
    assert body["checks"] == {"database": "unavailable", "embeddings": "ok"}
