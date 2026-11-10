"""P1-09: in-memory `EmbeddingStore` (load once, append without re-embedding)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.db.models import TitleType
from app.ingest.build_embeddings import EmbeddingError, EmbeddingIndex, save_index
from app.recsys.embeddings import EmbeddingStore

M, T = TitleType.MOVIE, TitleType.TV


def _write_index(directory: Path, ids: list[str], dim: int = 4) -> np.ndarray:
    vectors = np.eye(len(ids), dim, dtype=np.float32)
    save_index(directory, EmbeddingIndex(ids=list(ids), vectors=vectors, meta={"model": "fake"}))
    return vectors


def test_missing_index_is_not_loaded(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path)
    assert store.load() is False
    assert store.meta is None
    assert store.dim is None
    assert store.matrix() is None
    assert store.ids() == []
    assert len(store) == 0
    assert (1, M) not in store
    assert store.vector((1, M)) is None
    with pytest.raises(EmbeddingError):
        store.append((1, M), np.zeros(4, dtype=np.float32))


def test_loads_index_once(tmp_path: Path) -> None:
    vectors = _write_index(tmp_path, ["movie:1", "tv:1"])
    store = EmbeddingStore(tmp_path)
    assert store.load() is True
    assert len(store) == 2
    assert store.dim == 4
    assert store.meta == {"model": "fake"}
    assert (1, M) in store
    assert (1, T) in store
    assert (2, M) not in store
    assert "movie:1" not in store  # keys are (tmdb_id, type) tuples
    assert np.array_equal(store.vector((1, T)), vectors[1])

    # Later changes on disk are not picked up until reload().
    _write_index(tmp_path, ["movie:1", "tv:1", "movie:2"])
    assert (2, M) not in store
    assert store.reload() is True
    assert (2, M) in store


def test_unreadable_index_counts_as_missing(tmp_path: Path) -> None:
    _write_index(tmp_path, ["movie:1"])
    (tmp_path / "id_map.json").write_text('["movie:1", "movie:2"]')  # length mismatch
    assert EmbeddingStore(tmp_path).load() is False


def test_append_makes_vector_available_immediately(tmp_path: Path) -> None:
    vectors = _write_index(tmp_path, ["movie:1", "tv:1"])
    store = EmbeddingStore(tmp_path)
    new = np.array([0.0, 0.0, 0.6, 0.8], dtype=np.float32)

    assert store.append((7, M), new) is True

    assert (7, M) in store
    assert store.appended == 1
    assert np.array_equal(store.vector((7, M)), new)
    assert store.ids() == ["movie:1", "tv:1", "movie:7"]
    matrix = store.matrix()
    assert matrix is not None
    assert matrix.shape == (3, 4)
    assert np.array_equal(matrix[:2], vectors)
    assert np.array_equal(matrix[2], new)
    # Still readable after the buffer was folded into the matrix.
    assert np.array_equal(store.vector((7, M)), new)
    # The files on disk are not changed.
    assert EmbeddingStore(tmp_path).ids() == ["movie:1", "tv:1"]


def test_append_existing_key_is_a_noop(tmp_path: Path) -> None:
    vectors = _write_index(tmp_path, ["movie:1"])
    store = EmbeddingStore(tmp_path)
    assert store.append((1, M), np.ones(4, dtype=np.float32)) is False
    assert store.appended == 0
    assert np.array_equal(store.vector((1, M)), vectors[0])


def test_append_rejects_wrong_dimension(tmp_path: Path) -> None:
    _write_index(tmp_path, ["movie:1"])
    store = EmbeddingStore(tmp_path)
    with pytest.raises(EmbeddingError, match="dim"):
        store.append((2, M), np.ones(3, dtype=np.float32))
    assert (2, M) not in store


def test_reload_drops_appended_vectors(tmp_path: Path) -> None:
    _write_index(tmp_path, ["movie:1"])
    store = EmbeddingStore(tmp_path)
    store.append((2, M), np.ones(4, dtype=np.float32))
    store.reload()
    assert (2, M) not in store
    assert store.appended == 0


def test_loaded_flag_does_not_trigger_load(tmp_path: Path) -> None:
    _write_index(tmp_path, ["movie:1"])
    store = EmbeddingStore(tmp_path)
    assert store.loaded is False
    assert store.load() is True
    assert store.loaded is True
    assert EmbeddingStore(tmp_path / "missing").snapshot() is None


def test_snapshot_includes_appended_rows_and_stays_aligned(tmp_path: Path) -> None:
    _write_index(tmp_path, ["movie:1", "tv:1"])
    store = EmbeddingStore(tmp_path)
    store.append((7, M), np.full(4, 0.5, dtype=np.float32))
    shot = store.snapshot()
    assert shot is not None
    ids, matrix = shot
    assert ids == ["movie:1", "tv:1", "movie:7"]
    assert matrix.shape == (3, 4)
    np.testing.assert_allclose(matrix[2], 0.5)
    # A later append leaves the earlier snapshot untouched.
    store.append((8, M), np.ones(4, dtype=np.float32))
    assert matrix.shape == (3, 4)
    assert ids == ["movie:1", "tv:1", "movie:7"]
    new = store.snapshot()
    assert new is not None
    assert new[1].shape == (4, 4)
