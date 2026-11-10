"""Embedding build end to end with a fake encoder (P1-05, §8.1, D-011)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import respx
from sqlalchemy import Engine, delete
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Title, TitleOpenContent, TitleType
from app.ingest import build_embeddings as be
from app.ingest.build_embeddings import (
    CONTENT_SOURCE,
    EmbeddingError,
    EmbedOptions,
    build_document,
    build_embeddings,
    embed_one,
    load_documents,
    load_index,
)
from app.ingest.wikimedia_client import WikimediaClient
from scripts import embedding_neighbours
from tests.catalog_factory import SENTINEL, add_catalog, open_content, tmdb_title
from tests.fakes import FakeEncoder
from tests.wikimedia_fakes import Entity, FakeWikimedia

M, T = TitleType.MOVIE, TitleType.TV
OPTIONS = EmbedOptions(batch_size=4, languages=("en", "ko"))


@pytest.fixture
def catalog(migrated_engine: Engine) -> Engine:
    """6 catalog titles: 4 with a Wikipedia summary, 1 Wikidata-only, 1 never checked."""
    with Session(migrated_engine) as session:
        add_catalog(
            session,
            tmdb_title(1, M),
            tmdb_title(2, M),
            tmdb_title(3, T, "ko"),
            tmdb_title(1, T),
            tmdb_title(5, M),
            tmdb_title(6, M),
        )
        session.add_all(
            [
                open_content(1, M, label="Open Orchard", genres=["drama film", "romance film"]),
                open_content(2, M, label="Open Heist", genres=["heist film"]),
                open_content(3, T, label="Open Seoul", language="ko", genres=["drama film"]),
                open_content(1, T, label="Open Show"),
                open_content(5, M, summary=None),
            ]
        )
        session.commit()
    return migrated_engine


def norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=1)


def test_provenance_no_tmdb_field_reaches_the_builder(catalog: Engine) -> None:
    """TMDB columns hold sentinel strings; none may appear in any document (D-011)."""
    with Session(catalog) as session:
        documents = [build_document(d) for d in load_documents(session)]
    texts = [t for t in documents if t is not None]
    assert len(texts) == 4
    for text in texts:
        assert SENTINEL not in text
        assert SENTINEL.lower() not in text.lower()
    assert any("Open Orchard" in t for t in texts)


def test_provenance_no_tmdb_field_reaches_the_index(catalog: Engine, tmp_path: Path) -> None:
    encoder = FakeEncoder()
    build_embeddings(catalog, encoder, OPTIONS, tmp_path)
    assert all(SENTINEL not in text for text in encoder.encoded_texts)
    for name in (be.ID_MAP_FILE, be.META_FILE):
        assert SENTINEL not in (tmp_path / name).read_text(encoding="utf-8")


def test_full_build_writes_index(catalog: Engine, tmp_path: Path) -> None:
    encoder = FakeEncoder(dim=8)
    stats = build_embeddings(catalog, encoder, OPTIONS, tmp_path)

    index = load_index(tmp_path)
    assert index is not None
    # Vectors == titles with open content (not the whole catalog).
    assert stats.total == len(index.ids) == index.vectors.shape[0] == 4
    assert index.vectors.dtype == np.float32
    assert np.allclose(norms(index.vectors), 1.0, atol=1e-5)
    assert sorted(index.ids) == ["movie:1", "movie:2", "tv:1", "tv:3"]

    meta = index.meta
    assert meta["model"] == "fake/e5"
    assert meta["model_version"] == "v1"
    assert meta["dim"] == meta["text_dim"] == 8
    assert meta["catalog_count"] == 6
    assert meta["count"] == 4
    assert meta["skipped"] == 2
    assert meta["skipped_breakdown"] == {"no_open_content_row": 1, "no_wikipedia_summary": 1}
    assert meta["content_source"] == CONTENT_SOURCE == "wikidata+wikipedia"
    assert meta["licenses"] == {"wikidata": "CC0 1.0", "wikipedia": "CC BY-SA 4.0"}
    assert meta["coverage"] == {"en/movie": 2, "en/tv": 1, "ko/tv": 1}
    assert meta["onehot"]["enabled"] is False
    assert meta["normalized"] is True
    assert len(meta["config_hash"]) == 16
    assert meta["last_run"] == {"mode": "full", "embedded": 4, "removed": 0}
    assert sum(len(c) for c in encoder.calls) == 4


def test_incremental_embeds_only_new_titles(catalog: Engine, tmp_path: Path) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    before = load_index(tmp_path)
    assert before is not None
    with Session(catalog) as session:
        session.add(open_content(6, M, label="Open Newcomer"))
        session.merge(open_content(5, M, label="Open Late Summary"))
        session.execute(delete(Title).where(Title.tmdb_id == 2, Title.type == M))
        session.commit()

    encoder = FakeEncoder()
    stats = build_embeddings(
        catalog, encoder, dataclasses.replace(OPTIONS, incremental=True), tmp_path
    )

    assert stats.mode == "incremental"
    assert (stats.embedded, stats.reused, stats.removed) == (2, 3, 1)
    assert len(encoder.encoded_texts) == 2
    assert all("Newcomer" in t or "Late Summary" in t for t in encoder.encoded_texts)
    after = load_index(tmp_path)
    assert after is not None
    assert "movie:2" not in after.ids
    assert {"movie:5", "movie:6"} <= set(after.ids)
    assert len(after.ids) == after.vectors.shape[0] == 5
    # Kept vectors are reused unchanged.
    for id_ in ("movie:1", "tv:3"):
        assert np.array_equal(
            after.vectors[after.ids.index(id_)], before.vectors[before.ids.index(id_)]
        )
    assert after.meta["created_at"] == before.meta["created_at"]
    assert after.meta["last_run"] == {"mode": "incremental", "embedded": 2, "removed": 1}

    # Nothing new → nothing encoded.
    encoder = FakeEncoder()
    stats = build_embeddings(
        catalog, encoder, dataclasses.replace(OPTIONS, incremental=True), tmp_path
    )
    assert stats.embedded == 0
    assert encoder.calls == []


def test_incremental_without_index_does_full_build(catalog: Engine, tmp_path: Path) -> None:
    stats = build_embeddings(
        catalog, FakeEncoder(), dataclasses.replace(OPTIONS, incremental=True), tmp_path
    )
    assert stats.mode == "full"
    assert stats.total == 4


@pytest.mark.parametrize(
    ("encoder", "options"),
    [
        (FakeEncoder(name="other/model"), OPTIONS),
        (FakeEncoder(version="v2"), OPTIONS),
        (FakeEncoder(), dataclasses.replace(OPTIONS, onehot_enabled=True)),
    ],
)
def test_incremental_refuses_incompatible_index(
    catalog: Engine, tmp_path: Path, encoder: FakeEncoder, options: EmbedOptions
) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    with pytest.raises(EmbeddingError, match="full build"):
        build_embeddings(catalog, encoder, dataclasses.replace(options, incremental=True), tmp_path)


def test_onehot_block_when_enabled(catalog: Engine, tmp_path: Path) -> None:
    options = dataclasses.replace(OPTIONS, onehot_enabled=True, onehot_genre_vocab=2)
    stats = build_embeddings(catalog, FakeEncoder(dim=8), options, tmp_path)
    index = load_index(tmp_path)
    assert index is not None
    spec = index.meta["onehot"]
    assert spec["genres"] == ["drama film", "heist film"]
    assert spec["languages"] == ["en", "ko"]
    expected_dim = 8 + (2 + 1) + 2 + (len(spec["decades"]) + 1)
    assert stats.dim == index.meta["dim"] == index.vectors.shape[1] == expected_dim
    assert np.allclose(norms(index.vectors), 1.0, atol=1e-5)


def test_same_content_gives_same_vectors(catalog: Engine, tmp_path: Path) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path / "a")
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path / "b")
    a, b = load_index(tmp_path / "a"), load_index(tmp_path / "b")
    assert a is not None
    assert b is not None
    assert a.ids == b.ids
    assert np.array_equal(a.vectors, b.vectors)


def test_load_index_detects_mismatch(catalog: Engine, tmp_path: Path) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    (tmp_path / be.ID_MAP_FILE).write_text(json.dumps(["movie:1"]))
    with pytest.raises(EmbeddingError, match="ids"):
        load_index(tmp_path)
    assert load_index(tmp_path / "missing") is None


# --- embed_one ------------------------------------------------------------------


async def test_embed_one_matches_index_row(catalog: Engine, tmp_path: Path) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    index = load_index(tmp_path)
    assert index is not None
    with Session(catalog) as session:
        vector = await embed_one((3, T), session=session, encoder=FakeEncoder(), meta=index.meta)
    assert vector is not None
    assert np.allclose(vector, index.vectors[index.ids.index("tv:3")], atol=1e-6)


async def test_embed_one_without_open_content_returns_none(catalog: Engine, tmp_path: Path) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    index = load_index(tmp_path)
    assert index is not None
    with Session(catalog) as session:
        assert (
            await embed_one((5, M), session=session, encoder=FakeEncoder(), meta=index.meta) is None
        )
        assert (
            await embed_one((6, M), session=session, encoder=FakeEncoder(), meta=index.meta) is None
        )


async def test_embed_one_rejects_other_model(catalog: Engine, tmp_path: Path) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    index = load_index(tmp_path)
    assert index is not None
    with Session(catalog) as session, pytest.raises(EmbeddingError):
        await embed_one((1, M), session=session, encoder=FakeEncoder(name="x"), meta=index.meta)


@pytest.fixture
def fake_wikimedia() -> Iterator[FakeWikimedia]:
    fake = FakeWikimedia()
    with respx.mock(assert_all_called=False) as router:
        fake.install(router)
        yield fake


async def _no_sleep(_: float) -> None:
    return None


async def test_embed_one_fetches_missing_open_content(
    catalog: Engine, tmp_path: Path, fake_wikimedia: FakeWikimedia
) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    index = load_index(tmp_path)
    assert index is not None
    fake_wikimedia.add(
        6,
        M,
        Entity(qid="Q66", label="Open Fetched", sitelinks={"en": "Open Fetched"}),
        summary="Fetched on demand.",
    )
    encoder = FakeEncoder()
    async with WikimediaClient(contact="ops@whattowatch.invalid", sleep=_no_sleep) as wm:
        with Session(catalog) as session:
            vector = await embed_one(
                (6, M), session=session, encoder=encoder, meta=index.meta, wikimedia=wm
            )
            stored = session.get(TitleOpenContent, (6, M))
    assert vector is not None
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    assert encoder.encoded_texts == ["passage: Open Fetched. Fetched on demand."]
    assert stored is not None
    assert stored.summary == "Fetched on demand."


async def test_embed_one_fetch_without_article_returns_none(
    catalog: Engine, tmp_path: Path, fake_wikimedia: FakeWikimedia
) -> None:
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    index = load_index(tmp_path)
    assert index is not None
    async with WikimediaClient(contact="ops@whattowatch.invalid", sleep=_no_sleep) as wm:
        with Session(catalog) as session:
            vector = await embed_one(
                (6, M), session=session, encoder=FakeEncoder(), meta=index.meta, wikimedia=wm
            )
            stored = session.get(TitleOpenContent, (6, M))
    assert vector is None
    assert stored is not None
    assert stored.wikidata_qid is None


# --- CLI + sanity script ---------------------------------------------------------


def test_cli_builds_with_injected_encoder(
    settings: Settings, catalog: Engine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "emb"
    assert be.main(["--out-dir", str(out_dir)], settings=settings, encoder=FakeEncoder()) == 0
    assert "Full build: 4 vectors" in capsys.readouterr().out
    assert (
        be.main(
            ["--out-dir", str(out_dir), "--incremental"], settings=settings, encoder=FakeEncoder()
        )
        == 0
    )
    assert "Incremental build: 4 vectors" in capsys.readouterr().out
    assert (
        be.main(
            ["--out-dir", str(out_dir), "--incremental"],
            settings=settings,
            encoder=FakeEncoder(name="other"),
        )
        == 1
    )


def test_cli_requires_schema(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    assert be.main([], settings=settings, encoder=FakeEncoder()) == 2
    assert "make migrate" in capsys.readouterr().err


def test_neighbours_script(
    settings: Settings,
    catalog: Engine,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configured = settings.model_copy(update={"embeddings_dir": tmp_path})
    build_embeddings(catalog, FakeEncoder(), OPTIONS, tmp_path)
    monkeypatch.setattr(embedding_neighbours, "get_settings", lambda: configured)

    code = embedding_neighbours.main(
        [
            "--title",
            "open orchard",
            "--title",
            "Open Se",
            "--title",
            "Nope",
            "--key",
            "tv:1",
            "-k",
            "2",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "Index: 4 vectors" in out
    assert "Open Orchard (2015, en, movie:1)" in out
    assert "Open Seoul (2015, ko, tv:3)" in out
    assert "'Nope': not in the index" in out
    assert out.count("  0.") + out.count("  -0.") + out.count("  1.") >= 6
