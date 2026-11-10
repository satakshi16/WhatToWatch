"""Document builder, one-hot block and normalization (P1-05, §8.1, D-011)."""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pytest

from app.db.models import TitleOpenContent, TitleType
from app.ingest.build_embeddings import (
    EmbedOptions,
    OneHotSpec,
    OpenDocument,
    build_document,
    config_hash,
    finalize,
    genre_vocabulary,
    key_str,
    l2_normalize,
    onehot_block,
    parse_key,
)


def doc(**overrides: object) -> OpenDocument:
    values: dict[str, object] = {
        "key": (1, TitleType.MOVIE),
        "label": "The Glass Orchard",
        "genres": ("drama film", "romance film"),
        "directors": ("Ana Director",),
        "top_cast": ("Lead Actor", "Second Lead"),
        "summary": "Two sisters restore a greenhouse.",
        "original_language": "en",
        "publication_date": date(2019, 5, 1),
    }
    values.update(overrides)
    return OpenDocument(**values)  # type: ignore[arg-type]


def test_document_follows_template() -> None:
    assert build_document(doc()) == (
        "passage: The Glass Orchard. drama film, romance film. Directed by Ana Director. "
        "Starring Lead Actor, Second Lead. Two sisters restore a greenhouse."
    )


def test_document_omits_empty_parts_and_collapses_whitespace() -> None:
    text = build_document(doc(genres=(), directors=(" ",), top_cast=(), label="  A  Title. "))
    assert text == "passage: A Title. Two sisters restore a greenhouse."
    assert build_document(doc(label=None, genres=(), directors=(), top_cast=())) == (
        "passage: Two sisters restore a greenhouse."
    )


@pytest.mark.parametrize("summary", [None, "", "   "])
def test_no_summary_means_no_document(summary: str | None) -> None:
    assert build_document(doc(summary=summary)) is None
    assert not doc(summary=summary).embeddable


def test_open_document_has_no_tmdb_fields() -> None:
    """The builder's only input type must not carry TMDB-sourced columns (D-011)."""
    fields = {f.name for f in dataclasses.fields(OpenDocument)}
    tmdb_only = {
        "title",
        "original_title",
        "overview",
        "keywords",
        "localized_titles",
        "certification",
    }
    assert not fields & tmdb_only
    assert fields == {
        "key",
        "label",
        "genres",
        "directors",
        "top_cast",
        "summary",
        "original_language",
        "publication_date",
    }


def test_from_row_copies_open_content() -> None:
    row = TitleOpenContent(
        tmdb_id=3,
        type=TitleType.TV,
        label="L",
        genres=["g"],
        directors=[],
        top_cast=["a"],
        summary="S",
        original_language="ko",
        publication_date=date(2001, 1, 1),
    )
    d = OpenDocument.from_row(row)
    assert d.key == (3, TitleType.TV)
    assert d.genres == ("g",)
    assert d.top_cast == ("a",)
    assert build_document(d) == "passage: L. g. Starring a. S"


def test_keys_round_trip() -> None:
    assert key_str((27205, TitleType.MOVIE)) == "movie:27205"
    assert parse_key("tv:1429") == (1429, TitleType.TV)


def test_genre_vocabulary_by_frequency_then_name() -> None:
    docs = [doc(genres=("b", "a")), doc(genres=("a", "a")), doc(genres=("c", "b"))]
    assert genre_vocabulary(docs, 2) == ("a", "b")
    assert genre_vocabulary(docs, 10) == ("a", "b", "c")


def test_onehot_disabled_is_empty() -> None:
    assert onehot_block(doc(), OneHotSpec()).shape == (0,)
    assert OneHotSpec().dim == 0


def test_onehot_blocks_have_weight_norm() -> None:
    spec = OneHotSpec(enabled=True, weight=0.15, languages=("en", "ko"), genres=("drama film", "x"))
    block = onehot_block(doc(), spec)
    assert block.shape == (spec.dim,)
    lang, genres, decades = np.split(block, [3, 5])
    for part in (lang, genres, decades):
        assert np.linalg.norm(part) == pytest.approx(0.15)
    assert lang[0] > 0  # en
    assert genres[0] > 0
    assert genres[1] == 0
    assert decades[spec.decades.index(2010)] > 0


def test_onehot_unknowns_use_other_slots() -> None:
    spec = OneHotSpec(enabled=True, languages=("en",), genres=("x",))
    block = onehot_block(doc(original_language=None, genres=("zzz",), publication_date=None), spec)
    lang, genres, decades = np.split(block, [2, 3])
    assert lang[-1] > 0
    assert not genres.any()
    assert decades[-1] > 0


def test_onehot_spec_meta_round_trip() -> None:
    spec = OneHotSpec(enabled=True, weight=0.2, languages=("en",), genres=("a",))
    assert OneHotSpec.from_meta(spec.to_meta()) == spec


def test_finalize_normalizes_with_and_without_onehot() -> None:
    raw = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    docs = [doc(), doc()]
    out = finalize(raw, docs, OneHotSpec())
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])  # zero vector stays zero (no NaN)
    spec = OneHotSpec(enabled=True, languages=("en",), genres=("drama film",))
    out = finalize(raw[:1], docs[:1], spec)
    assert out.shape == (1, 2 + spec.dim)
    assert np.linalg.norm(out[0]) == pytest.approx(1.0)
    assert out.dtype == np.float32


def test_l2_normalize_rows() -> None:
    out = l2_normalize(np.array([[1.0, 1.0], [2.0, 0.0]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_config_hash_tracks_vector_relevant_settings() -> None:
    base = EmbedOptions(languages=("en",))
    assert config_hash("m", base) == config_hash("m", dataclasses.replace(base, batch_size=1))
    assert config_hash("m", base) != config_hash("other", base)
    assert config_hash("m", base) != config_hash(
        "m", dataclasses.replace(base, onehot_enabled=True)
    )
    # Weight only matters when the block is on.
    assert config_hash("m", base) == config_hash("m", dataclasses.replace(base, onehot_weight=0.9))
