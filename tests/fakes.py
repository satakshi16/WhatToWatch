"""Deterministic fake encoder for embedding tests (no model download)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from app.ingest.build_embeddings import Vectors


class FakeEncoder:
    """Hash-seeded Gaussian vectors: same text → same vector; records every call."""

    def __init__(self, dim: int = 8, name: str = "fake/e5", version: str = "v1") -> None:
        self._dim = dim
        self._name = name
        self._version = version
        self.calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str], batch_size: int) -> Vectors:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).normal(size=self._dim) * 3.0)
        return np.asarray(rows, dtype=np.float32).reshape(len(texts), self._dim)

    @property
    def encoded_texts(self) -> list[str]:
        return [t for call in self.calls for t in call]
