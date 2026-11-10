"""Small in-process TTL cache (single event loop / thread; not shared across workers)."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable


class TTLCache[K: Hashable, V]:
    """Mapping with per-entry expiry and a size cap (oldest entry evicted first).

    `ttl_seconds=0` disables caching: `set` stores nothing.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._ttl = ttl_seconds
        self._max = max_entries
        self._clock = clock
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()

    def get(self, key: K) -> V | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._data[key]
            return None
        return value

    def set(self, key: K, value: V) -> None:
        if self._ttl <= 0:
            return
        self._data.pop(key, None)
        self._data[key] = (self._clock() + self._ttl, value)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
