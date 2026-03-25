from __future__ import annotations

import time
from typing import Any


class SimpleTTLCache:
    """Простой in-memory TTL кэш для горячих endpoint'ов."""

    def __init__(self, ttl_seconds: int = 30):
        self._cache: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any:
        if key in self._cache:
            if time.time() - self._timestamps[key] < self._ttl:
                return self._cache[key]
            self.invalidate(key)
        return None

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def invalidate(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]

    def invalidate_prefix(self, prefix: str) -> None:
        keys = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys:
            self.invalidate(key)
