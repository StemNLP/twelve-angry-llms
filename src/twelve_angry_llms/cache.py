"""Disk-backed response cache.

The rigor requirements of panel studies (and the cost of re-running them)
make caching every raw model response non-negotiable. ``CachedClient``
wraps any ``LLMClient`` and stores completions in a local SQLite file,
keyed on the SHA-256 of (model, temperature, max_tokens, messages) - so a
re-run with identical inputs never re-bills, while any change to the
prompt, protocol, or sampling settings naturally misses.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Sequence

from .clients.base import LLMClient, Message


class ResponseCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def key(
        model: str,
        messages: Sequence[Message],
        temperature: float,
        max_tokens: int,
        salt: str = "",
    ) -> str:
        """``salt`` carries anything else that changes what serves the
        request (endpoint, provider routing pins); see ``cache_salt`` on
        clients."""
        payload = json.dumps(
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": list(messages),
                "salt": salt,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class CachedClient:
    """Wrap a client so identical requests are served from disk."""

    def __init__(self, client: LLMClient, cache: ResponseCache | str | Path) -> None:
        self._client = client
        self.cache = cache if isinstance(cache, ResponseCache) else ResponseCache(cache)
        self.hits = 0
        self.misses = 0

    @property
    def usage(self):
        return getattr(self._client, "usage", None)

    async def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        salt = getattr(self._client, "cache_salt", "") or ""
        key = ResponseCache.key(model, messages, temperature, max_tokens, salt=salt)
        cached = self.cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = await self._client.complete(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
        )
        self.cache.put(key, result)
        return result
