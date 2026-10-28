"""Compatibility adapter: opaque llamaindex state bypasses cache access in v0.2.

Use explicit RequestContext with the manual interface for a proven workflow.
"""

from __future__ import annotations

from typing import Any, Optional

from longcache import LongCache, CacheConfig


class CachedQueryEngine:
    """Compatibility adapter; opaque framework state bypasses caching in v0.2."""

    cache_bypass_reason = "unsupported_request_shape"

    def __init__(
        self,
        query_engine: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "llamaindex",
    ):
        """
        Initialize CachedQueryEngine.

        Args:
            query_engine: A LlamaIndex query engine instance.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
        """
        self.query_engine = query_engine
        self.cache = cache
        self._namespace = namespace

    def query(self, query_str: str, **kwargs) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = self.query_engine.query(query_str, **kwargs)
        return result

    async def aquery(self, query_str: str, **kwargs) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = await self.query_engine.aquery(query_str, **kwargs)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying query engine."""
        return getattr(self.query_engine, name)

    def __repr__(self) -> str:
        return f"CachedQueryEngine(engine={self.query_engine!r}, cache={self.cache!r})"
