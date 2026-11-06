"""Compatibility adapter: opaque llamaindex state bypasses cache access in v1.0.

Use explicit RequestContext with the manual interface for a proven workflow.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
import warnings

from longcache import LongCache, CacheConfig
from longcache.core.errors import LongCachePassThroughWarning


class CachedQueryEngine:
    """Compatibility adapter; opaque framework state bypasses caching in v1.0."""

    api_stability: Literal["pass-through"] = "pass-through"
    is_passthrough: bool = True
    cache_bypass_reason = "unsupported_request_shape"

    def __init__(
        self,
        query_engine: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "llamaindex",
        warn_passthrough: bool = True,
    ) -> None:
        """
        Initialize CachedQueryEngine.

        Args:
            query_engine: A LlamaIndex query engine instance.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
            warn_passthrough: Whether to emit LongCachePassThroughWarning on initialization.
        """
        if warn_passthrough:
            warnings.warn(
                f"{self.__class__.__name__} is a compatibility pass-through adapter in LongCache 1.0 "
                f"(reason={self.cache_bypass_reason!r}) and does not cache or reuse responses. "
                "Use LongCache.get_or_compute() or @cache.cached with an explicit RequestContext "
                "for verified semantic caching, or pass warn_passthrough=False to silence this warning.",
                LongCachePassThroughWarning,
                stacklevel=2,
            )
        self.query_engine = query_engine
        self.cache = cache
        self._namespace = namespace

    def query(self, query_str: str, **kwargs: Any) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = self.query_engine.query(query_str, **kwargs)
        return result

    async def aquery(self, query_str: str, **kwargs: Any) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = await self.query_engine.aquery(query_str, **kwargs)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying query engine."""
        return getattr(self.query_engine, name)

    def __repr__(self) -> str:
        return f"CachedQueryEngine(engine={self.query_engine!r}, cache={self.cache!r})"
