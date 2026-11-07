"""Compatibility adapter: opaque langgraph state bypasses cache access in v1.0.

Use explicit RequestContext with the manual interface for a proven workflow.
"""

from __future__ import annotations

from typing import Any, Literal
import warnings

from longcache import LongCache
from longcache.core.errors import LongCachePassThroughWarning


class CachedState:
    """Compatibility adapter; opaque framework state bypasses caching in v1.0."""

    api_stability: Literal["pass-through"] = "pass-through"
    is_passthrough: bool = True
    cache_bypass_reason = "unsupported_request_shape"

    @classmethod
    def _warn(cls) -> None:
        warnings.warn(
            f"{cls.__name__} is a compatibility pass-through adapter in LongCache 1.0 "
            f"(reason={cls.cache_bypass_reason!r}) and does not cache or reuse responses. "
            "Use LongCache.get_or_compute() or @cache.cached with an explicit RequestContext "
            "for verified semantic caching, or pass warn_passthrough=False to silence this warning.",
            LongCachePassThroughWarning,
            stacklevel=3,
        )

    @staticmethod
    def check(
        state: dict[str, Any],
        cache: LongCache,
        query_key: str = "messages",
        namespace: str = "langgraph",
        warn_passthrough: bool = True,
    ) -> dict[str, Any]:
        """Return a miss for opaque graph state, preserving legacy query extraction."""
        if warn_passthrough:
            CachedState._warn()
        query_input = state.get(query_key)

        # Extract string query
        query = ""
        if isinstance(query_input, str):
            query = query_input
        elif isinstance(query_input, list):
            for msg in reversed(query_input):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
                    query = str(msg.content)
                    break
                elif hasattr(msg, "content"):
                    query = str(msg.content)
            if not query and query_input:
                query = str(query_input[-1])
        elif isinstance(query_input, dict):
            query = str(query_input.get("query", str(query_input)))
        else:
            query = str(query_input)

        if not query:
            return {"__cache_hit": False}

        return {"__cache_hit": False, "cached_query": query}

    @staticmethod
    async def acheck(
        state: dict[str, Any],
        cache: LongCache,
        query_key: str = "messages",
        namespace: str = "langgraph",
        warn_passthrough: bool = True,
    ) -> dict[str, Any]:
        """Return a miss for opaque graph state, preserving legacy query extraction."""
        if warn_passthrough:
            CachedState._warn()
        query_input = state.get(query_key)

        # Extract string query
        query = ""
        if isinstance(query_input, str):
            query = query_input
        elif isinstance(query_input, list):
            for msg in reversed(query_input):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
                    query = str(msg.content)
                    break
                elif hasattr(msg, "content"):
                    query = str(msg.content)
            if not query and query_input:
                query = str(query_input[-1])
        elif isinstance(query_input, dict):
            query = str(query_input.get("query", str(query_input)))
        else:
            query = str(query_input)

        if not query:
            return {"__cache_hit": False}

        return {"__cache_hit": False, "cached_query": query}

    @staticmethod
    def save(
        state: dict[str, Any],
        response: Any,
        cache: LongCache,
        namespace: str = "langgraph",
        warn_passthrough: bool = True,
    ) -> None:
        """No-op: graph state has no supported identity/replay contract in v1.0."""
        if warn_passthrough:
            CachedState._warn()
        return None

    @staticmethod
    async def asave(
        state: dict[str, Any],
        response: Any,
        cache: LongCache,
        namespace: str = "langgraph",
        warn_passthrough: bool = True,
    ) -> None:
        """No-op: graph state has no supported identity/replay contract in v1.0."""
        if warn_passthrough:
            CachedState._warn()
        return None
