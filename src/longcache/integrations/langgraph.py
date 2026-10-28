"""Compatibility adapter: opaque langgraph state bypasses cache access in v0.2.

Use explicit RequestContext with the manual interface for a proven workflow.
"""

from __future__ import annotations

from typing import Any

from longcache import LongCache


class CachedState:
    """Compatibility adapter; opaque framework state bypasses caching in v0.2."""

    cache_bypass_reason = "unsupported_request_shape"

    @staticmethod
    def check(
        state: dict,
        cache: LongCache,
        query_key: str = "messages",
        namespace: str = "langgraph",
    ) -> dict:
        """Return a miss for opaque graph state, preserving legacy query extraction."""
        query_input = state.get(query_key)

        # Extract string query
        query = ""
        if isinstance(query_input, str):
            query = query_input
        elif isinstance(query_input, list):
            for msg in reversed(query_input):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
                    query = msg.content
                    break
                elif hasattr(msg, "content"):
                    query = msg.content
            if not query and query_input:
                query = str(query_input[-1])
        elif isinstance(query_input, dict):
            query = query_input.get("query", str(query_input))
        else:
            query = str(query_input)

        if not query:
            return {"__cache_hit": False}

        return {"__cache_hit": False, "cached_query": query}

    @staticmethod
    async def acheck(
        state: dict,
        cache: LongCache,
        query_key: str = "messages",
        namespace: str = "langgraph",
    ) -> dict:
        """Return a miss for opaque graph state, preserving legacy query extraction."""
        query_input = state.get(query_key)

        # Extract string query
        query = ""
        if isinstance(query_input, str):
            query = query_input
        elif isinstance(query_input, list):
            for msg in reversed(query_input):
                if hasattr(msg, "content") and getattr(msg, "type", "") == "human":
                    query = msg.content
                    break
                elif hasattr(msg, "content"):
                    query = msg.content
            if not query and query_input:
                query = str(query_input[-1])
        elif isinstance(query_input, dict):
            query = query_input.get("query", str(query_input))
        else:
            query = str(query_input)

        if not query:
            return {"__cache_hit": False}

        return {"__cache_hit": False, "cached_query": query}

    @staticmethod
    def save(
        state: dict,
        response: Any,
        cache: LongCache,
        namespace: str = "langgraph",
    ) -> None:
        """No-op: graph state has no supported identity/replay contract in v0.2."""
        # Opaque graph state has no complete identity/replay contract in v0.2.
        return None

    @staticmethod
    async def asave(
        state: dict,
        response: Any,
        cache: LongCache,
        namespace: str = "langgraph",
    ) -> None:
        """No-op: graph state has no supported identity/replay contract in v0.2."""
        return None
