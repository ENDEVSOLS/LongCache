"""Compatibility adapter: opaque langchain state bypasses cache access in v0.2.

Use explicit RequestContext with the manual interface for a proven workflow.
"""

from __future__ import annotations

from typing import Any, Optional

from longcache import LongCache, CacheConfig


class CachedLLM:
    """Compatibility adapter; opaque framework state bypasses caching in v0.2."""

    cache_bypass_reason = "unsupported_request_shape"

    def __init__(
        self,
        llm: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        namespace: str = "langchain",
    ):
        """
        Initialize CachedLLM.

        Args:
            llm: Any LangChain BaseLLM or BaseChatModel instance.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            namespace: Default namespace for cache entries.
        """
        self.llm = llm
        self.cache = cache
        self._namespace = namespace

    def _extract_query(self, input: Any) -> str:
        """
        Extract a query string from various LangChain input formats.

        Supports:
            - str: used directly
            - list of BaseMessage: extracts last HumanMessage content
            - dict with 'query', 'question', 'input', or 'messages' key

        Args:
            input: LangChain input in any supported format.

        Returns:
            Extracted query string.
        """
        if isinstance(input, str):
            return input

        if isinstance(input, list):
            # Try to extract from message list
            for msg in reversed(input):
                if hasattr(msg, "content") and hasattr(msg, "type"):
                    if msg.type == "human":
                        return msg.content
                # Fallback: use last message with content
                if hasattr(msg, "content"):
                    return msg.content
            # Last resort: stringify
            return str(input[-1]) if input else ""

        if isinstance(input, dict):
            for key in ("query", "question", "input", "messages", "prompt"):
                if key in input:
                    value = input[key]
                    if isinstance(value, str):
                        return value
                    if isinstance(value, list):
                        return self._extract_query(value)
                    return str(value)

        return str(input)

    def invoke(self, input: Any, *args, **kwargs) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = self.llm.invoke(input, *args, **kwargs)
        return result

    async def ainvoke(self, input: Any, *args, **kwargs) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = await self.llm.ainvoke(input, *args, **kwargs)
        return result

    def stream(self, input: Any, *args, **kwargs):
        """Delegate unchanged; unproven execution identity is never cached."""
        yield from self.llm.stream(input, *args, **kwargs)

    async def astream(self, input: Any, *args, **kwargs):
        """Delegate unchanged; unproven execution identity is never cached."""
        async for chunk in self.llm.astream(input, *args, **kwargs):
            yield chunk

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying LLM."""
        return getattr(self.llm, name)

    def __repr__(self) -> str:
        return f"CachedLLM(llm={self.llm!r}, cache={self.cache!r})"


class CachedChain:
    """Compatibility adapter; opaque framework state bypasses caching in v0.2."""

    cache_bypass_reason = "unsupported_request_shape"

    def __init__(
        self,
        chain: Any,
        cache: Optional[LongCache] = None,
        config: Optional[CacheConfig] = None,
        query_key: str = "question",
        namespace: str = "langchain_chain",
    ):
        """
        Initialize CachedChain.

        Args:
            chain: Any LCEL Runnable chain.
            cache: Existing LongCache instance. Creates a new one if not provided.
            config: CacheConfig for the new cache (ignored if cache is provided).
            query_key: Key to extract from input dict as the cache query.
            namespace: Default namespace for cache entries.
        """
        self.chain = chain
        self.cache = cache
        self.query_key = query_key
        self._namespace = namespace

    def _extract_query(self, input: Any) -> str:
        """
        Extract the query string from chain input.

        Args:
            input: Chain input (dict or string).

        Returns:
            The extracted query string.
        """
        if isinstance(input, dict):
            return input.get(self.query_key, str(input))
        return str(input)

    def invoke(self, input: Any, *args, **kwargs) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = self.chain.invoke(input, *args, **kwargs)
        return result

    async def ainvoke(self, input: Any, *args, **kwargs) -> Any:
        """Delegate unchanged; unproven execution identity is never cached."""

        result = await self.chain.ainvoke(input, *args, **kwargs)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attributes to the underlying chain."""
        return getattr(self.chain, name)

    def __repr__(self) -> str:
        return f"CachedChain(chain={self.chain!r}, cache={self.cache!r})"
