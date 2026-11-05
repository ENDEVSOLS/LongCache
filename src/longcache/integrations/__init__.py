"""Integrations with LLM frameworks."""

from __future__ import annotations

from typing import Any

from longcache.integrations.langchain import CachedChain, CachedLLM

__all__ = ["CachedLLM", "CachedChain", "get_llamaindex_wrapper", "get_openai_wrapper"]


# Lazy imports for optional integrations
def get_llamaindex_wrapper() -> type[Any]:
    """Import and return CachedQueryEngine. Raises ImportError if llama-index is not installed."""
    try:
        from longcache.integrations.llamaindex import CachedQueryEngine

        return CachedQueryEngine
    except ImportError:
        raise ImportError(
            "LlamaIndex integration requires the 'llamaindex' extra. "
            "Install with: uv pip install longcache[llamaindex]"
        )


def get_openai_wrapper() -> type[Any]:
    """Import and return CachedOpenAI. Raises ImportError if openai is not installed."""
    try:
        from longcache.integrations.openai import CachedOpenAI

        return CachedOpenAI
    except ImportError:
        raise ImportError(
            "OpenAI integration requires the 'openai' extra. "
            "Install with: uv pip install longcache[openai]"
        )
