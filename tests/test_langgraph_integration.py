"""Tests for the LangGraph integration (CachedState)."""

from unittest.mock import MagicMock, patch

import numpy as np

from longcache.config import CacheConfig


def _make_mock_embedder():
    embedder = MagicMock()
    embedder._model = MagicMock()
    embedder._dimension = 384

    def deterministic_embed(text: str):
        np.random.seed(hash(text) % (2**31))
        vector = np.random.randn(384).astype(np.float32)
        vector /= np.linalg.norm(vector)
        return vector

    embedder.embed = deterministic_embed
    embedder.dimension = 384
    return embedder


def _make_message(content: str, msg_type: str = "human"):
    msg = MagicMock()
    msg.content = content
    msg.type = msg_type
    return msg


@patch("longcache.core.cache.LocalEmbedder")
class TestCachedStateCheckSave:
    def _cache(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache import LongCache

        cache = LongCache(CacheConfig(backend="memory", similarity_threshold=0.9))
        cache.embedder = mock_embedder
        return cache

    def test_check_miss_then_save_then_hit(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        state = {"messages": "What is the refund policy?"}

        result = CachedState.check(state, cache, query_key="messages")
        assert result["__cache_hit"] is False
        assert result["cached_query"] == "What is the refund policy?"

        CachedState.save(result, "Refunds within 30 days.", cache)

        # Now a subsequent check with the same query should be a HIT
        result2 = CachedState.check(state, cache, query_key="messages")
        assert result2["__cache_hit"] is False
        assert "cached_response" not in result2

    def test_check_extracts_from_string(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        state = {"messages": "plain string query"}
        result = CachedState.check(state, cache, query_key="messages")
        assert result["cached_query"] == "plain string query"

    def test_check_extracts_last_human_message(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        messages = [
            _make_message("system prompt", "system"),
            _make_message("first human turn", "human"),
            _make_message("assistant reply", "ai"),
            _make_message("second human turn", "human"),
        ]
        state = {"messages": messages}
        result = CachedState.check(state, cache, query_key="messages")
        assert result["cached_query"] == "second human turn"

    def test_check_extracts_from_dict(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        state = {"messages": {"query": "dict-based query"}}
        result = CachedState.check(state, cache, query_key="messages")
        assert result["cached_query"] == "dict-based query"

    def test_check_empty_query_returns_no_hit(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        state = {"messages": ""}
        result = CachedState.check(state, cache, query_key="messages")
        assert result == {"__cache_hit": False}

    def test_save_without_cached_query_is_noop(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        # No 'cached_query' key present — should not raise, should not store anything
        CachedState.save({}, "some response", cache)
        assert cache.stats()["total_entries"] == 0

    async def test_acheck_and_asave_roundtrip(self, MockEmbedder):
        from longcache.integrations.langgraph import CachedState

        cache = self._cache(MockEmbedder)
        state = {"messages": "async query about billing"}

        result = await CachedState.acheck(state, cache, query_key="messages")
        assert result["__cache_hit"] is False

        await CachedState.asave(result, "billing answer", cache)

        result2 = await CachedState.acheck(state, cache, query_key="messages")
        assert result2["__cache_hit"] is False
        assert "cached_response" not in result2
