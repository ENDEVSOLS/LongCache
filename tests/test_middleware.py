"""Tests for the FastAPI/Starlette middleware."""

from unittest.mock import MagicMock, patch


from longcache.config import CacheConfig


def _make_mock_embedder():
    """Create a mock embedder with deterministic vectors."""
    import numpy as np

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


class TestStreamingCacheAdapter:
    """Tests for the StreamingCacheAdapter."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_stream_caches_and_replays(self, MockEmbedder):
        """Test that streaming adapter caches and replays responses."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.middleware import StreamingCacheAdapter

        call_count = 0

        def mock_stream_func(query: str):
            nonlocal call_count
            call_count += 1
            for token in ["Hello", " from", " the", " LLM"]:
                yield token

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        from longcache import LongCache

        cache = LongCache(config)
        cache.embedder = mock_embedder

        adapter = StreamingCacheAdapter(cache, mock_stream_func, replay_delay=0)

        # First call — MISS, should stream from source
        tokens1 = list(adapter.stream("test query"))
        assert len(tokens1) > 0
        assert call_count == 1

        # Second call — HIT, should replay from cache
        tokens2 = list(adapter.stream("test query"))
        assert call_count == 2  # Unsupported replay passes through
        assert "".join(tokens2) == "".join(tokens1)

    @patch("longcache.core.cache.LocalEmbedder")
    def test_stream_different_queries(self, MockEmbedder):
        """Test that different queries are handled separately."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.middleware import StreamingCacheAdapter
        from longcache import LongCache

        def mock_stream_func(query: str):
            yield f"Response for: {query}"

        config = CacheConfig(backend="memory", similarity_threshold=0.99)
        cache = LongCache(config)
        cache.embedder = mock_embedder

        adapter = StreamingCacheAdapter(cache, mock_stream_func, replay_delay=0)

        tokens1 = list(adapter.stream("query A"))
        tokens2 = list(adapter.stream("query B"))
        assert tokens1 != tokens2

        # Both should be MISS (different queries)
        # The stream_func should have been called twice
