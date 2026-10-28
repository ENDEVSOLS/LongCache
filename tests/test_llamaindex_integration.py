"""Tests for the LlamaIndex integration (CachedQueryEngine)."""

from unittest.mock import MagicMock, AsyncMock, patch

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


@patch("longcache.core.cache.LocalEmbedder")
class TestCachedQueryEngine:
    def _cached_engine(self, MockEmbedder, mock_engine):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.integrations.llamaindex import CachedQueryEngine

        engine = CachedQueryEngine(
            mock_engine,
            config=CacheConfig(backend="memory", similarity_threshold=0.9),
        )
        return engine

    def test_query_bypasses_opaque_context(self, MockEmbedder):
        mock_engine = MagicMock()
        mock_engine.query.return_value = "cached-worthy answer"

        engine = self._cached_engine(MockEmbedder, mock_engine)

        result1 = engine.query("What is the refund policy?")
        assert result1 == "cached-worthy answer"
        assert mock_engine.query.call_count == 1

        result2 = engine.query("What is the refund policy?")
        assert result2 == "cached-worthy answer"
        # Second call is a cache HIT — underlying engine not queried again
        assert mock_engine.query.call_count == 2

    def test_query_miss_for_different_queries(self, MockEmbedder):
        mock_engine = MagicMock()
        mock_engine.query.side_effect = lambda q, **kw: f"answer to {q}"

        engine = self._cached_engine(MockEmbedder, mock_engine)

        r1 = engine.query("query A")
        r2 = engine.query("completely unrelated query B")

        assert r1 == "answer to query A"
        assert r2 == "answer to completely unrelated query B"
        assert mock_engine.query.call_count == 2

    def test_proxy_attributes(self, MockEmbedder):
        mock_engine = MagicMock()
        mock_engine.retriever = "some_retriever"

        engine = self._cached_engine(MockEmbedder, mock_engine)
        assert engine.retriever == "some_retriever"

    async def test_aquery_bypasses_opaque_context(self, MockEmbedder):
        mock_engine = MagicMock()
        mock_engine.aquery = AsyncMock(return_value="async cached answer")

        engine = self._cached_engine(MockEmbedder, mock_engine)

        result1 = await engine.aquery("What is async RAG?")
        assert result1 == "async cached answer"
        assert mock_engine.aquery.call_count == 1

        result2 = await engine.aquery("What is async RAG?")
        assert result2 == "async cached answer"
        assert mock_engine.aquery.call_count == 2
