"""Tests for LangChain integration."""

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


@patch("longcache.core.cache.LocalEmbedder")
class TestCachedLLM:
    """Tests for the CachedLLM wrapper."""

    def test_invoke_bypasses_opaque_context(self, MockEmbedder):
        """Test that invoke() caches the LLM response."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.integrations.langchain import CachedLLM

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "LLM response about refunds"

        cached_llm = CachedLLM(
            mock_llm, config=CacheConfig(backend="memory", similarity_threshold=0.9)
        )

        # First call — MISS
        result1 = cached_llm.invoke("What is the refund policy?")
        assert result1 == "LLM response about refunds"
        assert mock_llm.invoke.call_count == 1

        # Second call with same query — HIT
        result2 = cached_llm.invoke("What is the refund policy?")
        assert result2 == "LLM response about refunds"
        # LLM should NOT be called again
        assert mock_llm.invoke.call_count == 2

    def test_extract_query_from_string(self, MockEmbedder):
        """Test extracting query from a plain string."""
        from longcache.integrations.langchain import CachedLLM

        cached_llm = CachedLLM(MagicMock())
        assert cached_llm._extract_query("Hello") == "Hello"

    def test_extract_query_from_messages(self, MockEmbedder):
        """Test extracting query from a message list."""
        from longcache.integrations.langchain import CachedLLM

        cached_llm = CachedLLM(MagicMock())

        # Create mock messages
        system_msg = MagicMock()
        system_msg.type = "system"
        system_msg.content = "You are helpful."

        human_msg = MagicMock()
        human_msg.type = "human"
        human_msg.content = "What is RAG?"

        result = cached_llm._extract_query([system_msg, human_msg])
        assert result == "What is RAG?"

    def test_extract_query_from_dict(self, MockEmbedder):
        """Test extracting query from a dict input."""
        from longcache.integrations.langchain import CachedLLM

        cached_llm = CachedLLM(MagicMock())
        result = cached_llm._extract_query({"question": "What is RAG?"})
        assert result == "What is RAG?"

    def test_proxy_attributes(self, MockEmbedder):
        """Test that unknown attributes are proxied to the underlying LLM."""
        from longcache.integrations.langchain import CachedLLM

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o"

        cached_llm = CachedLLM(mock_llm)
        assert cached_llm.model_name == "gpt-4o"


@patch("longcache.core.cache.LocalEmbedder")
class TestCachedChain:
    """Tests for the CachedChain wrapper."""

    def test_invoke_bypasses_opaque_context(self, MockEmbedder):
        """Test that invoke() caches the chain response."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache.integrations.langchain import CachedChain

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Chain response about RAG"

        cached_chain = CachedChain(
            mock_chain,
            config=CacheConfig(backend="memory", similarity_threshold=0.9),
        )

        # First call — MISS
        result1 = cached_chain.invoke({"question": "What is RAG?"})
        assert result1 == "Chain response about RAG"

        # Second call — HIT
        result2 = cached_chain.invoke({"question": "What is RAG?"})
        assert result2 == "Chain response about RAG"
        assert mock_chain.invoke.call_count == 2

    def test_custom_query_key(self, MockEmbedder):
        """Test extracting query with a custom key."""
        from longcache.integrations.langchain import CachedChain

        cached_chain = CachedChain(MagicMock(), query_key="input")
        result = cached_chain._extract_query({"input": "custom query"})
        assert result == "custom query"

    def test_proxy_attributes(self, MockEmbedder):
        """Test that unknown attributes are proxied to the underlying chain."""
        from longcache.integrations.langchain import CachedChain

        mock_chain = MagicMock()
        mock_chain.custom_attr = "test_value"

        cached_chain = CachedChain(mock_chain)
        assert cached_chain.custom_attr == "test_value"
