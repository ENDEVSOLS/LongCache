"""Tests for the LongCache main class."""

import concurrent.futures
import time
from unittest.mock import patch

import pytest

from longcache.config import CacheConfig
from longcache.core.cache import LongCache


from tests.helpers import HashEmbedder, trusted_context


def _make_mock_embedder():
    return HashEmbedder()


class TestLongCacheGetSet:
    """Tests for get() and set() operations."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_set_and_get_hit(self, MockEmbedder):
        """Test that storing and retrieving a query results in a cache hit."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        # Set a value
        cache.set("What is your refund policy?", "Our refund policy allows returns within 30 days.")

        # Get it back
        result = cache.get("What is your refund policy?")
        assert result is not None
        assert "refund policy" in result.lower() or "30 days" in result

    @patch("longcache.core.cache.LocalEmbedder")
    def test_get_miss(self, MockEmbedder):
        """Test that an uncached query returns None."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.99)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        # Empty cache should miss
        result = cache.get("anything")
        assert result is None

    @patch("longcache.core.cache.LocalEmbedder")
    def test_set_with_namespace(self, MockEmbedder):
        """Test namespace isolation."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", namespace="ns_A", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("query", "response_A", namespace="ns_A")

        # Same query in different namespace should not be found
        result = cache.get("query", namespace="ns_B")
        assert result is None

        # Should find in correct namespace
        result = cache.get("query", namespace="ns_A")
        assert result == "response_A"


class TestLongCacheDecorator:
    """Tests for the @cached decorator."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_cached_decorator_hit(self, MockEmbedder):
        """Test that the decorator caches function results."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        call_count = 0

        @cache.cached
        def expensive_func(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Result for: {query}"

        # First call — cache MISS
        result1 = expensive_func("What is RAG?")
        assert call_count == 1

        # Second call with same query — cache HIT
        result2 = expensive_func("What is RAG?")
        # Function should NOT be called again
        assert call_count == 1
        assert result1 == result2

    @patch("longcache.core.cache.LocalEmbedder")
    def test_cached_decorator_with_namespace(self, MockEmbedder):
        """Test decorator with custom namespace."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        @cache.cached(namespace="custom_ns")
        def my_func(query: str) -> str:
            return f"Response: {query}"

        result = my_func("test query")
        assert result == "Response: test query"


class TestLongCacheTTL:
    """Tests for TTL (time-to-live) functionality."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_ttl_expiry(self, MockEmbedder):
        """Test that entries expire after TTL."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9, ttl_seconds=1)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        # Set with very short TTL
        cache.set("query", "response", ttl=1)

        # Should be available immediately
        result = cache.get("query")
        assert result is not None

        # Wait for expiry
        time.sleep(1.5)

        # Should be expired now
        result = cache.get("query")
        assert result is None

    @patch("longcache.core.cache.LocalEmbedder")
    def test_ttl_zero_means_no_expiry(self, MockEmbedder):
        """Test that TTL=0 means entries never expire."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9, ttl_seconds=0)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("query", "response")

        # Should have no expiry
        result = cache.get("query")
        assert result is not None


class TestLongCacheStats:
    """Tests for cache statistics."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_stats_initial(self, MockEmbedder):
        """Test initial stats are zeroed."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", track_stats=True)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    @patch("longcache.core.cache.LocalEmbedder")
    def test_stats_after_operations(self, MockEmbedder):
        """Test stats after hits and misses."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9, track_stats=True)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("query1", "response1")

        # Hit
        cache.get("query1")
        # Miss
        cache.get("uncached_query")

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1  # get() now properly tracks misses
        assert stats["total_entries"] >= 1


class TestLongCacheInvalidate:
    """Tests for cache invalidation."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_invalidate_namespace(self, MockEmbedder):
        """Test invalidating a specific namespace."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("query", "response_A", namespace="ns_A")
        cache.set("query", "response_B", namespace="ns_B")

        # Invalidate ns_A
        count = cache.invalidate("ns_A")
        assert count >= 1

        # ns_A should be gone
        result = cache.get("query", namespace="ns_A")
        assert result is None

    @patch("longcache.core.cache.LocalEmbedder")
    def test_clear_all(self, MockEmbedder):
        """Test clearing the entire cache."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("q1", "r1", namespace="ns_A")
        cache.set("q2", "r2", namespace="ns_B")

        count = cache.clear(operator=True)
        assert count >= 2

        stats = cache.stats()
        assert stats["total_entries"] == 0


class TestLongCacheInspect:
    """Tests for the inspect() method."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_inspect_finds_entry(self, MockEmbedder):
        """Test that inspect finds the nearest entry."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", similarity_threshold=0.9)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("What is the refund policy?", "Our policy allows returns within 30 days.")

        # Use the same query string to ensure a HIT with the mock embedder
        result = cache.inspect("What is the refund policy?")
        assert result is not None
        assert result["outcome"] == "HIT"
        assert "query" not in result
        assert "similarity" in result

    @patch("longcache.core.cache.LocalEmbedder")
    def test_inspect_empty_cache(self, MockEmbedder):
        """Test inspect on an empty cache."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory")
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        result = cache.inspect("anything")
        assert result["outcome"] == "MISS"


class TestLongCacheEviction:
    """Tests for max_entries eviction."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_eviction_on_exceed(self, MockEmbedder):
        """Test that oldest entries are evicted when max_entries is exceeded."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(
            backend="memory",
            similarity_threshold=0.9,
            max_entries=3,
        )
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        # Add 5 entries with distinct queries
        for i in range(5):
            cache.set(f"query_{i}", f"response_{i}")

        # Cache should have been evicted down to max_entries
        stats = cache.stats()
        assert stats["total_entries"] <= 3


class TestThreadSafetyAndIndexLeak:
    """Tests for stats thread safety and index rebuild memory leak prevention."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_stats_concurrent_thread_safety(self, MockEmbedder):
        """Verify that concurrent hits and misses do not drop counts or crash stats()."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory", track_stats=True)
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        num_threads = 8
        ops_per_thread = 50
        errors = []

        cache.set("counted", "answer")

        def worker():
            for _ in range(ops_per_thread):
                cache.get("counted")
                cache.get("absent")
                try:
                    s = cache.stats()
                    assert "hits" in s
                except Exception as e:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            concurrent.futures.wait(futures)

        assert not errors, f"Errors during concurrent stats access: {errors}"
        stats = cache.stats()
        expected_total = num_threads * ops_per_thread
        assert stats["hits"] == expected_total
        assert stats["misses"] == expected_total
        assert stats["total_saved_ms"] == 0.0

    @patch("longcache.core.cache.LocalEmbedder")
    def test_rebuild_index_no_vector_duplication(self, MockEmbedder):
        """Verify that repeated _rebuild_index_from_backend calls do not accumulate duplicate vectors."""
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config = CacheConfig(backend="memory")
        cache = LongCache(config, context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("query_alpha", "response_alpha")
        cache.set("query_beta", "response_beta")
        cache.set("query_gamma", "response_gamma")

        initial_size = cache.index.size
        assert initial_size == 3

        # Repeatedly rebuild index from backend (e.g. as sweeper or eviction would trigger)
        for _ in range(5):
            cache._rebuild_index_from_backend()

        # Before fix, index.size would have grown to 3 + 3 * 5 = 18
        assert cache.index.size == 3


class TestCompositeCacheKeys:
    """Tests for model, system_prompt, and temperature composite keys."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_model_isolation(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        config_gpt = CacheConfig(backend="memory", model="gpt-4o")
        cache_gpt = LongCache(config_gpt, context=trusted_context())
        cache_gpt.embedder = mock_embedder

        config_flash = CacheConfig(backend="memory", model="gemini-1.5-flash")
        cache_flash = LongCache(config_flash, context=trusted_context())
        cache_flash.embedder = mock_embedder

        query = "Explain backpropagation"
        cache_gpt.set(query, "GPT-4o response")

        assert cache_gpt.get(query) == "GPT-4o response"
        assert cache_flash.get(query) is None

    @patch("longcache.core.cache.LocalEmbedder")
    def test_system_prompt_isolation(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        cache = LongCache(CacheConfig(backend="memory"), context=trusted_context())
        cache.embedder = mock_embedder

        query = "What is a stock?"
        cache.set(query, "Finance answer", system_prompt="You are a wall street trader")

        assert cache.get(query, system_prompt="You are a wall street trader") == "Finance answer"
        assert cache.get(query, system_prompt="You are a kindergarten teacher") is None

    @patch("longcache.core.cache.LocalEmbedder")
    def test_inspect_includes_context_and_tags(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        cache = LongCache(CacheConfig(backend="memory"), context=trusted_context())
        cache.embedder = mock_embedder

        cache.set(
            "What is compound interest?",
            "Math answer",
            model="gpt-4o",
            system_prompt="Act as banker",
            temperature=0.7,
            tags=["finance", "math"],
        )

        entry = cache.inspect(
            "What is compound interest?",
            model="gpt-4o",
            system_prompt="Act as banker",
            temperature=0.7,
        )
        assert entry is not None
        assert entry["outcome"] == "HIT"
        assert "system_prompt" not in entry
        assert "query" not in entry
        assert "similarity" in entry


class TestTagBasedInvalidation:
    """Tests for tag-based cache invalidation."""

    @patch("longcache.core.cache.LocalEmbedder")
    def test_invalidate_by_tag_memory(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        cache = LongCache(CacheConfig(backend="memory"), context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("q1", "r1", tags=["policy", "v1"])
        cache.set("q2", "r2", tags=["policy", "v2"])
        cache.set("q3", "r3", tags=["faq"])

        deleted = cache.invalidate_by_tag("v1")
        assert deleted == 1
        assert cache.get("q1") is None
        assert cache.get("q2") == "r2"
        assert cache.get("q3") == "r3"

    @patch("longcache.core.cache.LocalEmbedder")
    def test_invalidate_by_tag_sqlite(self, MockEmbedder, tmp_path):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        db_file = str(tmp_path / "tags_test.db")
        cache = LongCache(CacheConfig(backend="sqlite", db_path=db_file), context=trusted_context())
        cache.embedder = mock_embedder

        cache.set("q_refund", "Refund details", tags=["doc:refund", "version:1"])
        cache.set("q_terms", "Terms of service", tags=["doc:terms", "version:1"])

        assert cache.get("q_refund") == "Refund details"
        assert cache.get("q_terms") == "Terms of service"

        deleted = cache.invalidate_by_tag("doc:refund")
        assert deleted == 1
        assert cache.get("q_refund") is None
        assert cache.get("q_terms") == "Terms of service"


class TestStreamingSupport:
    """Tests for cached_stream decorator."""

    @patch("longcache.core.cache.LocalEmbedder")
    @pytest.mark.asyncio
    async def test_cached_stream_async(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        cache = LongCache(CacheConfig(backend="memory"), context=trusted_context())
        cache.embedder = mock_embedder

        call_count = 0

        @cache.cached_stream
        async def fake_llm_stream(query: str):
            nonlocal call_count
            call_count += 1
            tokens = ["Hello", " ", "world", "!"]
            for token in tokens:
                yield token

        # First call: stream from generator
        stream1 = [token async for token in fake_llm_stream("greet me")]
        assert "".join(stream1) == "Hello world!"
        assert call_count == 1

        # Second call: HIT, re-streams cached text
        stream2 = [token async for token in fake_llm_stream("greet me")]
        assert "".join(stream2) == "Hello world!"
        assert call_count == 2  # Streaming is deliberately pass-through in v0.2

    @patch("longcache.core.cache.LocalEmbedder")
    def test_cached_stream_sync(self, MockEmbedder):
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        cache = LongCache(CacheConfig(backend="memory"), context=trusted_context())
        cache.embedder = mock_embedder

        call_count = 0

        @cache.cached_stream
        def fake_sync_stream(query: str):
            nonlocal call_count
            call_count += 1
            for word in ["Quick", " ", "brown", " ", "fox"]:
                yield word

        # First call: MISS
        out1 = list(fake_sync_stream("fox query"))
        assert "".join(out1) == "Quick brown fox"
        assert call_count == 1

        # Second call: HIT
        out2 = list(fake_sync_stream("fox query"))
        assert "".join(out2) == "Quick brown fox"
        assert call_count == 2
