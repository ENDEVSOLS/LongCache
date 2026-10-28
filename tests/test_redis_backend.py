"""
Integration tests for the Redis and AsyncRedis cache backends.

These tests require a live Redis server. The URL is read from the
LONGCACHE_TEST_REDIS_URL environment variable (default:
redis://localhost:16379/0, matching the throwaway container used
during development). If no server is reachable, the whole module
is skipped.
"""

import os
import time

import numpy as np
import pytest
import pytest_asyncio

from longcache.core.entry import CacheEntry
from longcache.backends.redis import RedisCacheBackend
from longcache.backends.async_redis import AsyncRedisCacheBackend

REDIS_URL = os.environ.get("LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0")


def _redis_available() -> bool:
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=1)
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(), reason=f"No Redis server reachable at {REDIS_URL}"
)


def make_entry(
    namespace: str = "default",
    query: str = "test query",
    response="test response",
    expires_at=None,
) -> CacheEntry:
    vector = np.random.randn(384).astype(np.float32)
    vector /= np.linalg.norm(vector)
    return CacheEntry(
        namespace=namespace,
        query=query,
        vector=vector,
        response=response,
        created_at=time.time(),
        expires_at=expires_at,
    )


# ---------- Sync Redis Backend ----------


class TestRedisBackend:
    @pytest.fixture
    def backend(self):
        b = RedisCacheBackend(redis_url=REDIS_URL)
        b.clear()
        yield b
        b.clear()

    def test_store_and_retrieve(self, backend):
        entry = make_entry()
        entry_id = backend.store(entry)
        assert entry_id > 0

        result = backend.retrieve(entry_id)
        assert result is not None
        assert result.query == "test query"
        assert result.response == "test response"

    def test_vector_round_trip(self, backend):
        original = make_entry()
        entry_id = backend.store(original)

        result = backend.retrieve(entry_id)
        np.testing.assert_array_almost_equal(original.vector, result.vector, decimal=6)

    def test_retrieve_nonexistent(self, backend):
        assert backend.retrieve(999999) is None

    def test_delete(self, backend):
        entry_id = backend.store(make_entry())
        assert backend.delete(entry_id) is True
        assert backend.retrieve(entry_id) is None

    def test_delete_namespace(self, backend):
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        count = backend.delete_namespace("ns_A")
        assert count == 2
        assert backend.count("ns_A") == 0
        assert backend.count("ns_B") == 1

    def test_clear(self, backend):
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        count = backend.clear()
        assert count == 2

    def test_count(self, backend):
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_A"))
        backend.store(make_entry(namespace="ns_B"))

        assert backend.count("ns_A") == 2
        assert backend.count("ns_B") == 1
        assert backend.count("ns_C") == 0

    def test_list_entries(self, backend):
        backend.store(make_entry(query="q1", namespace="ns_A"))
        backend.store(make_entry(query="q2", namespace="ns_A"))
        backend.store(make_entry(query="q3", namespace="ns_B"))

        entries = backend.list_entries("ns_A")
        assert len(entries) == 2
        assert {e.query for e in entries} == {"q1", "q2"}

    def test_update_hit_count(self, backend):
        entry_id = backend.store(make_entry())

        backend.update_hit_count(entry_id)
        backend.update_hit_count(entry_id)
        result = backend.retrieve(entry_id)
        assert result.hit_count == 2

    def test_evict_oldest(self, backend):
        for i in range(5):
            entry = make_entry(query=f"q{i}")
            entry.created_at = 1000.0 + i
            backend.store(entry)

        evicted = backend.evict_oldest("default", 2)
        assert evicted == 2
        assert backend.count("default") == 3

    def test_ttl_expiry_via_redis_expire(self, backend):
        """Entries with a short TTL should be gone from Redis after it elapses."""
        entry = make_entry(expires_at=time.time() + 1)
        entry_id = backend.store(entry)
        assert backend.retrieve(entry_id) is not None

        time.sleep(1.5)
        # Redis EXPIRE should have removed the key server-side
        assert backend.retrieve(entry_id) is None

    def test_get_expired(self, backend):
        expired_id = backend.store(make_entry(expires_at=time.time() - 1))
        assert backend.retrieve(expired_id) is None
        valid_id = backend.store(make_entry(expires_at=time.time() + 3600))

        expired = backend.get_expired("default")
        # The expired entry may already be auto-deleted by list_entries'
        # retrieve() call, or reported explicitly — either is acceptable
        # as long as it is not treated as valid.
        assert valid_id not in expired

    def test_json_response_round_trip(self, backend):
        entry = make_entry(response={"key": "value", "nested": [1, 2, 3]})
        entry_id = backend.store(entry)

        result = backend.retrieve(entry_id)
        assert isinstance(result.response, dict)
        assert result.response["key"] == "value"
        assert result.response["nested"] == [1, 2, 3]

    def test_namespace_isolation_across_instances(self, backend):
        """Entries stored by one backend instance are visible via another,
        proving namespace/entry state lives in Redis, not in-process."""
        entry_id = backend.store(make_entry(namespace="shared"))

        other = RedisCacheBackend(redis_url=REDIS_URL)
        result = other.retrieve(entry_id)
        assert result is not None
        assert result.namespace == "shared"

    def test_longcache_restart_restores_every_namespace(self, backend):
        from longcache import CacheConfig, LongCache

        from tests.helpers import HashEmbedder, trusted_context

        config = CacheConfig(backend="redis", redis_url=REDIS_URL, ttl_seconds=0)
        first = LongCache(config, embedder=HashEmbedder(), context=trusted_context())
        first.set("same", "A", namespace="A")
        first.set("same", "B", namespace="B")
        first.close()
        restarted = LongCache(config, embedder=HashEmbedder(), context=trusted_context())
        assert restarted.get("same", namespace="A") == "A"
        assert restarted.get("same", namespace="B") == "B"
        restarted.close()


# ---------- Async Redis Backend ----------


class TestAsyncRedisBackend:
    @pytest_asyncio.fixture
    async def backend(self):
        b = AsyncRedisCacheBackend(redis_url=REDIS_URL)
        await b.clear()
        yield b
        await b.clear()

    async def test_store_and_retrieve(self, backend):
        entry = make_entry()
        entry_id = await backend.store(entry)
        assert entry_id > 0

        result = await backend.retrieve(entry_id)
        assert result is not None
        assert result.query == "test query"

    async def test_delete(self, backend):
        entry_id = await backend.store(make_entry())
        assert await backend.delete(entry_id) is True
        assert await backend.retrieve(entry_id) is None

    async def test_delete_namespace(self, backend):
        await backend.store(make_entry(namespace="ns_A"))
        await backend.store(make_entry(namespace="ns_A"))
        await backend.store(make_entry(namespace="ns_B"))

        count = await backend.delete_namespace("ns_A")
        assert count == 2
        assert await backend.count("ns_A") == 0
        assert await backend.count("ns_B") == 1

    async def test_count_and_list_entries(self, backend):
        await backend.store(make_entry(query="q1", namespace="ns_A"))
        await backend.store(make_entry(query="q2", namespace="ns_A"))

        assert await backend.count("ns_A") == 2
        entries = await backend.list_entries("ns_A")
        assert {e.query for e in entries} == {"q1", "q2"}

    async def test_update_hit_count(self, backend):
        entry_id = await backend.store(make_entry())
        await backend.update_hit_count(entry_id)
        result = await backend.retrieve(entry_id)
        assert result.hit_count == 1

    async def test_evict_oldest(self, backend):
        for i in range(5):
            entry = make_entry(query=f"q{i}")
            entry.created_at = 1000.0 + i
            await backend.store(entry)

        evicted = await backend.evict_oldest("default", 2)
        assert evicted == 2
        assert await backend.count("default") == 3

    async def test_enforce_max_entries_on_store(self):
        """AsyncRedisCacheBackend auto-evicts on store() when over max_entries."""
        backend = AsyncRedisCacheBackend(redis_url=REDIS_URL, max_entries=3)
        await backend.clear()
        try:
            for i in range(5):
                entry = make_entry(query=f"q{i}", namespace="capped")
                entry.created_at = 1000.0 + i
                await backend.store(entry)

            assert await backend.count("capped") <= 3
        finally:
            await backend.clear()

    async def test_ttl_expiry_via_redis_expire(self, backend):
        entry = make_entry(expires_at=time.time() + 1)
        entry_id = await backend.store(entry)
        assert await backend.retrieve(entry_id) is not None

        import asyncio

        await asyncio.sleep(1.5)
        assert await backend.retrieve(entry_id) is None

    async def test_sync_and_async_backends_share_state(self):
        """Sync and async backends against the same Redis instance
        should see each other's writes — this is the whole point of
        the Redis backend (multi-instance sharing)."""
        sync_backend = RedisCacheBackend(redis_url=REDIS_URL)
        async_backend = AsyncRedisCacheBackend(redis_url=REDIS_URL)

        entry_id = sync_backend.store(make_entry(namespace="cross"))
        result = await async_backend.retrieve(entry_id)

        assert result is not None
        assert result.namespace == "cross"

        await async_backend.delete_namespace("cross")


def test_v2_clear_preserves_legacy_keyspace():
    backend = RedisCacheBackend(redis_url=REDIS_URL, cache_id="migration-check")
    legacy = "longcache:entry:v011-migration-fixture"
    backend._client.set(legacy, "preserved legacy data")
    try:
        backend.store(make_entry(namespace="owned"))
        backend.clear()
        assert backend._client.get(legacy) == "preserved legacy data"
    finally:
        backend._client.delete(legacy)
        backend.close()


@pytest.mark.asyncio
async def test_async_cache_over_sync_redis_obeys_scope_and_manifest():
    from longcache import CacheConfig, LongCache, RequestScope, EmbeddingDescriptor
    from tests.helpers import HashEmbedder, trusted_context

    config = CacheConfig(
        backend="redis", redis_url=REDIS_URL, cache_id="v02-async-contract", ttl_seconds=0
    )
    with LongCache(config, embedder=HashEmbedder()) as cache:
        cache.clear(operator=True)
        stored = await cache.aset("refund", "answer", context=trusted_context())
        assert stored.outcome == "STORED"
        assert await cache.aget("refund", context=trusted_context()) == "answer"
        other = trusted_context(scope=RequestScope("test-tenant", "other"))
        assert (await cache.alookup("refund", context=other)).outcome == "MISS"
    different = HashEmbedder()
    different.descriptor = EmbeddingDescriptor("test-hash", "different-revision", 384)
    with pytest.raises(ValueError, match="Incompatible"):
        LongCache(config, embedder=different)
    backend = RedisCacheBackend(redis_url=REDIS_URL, cache_id="v02-async-contract")
    backend.clear()
    backend.close()
