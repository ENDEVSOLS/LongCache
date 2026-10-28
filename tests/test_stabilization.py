"""Regression tests for the 0.1.1 cache and index fixes."""

import numpy as np
import pytest
from longcache import BaseEmbedder, EmbeddingDescriptor
from tests.helpers import trusted_context

from longcache.backends.memory import InMemoryCacheBackend
from longcache.config import CacheConfig
from longcache.core.cache import LongCache
from longcache.core.entry import CacheEntry
from longcache.core.index import SimilarityIndex


def vector(x, y):
    value = np.array([x, y], dtype=np.float32)
    return value / np.linalg.norm(value)


class SmallEmbedder(BaseEmbedder):
    dimension = 2
    descriptor = EmbeddingDescriptor("small", "v1", 2)

    def embed(self, query):
        return vector(1, 0) if query == "first" else vector(0, 1)


def make_cache(monkeypatch, config):
    return LongCache(config, embedder=SmallEmbedder(), context=trusted_context())


def test_lookup_skips_other_namespace_and_counts_once():
    backend = InMemoryCacheBackend()
    index = SimilarityIndex(backend, CacheConfig(backend="memory"))
    near = vector(1, 0)
    second = vector(0.99, 0.1)
    for namespace, value, response in [
        ("other", near, "wrong"),
        ("wanted", second, "right"),
    ]:
        entry = CacheEntry(namespace=namespace, query=response, vector=value, response=response)
        entry_id = backend.store(entry)
        index.add(value, entry_id)
    result = index.find_similar(near, threshold=0.9, namespace="wanted")
    assert result.response == "right"
    assert backend.retrieve(result.id).hit_count == 1


def test_lookup_skips_expired_and_orphaned_entries():
    backend = InMemoryCacheBackend()
    index = SimilarityIndex(backend, CacheConfig(backend="memory"))
    nearest = vector(1, 0)
    candidate = vector(0.99, 0.1)
    expired = CacheEntry(
        namespace="default", query="old", vector=nearest, response="old", expires_at=1.0
    )
    old_id = backend.store(expired)
    index.add(nearest, old_id)
    valid = CacheEntry(namespace="default", query="new", vector=candidate, response="new")
    valid_id = backend.store(valid)
    index.add(candidate, valid_id)
    assert index.find_similar(nearest, threshold=0.9).response == "new"
    backend.delete(valid_id)
    assert index.find_similar(nearest, threshold=0.9) is None


def test_inspection_does_not_record_a_hit(monkeypatch):
    cache = make_cache(monkeypatch, CacheConfig(backend="memory", ttl_seconds=0))
    cache.set("first", "answer")
    entry_id = cache.inspect("first")["id"]
    assert cache.backend.retrieve(entry_id).hit_count == 0
    assert cache.get("first") == "answer"
    assert cache.backend.retrieve(entry_id).hit_count == 1


def test_rebuild_keeps_custom_namespaces_after_invalidation(monkeypatch):
    cache = make_cache(monkeypatch, CacheConfig(backend="memory", ttl_seconds=0))
    cache.set("first", "default")
    cache.set("second", "custom", namespace="custom")
    cache.invalidate()
    assert cache.get("second", namespace="custom") == "custom"


def test_sqlite_restart_restores_context_and_custom_namespace(monkeypatch, tmp_path):
    config = CacheConfig(backend="sqlite", db_path=str(tmp_path / "cache.db"), ttl_seconds=0)
    first = make_cache(monkeypatch, config)
    first.set("first", "scoped", namespace="custom", model="model-a")
    first.set("second", "plain", namespace="other")
    first.close()
    second = make_cache(monkeypatch, config)
    assert second.get("first", namespace="custom", model="model-a") == "scoped"
    assert second.get("second", namespace="other") == "plain"
    second.close()


def test_rebuild_preserves_hnsw(monkeypatch):
    cache = make_cache(
        monkeypatch,
        CacheConfig(backend="memory", ttl_seconds=0, faiss_index_type="hnsw", hnsw_ef_search=32),
    )
    cache.set("first", "answer")
    cache._rebuild_index_from_backend()
    assert cache.index._faiss_index.__class__.__name__ == "IndexHNSWFlat"
    assert cache.index._faiss_index.hnsw.efSearch == 32


def test_changed_embedding_dimension_fails_clearly():
    backend = InMemoryCacheBackend()
    index = SimilarityIndex(backend, CacheConfig(backend="memory"))
    entry = CacheEntry(namespace="default", query="first", vector=vector(1, 0))
    index.add(entry.vector, backend.store(entry))
    with pytest.raises(ValueError, match="Embedding dimension changed"):
        index.find_similar(np.ones(3, dtype=np.float32))


def test_async_redis_cache_configuration_fails_fast():
    with pytest.raises(ValueError, match="async_redis"):
        LongCache(CacheConfig(backend="async_redis"))


def test_context_manager_stops_sweeper(monkeypatch):
    with make_cache(monkeypatch, CacheConfig(backend="memory", ttl_seconds=1)) as cache:
        assert cache._sweep_thread is None  # Memory has no sweeper resource to stop.
    assert cache._closed
    cache.close()


def test_decorator_bypasses_arguments_missing_from_cache_key(monkeypatch):
    cache = make_cache(monkeypatch, CacheConfig(backend="memory", ttl_seconds=0))
    calls = []

    @cache.cached
    def answer(query, user_id):
        calls.append(user_id)
        return user_id

    assert answer("first", "alice") == "alice"
    assert answer("first", "bob") == "bob"
    assert calls == ["alice", "bob"]
    assert cache.index.size == 0
