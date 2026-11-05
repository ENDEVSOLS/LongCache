from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from longcache import CacheConfig
from longcache.backends.memory import InMemoryCacheBackend
from longcache.backends.redis import RedisCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend
from longcache.core.entry import CacheEntry
from longcache.core.index import SimilarityIndex
from longcache.core.types import Float32Vector


def _vector(dim: int = 384, seed: int = 42) -> Float32Vector:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_sqlite_iter_index_entries_paged_and_payload_free() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "cache.db")
        backend = SQLiteCacheBackend(db_path)
        vec1 = _vector(384, 1)
        vec2 = _vector(384, 2)
        vec3 = _vector(384, 3)

        backend.store(
            CacheEntry(query="q1", vector=vec1, response={"ans": "one" * 1000}, metadata={"m": 1})
        )
        backend.store(
            CacheEntry(query="q2", vector=vec2, response={"ans": "two" * 1000}, metadata={"m": 2})
        )
        backend.store(
            CacheEntry(query="q3", vector=vec3, response={"ans": "three" * 1000}, metadata={"m": 3})
        )

        # Iterate with batch_size=2 to test pagination across batches
        items = list(backend.iter_index_entries(batch_size=2))
        assert len(items) == 3
        for item in items:
            assert item.id is not None
            assert item.vector is not None
            assert item.query == ""  # payload-free
            assert item.response is None  # payload-free
            assert "m" in item.metadata

        backend.close()


def test_similarity_index_record_remote_invalidation_threshold() -> None:
    backend = InMemoryCacheBackend()
    config = CacheConfig(similarity_threshold=0.9)
    index = SimilarityIndex(backend, config)

    # Empty index
    assert index.raw_total == 0
    assert index.orphan_count == 0
    assert not index.record_remote_invalidation(1)

    # Add 100 vectors
    for i in range(1, 101):
        index.add(_vector(384, i), entry_id=i)

    assert index.raw_total == 100
    assert index.size == 100
    assert index.orphan_count == 1  # From previous call

    # 10 invalidations: 11 / 100 = 11% < 25% -> False
    assert not index.record_remote_invalidation(10)
    assert index.orphan_count == 11

    # 15 more invalidations: 26 / 100 = 26% >= 25% (and >= 64 threshold test)
    # With < 64 orphans, returns False to avoid thrashing on small indexes
    assert not index.record_remote_invalidation(15)
    assert index.orphan_count == 26

    # Bring orphans above 64
    assert index.record_remote_invalidation(40)  # 66 orphans out of 100 (66% >= 25% and >= 64)
    assert index.orphan_count == 66


@pytest.mark.skipif(not os.environ.get("LONGCACHE_TEST_REDIS_URL"), reason="Redis not configured")
def test_redis_backends_iter_index_entries() -> None:
    redis_url = os.environ["LONGCACHE_TEST_REDIS_URL"]
    backend = RedisCacheBackend(redis_url=redis_url, cache_id="test-paged-redis")
    backend.clear()

    vec1 = _vector(384, 10)
    vec2 = _vector(384, 20)
    backend.store(
        CacheEntry(
            namespace="test_paged_redis",
            query="q1",
            vector=vec1,
            response={"ans": "huge" * 500},
            metadata={"id": 10},
        )
    )
    backend.store(
        CacheEntry(
            namespace="test_paged_redis",
            query="q2",
            vector=vec2,
            response={"ans": "huge" * 500},
            metadata={"id": 20},
        )
    )

    entries = list(backend.iter_index_entries(batch_size=1))
    assert len(entries) >= 2
    for e in entries:
        assert e.query == ""
        assert e.response is None
        assert e.vector is not None
    backend.clear()
