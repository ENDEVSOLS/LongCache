"""Public concurrency contract, synchronized by registration rather than sleeps."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from uuid import uuid4

import pytest

from longcache import (
    CacheConfig,
    LongCache,
    SingleflightTimeoutError,
    SingleflightLeaderError,
    SingleflightInvalidatedError,
    SingleflightReentrancyError,
    SingleflightCapacityError,
    RequestScope,
)
from dataclasses import replace
from tests.helpers import HashEmbedder, trusted_context


@pytest.fixture(params=[(b, i) for b in ("memory", "sqlite", "redis") for i in ("flat", "hnsw")])
def cache_v3(request, tmp_path):
    backend, index = request.param
    if backend == "redis" and not os.environ.get("LONGCACHE_TEST_REDIS_URL"):
        pytest.skip("Real Redis service required")
    config = CacheConfig(
        backend=backend,
        faiss_index_type=index,
        ttl_seconds=0,
        db_path=str(tmp_path / "cache.db"),
        generation=uuid4().hex,
        redis_url=os.environ.get("LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0"),
    )
    cache = LongCache(config, embedder=HashEmbedder(), context=trusted_context())
    yield cache
    cache.clear(operator=True)
    cache.close()


def joined_event(cache, monkeypatch, expected):
    ready = threading.Event()
    original = cache._execution.registry.claim

    def claim(key):
        result = original(key)
        if cache._execution.registry.snapshot()["followers"] >= expected:
            ready.set()
        return result

    monkeypatch.setattr(cache._execution.registry, "claim", claim)
    return ready


def test_twenty_identical_sync_requests_execute_once(cache_v3, monkeypatch):
    cache = cache_v3
    joined = joined_event(cache, monkeypatch, 19)
    barrier, calls = threading.Barrier(20), []

    def origin():
        calls.append(1)
        assert joined.wait(10)
        return {"answer": ["approved"]}

    def request(_):
        barrier.wait(timeout=10)
        return cache.get_or_compute("policy", origin)

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(request, range(20)))
    assert len(calls) == 1
    assert results == [{"answer": ["approved"]}] * 20
    results[0]["answer"].append("changed")
    assert results[1] == {"answer": ["approved"]}
    assert cache._execution.registry.snapshot() == {
        "active_flights": 0,
        "followers": 0,
        "retained_replay_bytes": 0,
    }
    stats = cache.stats()
    assert stats["hits"] == 19 and stats["misses"] == 1
    assert cache.entries()[0].hit_count == 19


@pytest.mark.asyncio
async def test_twenty_identical_async_requests_execute_once(cache_v3, monkeypatch):
    cache = cache_v3
    joined = joined_event(cache, monkeypatch, 19)
    calls = []

    async def origin():
        calls.append(1)
        assert await asyncio.to_thread(joined.wait, 10)
        return {"answer": "approved"}

    results = await asyncio.wait_for(
        asyncio.gather(*[cache.aget_or_compute("policy", origin) for _ in range(20)]), 15
    )
    assert results == [{"answer": "approved"}] * 20
    assert len(calls) == 1
    assert cache.stats()["hits"] == 19
    assert cache.stats()["misses"] == 1
    assert cache._execution.registry.snapshot()["followers"] == 0


def plain_cache(**kwargs):
    return LongCache(
        CacheConfig(backend="memory", ttl_seconds=0, **kwargs),
        embedder=HashEmbedder(),
        context=trusted_context(),
    )


def test_follower_timeout_does_not_start_another_origin():
    with plain_cache(singleflight_wait_timeout_seconds=0.02) as cache:
        entered, release, calls = threading.Event(), threading.Event(), []

        def origin():
            calls.append(1)
            entered.set()
            assert release.wait(5)
            return "answer"

        with ThreadPoolExecutor(max_workers=1) as pool:
            leader = pool.submit(cache.get_or_compute, "policy", origin)
            assert entered.wait(5)
            try:
                with pytest.raises(SingleflightTimeoutError):
                    cache.get_or_compute("policy", origin)
                assert len(calls) == 1
            finally:
                release.set()
            assert leader.result(timeout=5) == "answer"


def test_provider_error_releases_followers_without_retry(monkeypatch):
    with plain_cache() as cache:
        joined = joined_event(cache, monkeypatch, 1)
        calls = []

        def origin():
            calls.append(1)
            assert joined.wait(5)
            raise ValueError("provider failed")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(cache.get_or_compute, "policy", origin) for _ in range(2)]
            errors = [f.exception(timeout=5) for f in futures]
        assert any(type(e) is ValueError for e in errors)
        assert any(type(e) is SingleflightLeaderError for e in errors)
        assert len(calls) == 1
        assert cache.get_or_compute("policy", lambda: "recovered") == "recovered"


@pytest.mark.asyncio
async def test_follower_cancellation_does_not_cancel_leader(monkeypatch):
    async with plain_cache() as cache:
        entered, release = asyncio.Event(), asyncio.Event()
        joined = joined_event(cache, monkeypatch, 1)
        calls = []

        async def origin():
            calls.append(1)
            entered.set()
            await release.wait()
            return "answer"

        leader = asyncio.create_task(cache.aget_or_compute("policy", origin))
        await entered.wait()
        follower = asyncio.create_task(cache.aget_or_compute("policy", origin))
        assert await asyncio.to_thread(joined.wait, 5)
        follower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await follower
        release.set()
        assert await leader == "answer"
        assert len(calls) == 1
        assert cache._execution.registry.snapshot()["followers"] == 0


@pytest.mark.asyncio
async def test_leader_cancellation_releases_followers(monkeypatch):
    async with plain_cache() as cache:
        entered, release = asyncio.Event(), asyncio.Event()
        joined = joined_event(cache, monkeypatch, 1)

        async def origin():
            entered.set()
            await release.wait()
            return "answer"

        leader = asyncio.create_task(cache.aget_or_compute("policy", origin))
        await entered.wait()
        follower = asyncio.create_task(cache.aget_or_compute("policy", origin))
        assert await asyncio.to_thread(joined.wait, 5)
        leader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await leader
        with pytest.raises(SingleflightLeaderError):
            await asyncio.wait_for(follower, 5)
        assert cache.get("policy") is None


def test_empty_store_invalidation_fences_inflight_write(monkeypatch):
    with plain_cache() as cache:
        joined = joined_event(cache, monkeypatch, 1)
        release, entered = threading.Event(), threading.Event()

        def origin():
            entered.set()
            assert release.wait(5)
            return "old answer"

        with ThreadPoolExecutor(max_workers=2) as pool:
            leader = pool.submit(cache.get_or_compute, "policy", origin)
            assert entered.wait(5)
            follower = pool.submit(cache.get_or_compute, "policy", origin)
            assert joined.wait(5)
            try:
                assert cache.invalidate() == 0
                with pytest.raises(SingleflightInvalidatedError):
                    follower.result(timeout=5)
            finally:
                release.set()
            assert leader.result(timeout=5) == "old answer"
        assert cache.get("policy") is None


def test_recursive_execution_does_not_deadlock():
    with plain_cache() as cache:
        with pytest.raises(SingleflightReentrancyError):
            cache.get_or_compute("policy", lambda: cache.get_or_compute("policy", lambda: "never"))


@pytest.mark.parametrize("policy", ["raise", "bypass"])
def test_store_failure_never_repeats_provider(monkeypatch, policy):
    with plain_cache(cache_error_policy=policy) as cache:
        calls = []

        def origin():
            calls.append(1)
            return "answer"

        def unavailable(*_):
            raise OSError("storage offline")

        monkeypatch.setattr(cache.backend, "store", unavailable)
        if policy == "raise":
            from longcache import CacheOperationError

            with pytest.raises(CacheOperationError):
                cache.get_or_compute("policy", origin)
        else:
            assert cache.get_or_compute("policy", origin) == "answer"
        assert len(calls) == 1


def test_nonshareable_response_calls_each_origin_once(monkeypatch):
    with plain_cache() as cache:
        joined = joined_event(cache, monkeypatch, 1)
        calls = []

        def origin():
            calls.append(1)
            assert joined.wait(5)
            return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(cache.get_or_compute, "policy", origin) for _ in range(2)]
            assert [f.result(timeout=5) for f in futures] == [None, None]
        assert len(calls) == 2
        assert cache.entries() == []


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": RequestScope("other-tenant", "test-user")},
        {"scope": RequestScope("test-tenant", "other-user")},
        {"model": "other-model"},
        {"operation": "other.operation"},
        {"provider_scope": "other-deployment"},
        {"prompt_version": "v2"},
        {"corpus_version": "corpus-v2"},
        {"generation": {"temperature": 1}},
    ],
)
def test_context_changes_never_join_the_same_flight(changes):
    with plain_cache() as cache:
        barrier, calls = threading.Barrier(2), []

        def origin():
            calls.append(1)
            barrier.wait(timeout=5)
            return "answer"

        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(cache.get_or_compute, "policy", origin, context=trusted_context())
            b = pool.submit(
                cache.get_or_compute,
                "policy",
                origin,
                context=replace(trusted_context(), **changes),
            )
            assert a.result(timeout=5) == b.result(timeout=5) == "answer"
        assert len(calls) == 2


def test_capacity_rejection_does_not_bypass_into_duplicate_work():
    with plain_cache(singleflight_max_inflight=1, cache_error_policy="bypass") as cache:
        entered, release = threading.Event(), threading.Event()

        def origin():
            entered.set()
            assert release.wait(5)
            return "answer"

        with ThreadPoolExecutor(max_workers=1) as pool:
            leader = pool.submit(cache.get_or_compute, "policy", origin)
            assert entered.wait(5)
            try:
                with pytest.raises(SingleflightCapacityError):
                    cache.get_or_compute("another query", lambda: pytest.fail("must not run"))
            finally:
                release.set()
            assert leader.result(timeout=5) == "answer"


@pytest.mark.asyncio
async def test_sync_leader_and_async_follower_share_a_flight(monkeypatch):
    async with plain_cache() as cache:
        entered, release = threading.Event(), threading.Event()
        joined = joined_event(cache, monkeypatch, 1)

        def origin():
            entered.set()
            assert release.wait(5)
            return {"answer": "approved"}

        async def never():
            pytest.fail("follower must not execute origin")

        with ThreadPoolExecutor(max_workers=1) as pool:
            leader = pool.submit(cache.get_or_compute, "policy", origin)
            assert await asyncio.to_thread(entered.wait, 5)
            follower = asyncio.create_task(cache.aget_or_compute("policy", never))
            assert await asyncio.to_thread(joined.wait, 5)
            release.set()
            assert await follower == leader.result(timeout=5) == {"answer": "approved"}


def test_two_event_loops_can_share_one_cache(monkeypatch):
    with plain_cache() as cache:
        joined = joined_event(cache, monkeypatch, 1)
        calls = []

        async def origin():
            calls.append(1)
            assert await asyncio.to_thread(joined.wait, 5)
            return "answer"

        async def request():
            return await cache.aget_or_compute("policy", origin)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(asyncio.run, request()) for _ in range(2)]
            assert [f.result(timeout=10) for f in futures] == ["answer", "answer"]
        assert len(calls) == 1
