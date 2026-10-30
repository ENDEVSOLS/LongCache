"""Cancellation does not abandon worker-owned resources or permit late writes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from longcache import CacheConfig, LongCache, PolicyDecision
from longcache.core.errors import CacheClosedError, CacheShutdownTimeoutError
from tests.helpers import HashEmbedder, trusted_context


def make_cache(tmp_path, **kwargs):
    return LongCache(
        CacheConfig(db_path=str(tmp_path / "cache.db"), ttl_seconds=0, **kwargs),
        embedder=HashEmbedder(),
        context=trusted_context(),
    )


def test_sqlite_connection_is_owned_and_closed_across_threads(tmp_path):
    cache = make_cache(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert all(
            workers.map(lambda n: cache.set(f"query {n}", {"n": n}).outcome == "STORED", range(20))
        )
        assert all(workers.map(lambda n: cache.get(f"query {n}") == {"n": n}, range(20)))
    connection = cache.backend._get_connection()
    cache.close()
    cache.close()
    assert cache._closed
    assert cache.backend._connection is None
    with pytest.raises(Exception, match="closed"):
        connection.execute("SELECT 1")
    with pytest.raises(CacheClosedError):
        cache.get("query")
    with pytest.raises(RuntimeError, match="closed"):
        cache.backend._get_connection()


@pytest.mark.asyncio
async def test_async_context_manager_closes_sqlite_workers(tmp_path):
    async with make_cache(tmp_path) as cache:
        await cache.aset("policy", {"answer": "approved"})
        assert await cache.aget("policy") == {"answer": "approved"}
    assert cache._closed
    assert cache._lifecycle.snapshot()["cache_jobs"] == 0
    assert cache.backend._connection is None


@pytest.mark.asyncio
async def test_cancelled_write_is_tracked_until_worker_exits(tmp_path):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def admission(*_):
        entered.set()
        assert release.wait(5)
        return PolicyDecision("accept")

    cache = LongCache(
        CacheConfig(db_path=str(tmp_path / "cache.db"), ttl_seconds=0),
        embedder=HashEmbedder(),
        context=trusted_context(),
        admit=admission,
    )
    task = asyncio.create_task(cache.aset("policy", "answer"))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(CacheShutdownTimeoutError):
        await cache.aclose(timeout=0.02)
    assert not cache._closed
    assert cache._lifecycle.snapshot()["state"] == "CLOSING"
    original = cache._lifecycle._finalize

    def finalize():
        original()
        finished.set()

    cache._lifecycle._finalize = finalize
    release.set()
    assert await asyncio.to_thread(finished.wait, 5)
    await cache.aclose()
    assert cache._closed
    with make_cache(tmp_path) as reopened:
        assert reopened.get("policy") is None


def test_close_inside_hook_does_not_deadlock(tmp_path):
    cache = make_cache(tmp_path)

    def hook(*_):
        with pytest.raises(CacheClosedError, match="own active"):
            cache.close()
        return PolicyDecision("accept")

    cache._admit = hook
    assert cache.set("policy", "answer").outcome == "STORED"
    cache.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("singleflight_enabled", "false"),
        ("singleflight_max_inflight", True),
        ("singleflight_max_waiters_per_key", 0),
        ("singleflight_wait_timeout_seconds", float("nan")),
        ("backend_timeout_seconds", float("inf")),
        ("shutdown_timeout_seconds", -1),
        ("cache_error_policy", "ignore"),
    ],
)
def test_reliability_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        CacheConfig(**{field: value})


def test_reliability_settings_environment_roundtrip(monkeypatch):
    monkeypatch.setenv("LONGCACHE_SINGLEFLIGHT_ENABLED", "false")
    monkeypatch.setenv("LONGCACHE_BACKEND_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("LONGCACHE_SINGLEFLIGHT_MAX_INFLIGHT", "12")
    config = CacheConfig.from_env()
    assert config.singleflight_enabled is False
    assert config.backend_timeout_seconds == 1.5
    assert config.singleflight_max_inflight == 12
