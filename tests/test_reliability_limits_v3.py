"""Resource bounds and shutdown races exercised without network/model downloads."""

import asyncio
import threading
import subprocess
import sys
import textwrap
import pytest

from longcache import (
    CacheConfig,
    CacheCapacityError,
    CacheShutdownTimeoutError,
    LongCache,
    PolicyDecision,
    SingleflightCapacityError,
    SingleflightTimeoutError,
    CacheClosedError,
)
from tests.helpers import HashEmbedder, trusted_context
from tests.test_singleflight_v3 import plain_cache, joined_event
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.asyncio
async def test_cancelled_queued_jobs_keep_capacity_until_dequeued():
    entered, release, lock = threading.Event(), threading.Event(), threading.Lock()
    running = []

    def admission(*_):
        with lock:
            running.append(1)
            if len(running) == 4:
                entered.set()
        assert release.wait(5)
        return PolicyDecision("accept")

    cache = LongCache(
        CacheConfig(backend="memory", ttl_seconds=0),
        embedder=HashEmbedder(),
        context=trusted_context(),
        admit=admission,
    )
    tasks = [asyncio.create_task(cache.aset(f"policy {n}", "answer")) for n in range(64)]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert cache.metrics(operator=True)["resources"]["cache_jobs"] == 64
        with pytest.raises(CacheCapacityError):
            await cache.aset("must not enter another queue item", "answer")
    finally:
        release.set()
        await cache.aclose()
    assert cache.backend.list_all_entries() == []
    assert cache.metrics(operator=True)["resources"]["cache_jobs"] == 0


@pytest.mark.asyncio
async def test_shutdown_deadline_cancels_owned_async_leader(monkeypatch):
    cache = plain_cache()
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
    with pytest.raises(CacheShutdownTimeoutError):
        await cache.aclose(timeout=0.02)
    results = await asyncio.wait_for(asyncio.gather(leader, follower, return_exceptions=True), 5)
    assert all(isinstance(result, BaseException) for result in results)
    await cache.aclose()
    assert cache._closed
    assert cache.metrics(operator=True)["resources"]["active_flights"] == 0


def test_waiter_capacity_does_not_duplicate_the_origin(monkeypatch):
    with plain_cache(singleflight_max_waiters_per_key=1) as cache:
        entered, release = threading.Event(), threading.Event()
        joined = joined_event(cache, monkeypatch, 1)

        def origin():
            entered.set()
            assert release.wait(5)
            return "answer"

        with ThreadPoolExecutor(max_workers=2) as workers:
            leader = workers.submit(cache.get_or_compute, "policy", origin)
            assert entered.wait(5)
            follower = workers.submit(cache.get_or_compute, "policy", origin)
            assert joined.wait(5)
            try:
                with pytest.raises(SingleflightCapacityError):
                    cache.get_or_compute("policy", lambda: pytest.fail("duplicate origin"))
            finally:
                release.set()
            assert leader.result(timeout=5) == follower.result(timeout=5) == "answer"


@pytest.mark.asyncio
async def test_one_hundred_sqlite_lifecycles_do_not_accumulate_owned_connections(tmp_path):
    for n in range(100):
        cache = LongCache(
            CacheConfig(db_path=str(tmp_path / f"cache-{n}.db"), ttl_seconds=0),
            embedder=HashEmbedder(),
            context=trusted_context(),
        )
        await cache.aset("policy", "answer")
        await cache.aclose()
        assert cache.backend._connection is None
        assert cache.metrics(operator=True)["resources"]["cache_jobs"] == 0
        assert cache.metrics(operator=True)["resources"]["active_operations"] == 0


def test_low_sample_percentiles_are_not_release_evidence():
    from evaluation.reliability import summarize, compare

    values = summarize([1.0] * 20)
    assert values["p99_ms"] is None
    assert values["percentile_evidence_sufficient"] is False
    budgets = {
        "warm_p95_multiplier": 1.15,
        "warm_p95_add_ms": 1.0,
        "warm_p99_multiplier": 1.2,
        "warm_p99_add_ms": 2.0,
    }
    assert compare(values, values, budgets)["p99_passed"] is None
    baseline = summarize([1.0] * 10000)
    regression = summarize([5.0] * 10000)
    assert compare(regression, baseline, budgets)["p95_passed"] is False
    assert compare(regression, baseline, budgets)["p99_passed"] is False


def test_failed_resource_cleanup_does_not_claim_closed(monkeypatch):
    cache = plain_cache()

    def broken():
        raise OSError("resource cleanup failed")

    monkeypatch.setattr(cache.backend, "close", broken, raising=False)
    with pytest.raises(CacheClosedError, match="resources may remain open"):
        cache.close()
    assert cache.metrics(operator=True)["resources"]["state"] == "CLOSING"
    assert not cache._closed
    with pytest.raises(CacheClosedError):
        cache.close()


@pytest.mark.asyncio
async def test_cancellation_after_election_does_not_abandon_a_flight(monkeypatch):
    cache = plain_cache()
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original_start = cache._execution.start
    original_complete = cache._execution.registry.complete

    def start(*args, **kwargs):
        state = original_start(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return state

    def complete(*args):
        original_complete(*args)
        completed.set()

    monkeypatch.setattr(cache._execution, "start", start)
    monkeypatch.setattr(cache._execution.registry, "complete", complete)

    async def never():
        pytest.fail("cancelled caller cannot start origin")

    task = asyncio.create_task(cache.aget_or_compute("policy", never))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(completed.wait, 5)
    await cache.aclose()
    assert cache.metrics(operator=True)["resources"]["active_flights"] == 0


def test_uncooperative_sync_provider_cannot_hang_shutdown_process():
    script = textwrap.dedent("""
        import threading
        import numpy as np
        from longcache import (BaseEmbedder, EmbeddingDescriptor, CacheConfig, LongCache,
            RequestContext, RequestScope, CacheShutdownTimeoutError)
        class Embedder(BaseEmbedder):
            dimension = 2
            descriptor = EmbeddingDescriptor("fixture", "v1", 2)
            def embed(self, text):
                return np.array([1, 0], dtype=np.float32)
        ctx = RequestContext(scope=RequestScope("tenant", "user"), operation="fixture",
            provider="fixture", provider_scope="v1", model="fixture", prompt_version="v1")
        cache = LongCache(CacheConfig(backend="memory", ttl_seconds=0), embedder=Embedder(), context=ctx)
        entered = threading.Event()
        def blocked():
            entered.set()
            threading.Event().wait()
        thread = threading.Thread(target=lambda: cache.get_or_compute("policy", blocked), daemon=True)
        thread.start()
        assert entered.wait(2)
        try:
            cache.close(timeout=0.02)
        except CacheShutdownTimeoutError:
            pass
        else:
            raise AssertionError("Shutdown incorrectly claimed completion")
        assert cache.metrics(operator=True)["resources"]["state"] == "CLOSING"
        assert not cache._closed
    """)
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


@pytest.mark.asyncio
async def test_timed_out_followers_do_not_accumulate_shared_future_callbacks():
    async with plain_cache(singleflight_wait_timeout_seconds=0.001) as cache:
        entered, release = asyncio.Event(), asyncio.Event()

        async def origin():
            entered.set()
            await release.wait()
            return "answer"

        leader = asyncio.create_task(cache.aget_or_compute("policy", origin))
        await entered.wait()
        flight = next(iter(cache._execution.registry._flights.values()))
        try:
            for _ in range(50):
                with pytest.raises(SingleflightTimeoutError):
                    await cache.aget_or_compute("policy", origin)
                assert flight.async_waiters == {}
                assert flight.future._done_callbacks == []
                assert cache.metrics(operator=True)["resources"]["followers"] == 0
        finally:
            release.set()
            assert await leader == "answer"


@pytest.mark.asyncio
async def test_concurrent_close_callers_share_one_cleanup_worker():
    cache = plain_cache()
    entered, release = threading.Event(), threading.Event()

    def blocked(*_):
        entered.set()
        assert release.wait(5)
        return PolicyDecision("accept")

    cache._admit = blocked
    write = asyncio.create_task(cache.aset("policy", "answer"))
    assert await asyncio.to_thread(entered.wait, 5)
    closers = [asyncio.create_task(cache.aclose()) for _ in range(50)]
    # Each closer has started once the event-loop sentinel is reached.
    sentinel = asyncio.Event()
    asyncio.get_running_loop().call_soon(sentinel.set)
    await sentinel.wait()
    assert cache.metrics(operator=True)["resources"]["cleanup_workers"] == 1
    for closer in closers[:-1]:
        closer.cancel()
    await asyncio.gather(*closers[:-1], return_exceptions=True)
    release.set()
    await write
    await closers[-1]
    assert cache._closed
    assert cache.metrics(operator=True)["resources"]["cleanup_workers"] == 0


@pytest.mark.asyncio
async def test_joining_close_caller_can_use_a_shorter_wait_timeout():
    cache = plain_cache()
    entered, release = threading.Event(), threading.Event()

    def blocked(*_):
        entered.set()
        assert release.wait(5)
        return PolicyDecision("accept")

    cache._admit = blocked
    write = asyncio.create_task(cache.aset("policy", "answer"))
    assert await asyncio.to_thread(entered.wait, 5)
    first = asyncio.create_task(cache.aclose(timeout=2))
    sentinel = asyncio.Event()
    asyncio.get_running_loop().call_soon(sentinel.set)
    await sentinel.wait()
    try:
        with pytest.raises(CacheShutdownTimeoutError):
            await cache.aclose(timeout=0.01)
        assert not first.done()
        assert cache.metrics(operator=True)["resources"]["cleanup_workers"] == 1
    finally:
        release.set()
        await write
        await first
