"""Multi-process Redis qualification stress and memory bound suite for v0.6.0."""

from __future__ import annotations

from multiprocessing import get_context
import os
from pathlib import Path
import time
import tracemalloc
from typing import Any
from uuid import uuid4

import pytest
import redis

from longcache import CacheConfig, LongCache, RequestScope
from longcache.core.errors import CacheOperationError
from longcache.core.types import ErrorPolicy
from tests.helpers import HashEmbedder, trusted_context

REDIS_URL = os.environ.get("LONGCACHE_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="Real Redis service required")


def _config(
    cache_id: str,
    *,
    ttl: int = 60,
    lease: float = 2.0,
    renew: float = 0.5,
    error_policy: ErrorPolicy = "raise",
) -> CacheConfig:
    return CacheConfig(
        backend="redis",
        deployment_mode="shared",
        redis_url=REDIS_URL or "redis://localhost:6379/0",
        cache_id=cache_id,
        generation="v06",
        ttl_seconds=ttl,
        distributed_lease_seconds=lease,
        distributed_renew_interval_seconds=renew,
        cache_error_policy=error_policy,
        max_entries=200,
        shared_max_entries=500,
    )


def _churn_worker(
    config: CacheConfig, worker_id: int, barrier: Any, error_queue: Any, iterations: int
) -> None:
    try:
        cache = LongCache(config, embedder=HashEmbedder())
        context = trusted_context(
            scope=RequestScope(f"tenant-{worker_id % 2}", f"user-{worker_id}")
        )
        payload_32k = {"data": "x" * 32768, "worker": worker_id}

        barrier.wait(timeout=10)
        for i in range(iterations):
            key = f"key-{worker_id}-{i % 20}"
            tag = f"tag-{i % 5}"
            # 1. Set
            decision = cache.set(key, payload_32k, context=context, tags=[tag])
            assert decision.outcome in ("STORED", "REJECTED")

            # 2. Get
            val = cache.get(key, context=context)
            if val is not None:
                assert isinstance(val, dict)
                assert val.get("data") == payload_32k["data"]

            # 3. Intermittent delete & tag invalidation
            if i % 7 == 0:
                cache.invalidate_by_tag(tag, context=context)

        cache.close()
    except Exception as exc:
        error_queue.put(f"Worker {worker_id} failed: {type(exc).__name__}: {exc}")


def test_multiprocess_sustained_write_delete_invalidate_churn_memory_bounds() -> None:
    cache_id = "churn-" + uuid4().hex
    config = _config(cache_id, ttl=120)
    process_context = get_context("spawn")
    barrier = process_context.Barrier(3)
    error_queue = process_context.Queue()

    workers = [
        process_context.Process(
            target=_churn_worker, args=(config, w_id, barrier, error_queue, 100)
        )
        for w_id in range(3)
    ]
    try:
        for w in workers:
            w.start()

        for w in workers:
            w.join(timeout=30)
            assert not w.is_alive()
            assert w.exitcode == 0

        # Check for worker errors
        errors = []
        while not error_queue.empty():
            errors.append(error_queue.get_nowait())
        assert not errors, f"Worker errors: {errors}"

        # Settle and verify memory & index bounds
        with LongCache(config, embedder=HashEmbedder()) as cache:
            # Rebuild / sync check
            cache._synchronize_shared()
            # Raw index size must remain bounded within 1.25x of shared capacity
            assert cache.index.raw_total <= config.shared_max_entries * 1.5
            cache.clear(operator=True)
    finally:
        for w in workers:
            if w.is_alive():
                w.terminate()
                w.join(timeout=5)


def _crashing_lease_leader(config: CacheConfig, ready_event: Any) -> None:
    cache = LongCache(config, embedder=HashEmbedder())

    def slow_origin() -> str:
        ready_event.set()
        time.sleep(10)  # Will be killed during computation
        return "leader-never"

    try:
        cache.get_or_compute("slow-lease-key", slow_origin, context=trusted_context())
    finally:
        cache.close()


def test_multiprocess_singleflight_lease_churn_under_worker_kills() -> None:
    cache_id = "kill-lease-" + uuid4().hex
    # Short lease duration: 1.0s so follower can take over quickly
    config = _config(cache_id, lease=1.0, renew=0.3)
    process_context = get_context("spawn")
    ready_event = process_context.Event()

    leader = process_context.Process(target=_crashing_lease_leader, args=(config, ready_event))
    try:
        leader.start()
        assert ready_event.wait(timeout=5)

        # Leader is inside slow origin holding lease; kill it with SIGKILL
        leader.kill()
        leader.join(timeout=5)

        # Now a new worker must take over once lease expires
        time.sleep(1.2)  # Wait for 1.0s lease to expire in Redis
        with LongCache(config, embedder=HashEmbedder()) as cache:
            result = cache.get_or_compute(
                "slow-lease-key", lambda: "follower-recovered", context=trusted_context()
            )
            assert result == "follower-recovered"
            assert cache.get("slow-lease-key", context=trusted_context()) == "follower-recovered"
            cache.clear(operator=True)
    finally:
        if leader.is_alive():
            leader.kill()
            leader.join(timeout=5)


def test_multiprocess_redis_restart_data_loss_quarantines_workers() -> None:
    cache_id = "restart-" + uuid4().hex
    config = _config(cache_id)
    client = redis.Redis.from_url(REDIS_URL or "redis://localhost:6379/0")

    with (
        LongCache(config, embedder=HashEmbedder()) as cache1,
        LongCache(config, embedder=HashEmbedder()) as cache2,
    ):
        context = trusted_context()
        assert cache1.set("q1", "ans1", context=context).outcome == "STORED"
        assert cache2.get("q1", context=context) == "ans1"

        # Simulate Redis volatile restart / total data loss in this generation by deleting manifest
        manifest_key = f"longcache:shared-v1:{config.cache_id}:{config.generation}:manifest"
        client.delete(manifest_key)

        # Both workers must immediately fail closed and quarantine rather than serving old local FAISS hits
        with pytest.raises((CacheOperationError, ValueError)):
            cache2.lookup("q1", context=context)

        with pytest.raises((CacheOperationError, ValueError)):
            cache1.set("q2", "ans2", context=context)


def test_multiprocess_transient_outage_and_reconnect_cycles_raise_and_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = trusted_context()

    # Cycle 5 times through simulated connection outage / restore
    for policy in ("raise", "bypass"):
        cache_id = f"outage-cycles-{policy}-{uuid4().hex}"
        config = _config(cache_id, error_policy=policy)
        with LongCache(config, embedder=HashEmbedder()) as cache:
            # Baseline store
            assert cache.set("q-init", "ans-init", context=context).outcome == "STORED"
            assert cache.get("q-init", context=context) == "ans-init"

            real_epoch = cache.backend.epoch

            for cycle in range(5):
                # 1. Inject failure
                def failing_epoch() -> int:
                    raise redis.exceptions.ConnectionError(f"Simulated network drop cycle {cycle}")

                monkeypatch.setattr(cache.backend, "epoch", failing_epoch)

                compute_called = False

                def compute() -> str:
                    nonlocal compute_called
                    compute_called = True
                    return f"origin-{cycle}"

                if policy == "raise":
                    with pytest.raises(CacheOperationError):
                        cache.get_or_compute(f"q-cycle-{cycle}", compute, context=context)
                    assert not compute_called
                else:
                    # Under bypass, should invoke origin directly
                    res = cache.get_or_compute(f"q-cycle-{cycle}", compute, context=context)
                    assert res == f"origin-{cycle}"
                    assert compute_called

                # 2. Restore connectivity
                monkeypatch.setattr(cache.backend, "epoch", real_epoch)

                # Post-restore operation succeeds and local FAISS was not corrupted
                assert cache.get("q-init", context=context) == "ans-init"

            cache.clear(operator=True)


def test_snapshot_rebuild_does_not_load_response_payloads_tracemalloc(tmp_path: Path) -> None:
    # 1. SQLite backend test
    db_path = str(tmp_path / "large_payload_rebuild.db")
    config = CacheConfig(backend="sqlite", db_path=db_path)
    context = trusted_context()

    # Populate 300 entries with 64 KiB responses (~19.2 MiB of payload data)
    payload_64k = {"blob": "Y" * 65536}
    with LongCache(config, embedder=HashEmbedder()) as cache:
        for i in range(300):
            cache.set(f"query-{i}", payload_64k, context=context)

        # Rebuild index and measure peak memory with tracemalloc
        tracemalloc.start()
        before_mem = tracemalloc.get_traced_memory()[0]

        cache._rebuild_index_from_backend()

        peak_mem = tracemalloc.get_traced_memory()[1] - before_mem
        tracemalloc.stop()

        # Peak memory during vector-only rebuild should be far below the 19.2 MiB payload size (< 5 MiB)
        assert peak_mem < 5 * 1024 * 1024, (
            f"Peak memory during rebuild too high: {peak_mem / (1024 * 1024):.2f} MiB"
        )
