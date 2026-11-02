"""Real Redis acceptance for the opt-in shared deployment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
import asyncio
import json
import os
import threading
import time
from uuid import uuid4

import pytest

from longcache import CacheConfig, LongCache, RequestScope
from longcache.core.errors import (
    CacheOperationError,
    SingleflightCapacityError,
    SingleflightLeaderError,
)
from tests.helpers import HashEmbedder, trusted_context


REDIS_URL = os.environ.get("LONGCACHE_TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="Real Redis service required")


def _config(cache_id, *, ttl=60, index="flat", lease=15.0, renew=5.0):
    return CacheConfig(
        backend="redis",
        deployment_mode="shared",
        redis_url=REDIS_URL,
        cache_id=cache_id,
        generation="v04",
        ttl_seconds=ttl,
        faiss_index_type=index,
        distributed_lease_seconds=lease,
        distributed_renew_interval_seconds=renew,
    )


@pytest.fixture
def pair():
    config = _config("shared-" + uuid4().hex)
    first = LongCache(config, embedder=HashEmbedder())
    second = LongCache(config, embedder=HashEmbedder())
    try:
        yield first, second
    finally:
        first.clear(operator=True)
        first.close()
        second.close()


def test_remote_write_invalidation_and_new_version(pair):
    first, second = pair
    old = trusted_context(corpus_version="one", document_versions={"doc": "one"})
    new = trusted_context(corpus_version="two", document_versions={"doc": "two"})
    assert first.set("question", {"answer": "old"}, context=old, tags=["doc"]).outcome == "STORED"
    assert second.get("question", context=old) == {"answer": "old"}
    assert first.invalidate_by_tag("doc", context=old) == 1
    assert second.get("question", context=old) is None
    assert first.set("question", {"answer": "new"}, context=new, tags=["doc"]).outcome == "STORED"
    assert second.get("question", context=new) == {"answer": "new"}
    assert second.get("question", context=old) is None


def test_empty_store_invalidation_fences_old_writer(pair):
    first, second = pair
    context = trusted_context()
    old_epoch = first.backend.epoch()
    assert second.invalidate_by_tag("doc", context=context) == 0
    decision = first.set(
        "question",
        "old",
        context=context,
        tags=["doc"],
        _expected_shared_epoch=old_epoch,
    )
    assert decision.outcome == "REJECTED"
    assert decision.reason == "invalidated"
    assert second.get("question", context=context) is None


def test_expiry_is_observed_across_instances():
    config = _config("expiry-" + uuid4().hex, ttl=1)
    first = LongCache(config, embedder=HashEmbedder())
    second = LongCache(config, embedder=HashEmbedder())
    try:
        context = trusted_context()
        assert first.set("question", "answer", context=context).outcome == "STORED"
        assert second.get("question", context=context) == "answer"
        time.sleep(1.1)
        assert second.get("question", context=context) is None
    finally:
        first.clear(operator=True)
        first.close()
        second.close()


def test_incremental_write_catchup_and_delete_rebuild(pair):
    first, second = pair
    context = trusted_context()
    assert second.get("first", context=context) is None
    assert first.set("first", "one", context=context).outcome == "STORED"
    assert second.get("first", context=context) == "one"
    assert first.set("second", "two", context=context).outcome == "STORED"
    assert second.get("second", context=context) == "two"
    assert first.invalidate(context=context) == 2
    assert second.get("first", context=context) is None
    assert second.get("second", context=context) is None


def test_local_store_cursor_does_not_skip_remote_commit(pair):
    first, second = pair
    context = trusted_context()
    assert first.set("first", "one", context=context).outcome == "STORED"
    assert first._shared_revision == first.backend.revision()
    assert second.set("remote", "two", context=context).outcome == "STORED"
    assert first.set("third", "three", context=context).outcome == "STORED"
    assert first._shared_revision < first.backend.revision()
    assert first.get("remote", context=context) == "two"
    assert first._shared_revision == first.backend.revision()


def test_trimmed_journal_rebuilds_from_authoritative_store(pair):
    first, second = pair
    context = trusted_context()
    assert second.get("question", context=context) is None
    assert first.set("question", "answer", context=context).outcome == "STORED"
    first.backend._client.xtrim(first.backend.JOURNAL, maxlen=0, approximate=False)
    assert second.get("question", context=context) == "answer"


def test_connection_pool_reconnect_revalidates_remote_invalidation(pair):
    first, second = pair
    context = trusted_context()
    assert first.set("question", "old", context=context, tags=["doc"]).outcome == "STORED"
    assert second.get("question", context=context) == "old"
    second.backend._client.connection_pool.disconnect()
    assert first.invalidate_by_tag("doc", context=context) == 1
    assert second.get("question", context=context) is None
    assert first.set("question", "new", context=context, tags=["doc"]).outcome == "STORED"
    assert second.get("question", context=context) == "new"


def test_lost_or_recreated_generation_manifest_quarantines_old_process(pair):
    first, second = pair
    context = trusted_context()
    assert first.set("question", "answer", context=context).outcome == "STORED"
    assert second.get("question", context=context) == "answer"
    key = first.backend.ROOT + ":manifest"
    original = first.backend._client.get(key)
    try:
        first.backend._client.delete(key)
        with pytest.raises(ValueError, match="changed or lost"):
            second.get("question", context=context)
        changed = json.loads(original)
        changed["incarnation"] = uuid4().hex
        first.backend._client.set(key, json.dumps(changed, sort_keys=True, separators=(",", ":")))
        with pytest.raises(ValueError, match="changed or lost"):
            second.get("question", context=context)
    finally:
        first.backend._client.set(key, original)


def test_tag_invalidation_fences_write_missing_from_its_pre_read(pair, monkeypatch):
    first, second = pair
    context = trusted_context()
    prepared, reason = first._prepare("question", context, None, {})
    assert reason is None
    tag_key = first.backend._tag_key(prepared[1], "doc")
    original = first.backend._client.sscan

    def concurrent_write(key, *args, **kwargs):
        result = original(key, *args, **kwargs)
        if key == tag_key:
            monkeypatch.setattr(first.backend._client, "sscan", original)
            assert second.set("question", "old", context=context, tags=["doc"]).outcome == "STORED"
        return result

    monkeypatch.setattr(first.backend._client, "sscan", concurrent_write)
    assert first.invalidate_by_tag("doc", context=context) == 1
    assert second.get("question", context=context) is None
    assert first.backend.sweep_expired() == 1
    assert first.backend.count(prepared[1]) == 0


def test_remote_invalidation_replay_does_not_rebuild_full_index(pair, monkeypatch):
    first, second = pair
    context = trusted_context()
    assert first.set("question", "old", context=context, tags=["doc"]).outcome == "STORED"
    assert second.get("question", context=context) == "old"
    assert first.invalidate_by_tag("doc", context=context) == 1

    def unexpected_rebuild():
        raise AssertionError("Remote invalidation must use authoritative replay guard")

    monkeypatch.setattr(second, "_rebuild_index_from_backend", unexpected_rebuild)
    assert second.get("question", context=context) is None
    assert second._shared_compaction_pending


def test_large_tag_invalidation_is_logical_then_cleaned_in_pages():
    config = _config("bulk-invalidate-" + uuid4().hex, ttl=0)
    with LongCache(config, embedder=HashEmbedder()) as cache:
        context = trusted_context()
        for number in range(600):
            assert (
                cache.set(f"question-{number}", "old", context=context, tags=["bulk"]).outcome
                == "STORED"
            )
        assert cache.invalidate_by_tag("bulk", context=context) == 600
        assert cache.get("question-0", context=context) is None
        assert cache.get("question-599", context=context) is None
        assert cache.backend._client.scard(cache.backend.ALL_IDS) == 600
        for _ in range(10):
            cache.backend.sweep_expired()
            if cache.backend._client.scard(cache.backend.ALL_IDS) == 0:
                break
        assert cache.backend._client.scard(cache.backend.ALL_IDS) == 0
        assert (
            cache.set("new question", "fresh", context=context, tags=["bulk"]).outcome == "STORED"
        )
        assert cache.get("new question", context=context) == "fresh"
        cache.clear(operator=True)


def test_shared_lease_capacity_and_expiry():
    config = CacheConfig.from_dict(
        _config("lease-cap-" + uuid4().hex).to_dict() | {"shared_max_leases": 2}
    )
    with LongCache(config, embedder=HashEmbedder()) as cache:
        cache._ensure_ready()
        key_one, token_one = cache.backend.acquire_flight("one", 1000)
        _, token_two = cache.backend.acquire_flight("two", 50)
        assert token_one and token_two
        with pytest.raises(SingleflightCapacityError):
            cache.backend.acquire_flight("three", 1000)
        time.sleep(0.06)
        key_three, token_three = cache.backend.acquire_flight("three", 1000)
        assert token_three and key_three != key_one
        assert int(token_three.rsplit(":", 1)[1]) > int(token_one.rsplit(":", 1)[1])
        cache.backend.release_flight(key_one, token_one)
        assert cache.backend._client.zcard(cache.backend.LEASES) == 1
        cache.backend.release_flight(key_three, token_three)


def test_lost_store_response_resolves_same_operation_receipt(pair, monkeypatch):
    import redis
    from longcache.backends.redis_shared import _STORE

    first, second = pair
    context = trusted_context()
    original = first.backend._client.eval
    lost = False

    def lose_first_response(script, *args):
        nonlocal lost
        result = original(script, *args)
        if script == _STORE and not lost:
            lost = True
            raise redis.exceptions.ConnectionError("response lost after Redis commit")
        return result

    monkeypatch.setattr(first.backend._client, "eval", lose_first_response)
    assert first.set("question", "answer", context=context).outcome == "STORED"
    assert lost
    assert first.backend.revision() == 1
    assert first.backend._client.scard(first.backend.ALL_IDS) == 1
    assert second.get("question", context=context) == "answer"


def test_lost_invalidation_response_returns_original_count(pair, monkeypatch):
    import redis
    from longcache.backends.redis_shared import _INVALIDATE

    first, second = pair
    context = trusted_context()
    assert first.set("question", "answer", context=context, tags=["doc"]).outcome == "STORED"
    original = first.backend._client.eval
    lost = False

    def lose_first_response(script, *args):
        nonlocal lost
        result = original(script, *args)
        if script == _INVALIDATE and not lost:
            lost = True
            raise redis.exceptions.ConnectionError("response lost after invalidation")
        return result

    monkeypatch.setattr(first.backend._client, "eval", lose_first_response)
    assert first.invalidate_by_tag("doc", context=context) == 1
    assert lost
    assert first.backend.revision() == 2
    assert first.backend.epoch() == 1
    assert second.get("question", context=context) is None


def test_shared_store_receipt_capacity_is_bounded():
    config = CacheConfig.from_dict(
        _config("receipt-cap-" + uuid4().hex).to_dict() | {"shared_max_receipts": 2}
    )
    with LongCache(config, embedder=HashEmbedder()) as cache:
        context = trusted_context()
        assert cache.set("one", "answer", context=context).outcome == "STORED"
        assert cache.set("two", "answer", context=context).outcome == "STORED"
        rejected = cache.set("three", "answer", context=context)
        assert rejected.outcome == "REJECTED" and rejected.reason == "capacity"
        assert cache.backend._client.zcard(cache.backend.RECEIPTS) == 2
        expired_receipt = cache.backend._client.zrange(cache.backend.RECEIPTS, 0, 0)[0]
        cache.backend._client.zadd(cache.backend.RECEIPTS, {expired_receipt: 0})
        assert cache.set("three", "answer", context=context).outcome == "STORED"
        assert cache.backend._client.zcard(cache.backend.RECEIPTS) == 2
        expired_receipt = cache.backend._client.zrange(cache.backend.RECEIPTS, 0, 0)[0]
        cache.backend._client.zadd(cache.backend.RECEIPTS, {expired_receipt: 0})
        cache.clear(operator=True)


def test_partition_tag_and_per_entry_metadata_limits():
    config = CacheConfig.from_dict(
        _config("metadata-cap-" + uuid4().hex).to_dict()
        | {
            "shared_max_partitions": 1,
            "shared_max_distinct_tags": 1,
            "shared_max_tags_per_entry": 1,
            "shared_max_tag_chars": 3,
        }
    )
    with LongCache(config, embedder=HashEmbedder()) as cache:
        context = trusted_context()
        other = trusted_context(scope=RequestScope("other-tenant", "other-user"))
        assert cache.set("one", "answer", context=context, tags=["one"]).outcome == "STORED"
        assert (
            cache.set("two", "answer", context=context, tags=["one", "two"]).reason
            == "tag_capacity"
        )
        assert cache.set("two", "answer", context=context, tags=["long"]).reason == "tag_capacity"
        assert cache.set("two", "answer", context=context, tags=["two"]).reason == "capacity"
        assert cache.set("two", "answer", context=other, tags=["one"]).reason == "capacity"
        _, partition, _ = cache._prepare("maintenance", context, None, {})[0]
        _, other_partition, _ = cache._prepare("maintenance", other, None, {})[0]
        with pytest.raises(CacheOperationError, match="metadata_capacity"):
            cache.backend.delete_by_tag("two", partition)
        with pytest.raises(CacheOperationError, match="metadata_capacity"):
            cache.backend.delete_namespace(other_partition)
        assert cache.invalidate_by_tag("one", context=context) == 1
        assert cache.backend._client.scard(cache.backend.NAMESPACE_SET) == 1
        assert cache.backend._client.scard(cache.backend.TAGS) == 1
        cache.clear(operator=True)


@pytest.mark.asyncio
async def test_long_sync_origin_renews_lease_for_async_follower():
    config = _config("mixed-renew-" + uuid4().hex, lease=0.3, renew=0.09)
    first = LongCache(config, embedder=HashEmbedder())
    second = LongCache(config, embedder=HashEmbedder())
    context = trusted_context()
    started = threading.Event()
    follower_calls = 0

    def slow_origin():
        started.set()
        time.sleep(0.8)
        return "first answer"

    async def follower_origin():
        nonlocal follower_calls
        follower_calls += 1
        return "duplicate answer"

    try:
        leader = asyncio.create_task(
            asyncio.to_thread(first.get_or_compute, "question", slow_origin, context=context)
        )
        for _ in range(50):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        await asyncio.sleep(0.4)
        follower = asyncio.create_task(
            second.aget_or_compute("question", follower_origin, context=context)
        )
        assert await asyncio.wait_for(leader, 5) == "first answer"
        assert await asyncio.wait_for(follower, 5) == "first answer"
        assert follower_calls == 0
    finally:
        first.clear(operator=True)
        first.close()
        second.close()


@pytest.mark.asyncio
async def test_async_leader_cancellation_releases_lease_and_notifies_follower(pair):
    first, second = pair
    context = trusted_context()
    started = asyncio.Event()
    follower_calls = 0

    async def leader_origin():
        started.set()
        await asyncio.Event().wait()

    async def follower_origin():
        nonlocal follower_calls
        follower_calls += 1
        return "duplicate"

    leader = asyncio.create_task(first.aget_or_compute("question", leader_origin, context=context))
    await asyncio.wait_for(started.wait(), 5)
    follower = asyncio.create_task(
        second.aget_or_compute("question", follower_origin, context=context)
    )
    await asyncio.sleep(0.1)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    with pytest.raises(SingleflightLeaderError):
        await asyncio.wait_for(follower, 5)
    assert follower_calls == 0
    assert first.backend._client.zcard(first.backend.LEASES) == 0


def test_new_process_rebuilds_hnsw_from_shared_generation():
    config = _config("hnsw-" + uuid4().hex, index="hnsw")
    context = trusted_context()
    first = LongCache(config, embedder=HashEmbedder())
    try:
        assert first.set("question", "answer", context=context).outcome == "STORED"
        first.close()
        with LongCache(config, embedder=HashEmbedder()) as restarted:
            assert restarted.get("question", context=context) == "answer"
            assert restarted.invalidate(context=context) == 1
        with LongCache(config, embedder=HashEmbedder()) as another:
            assert another.get("question", context=context) is None
            another.clear(operator=True)
    finally:
        if not first._closed:
            first.close()


def test_expired_owner_cannot_publish_or_release_new_owner(pair):
    first, second = pair
    context = trusted_context()
    first._ensure_ready()
    second._ensure_ready()
    key, old_token = first.backend.acquire_flight("lease-race", 30)
    assert old_token is not None
    time.sleep(0.05)
    same_key, new_token = second.backend.acquire_flight("lease-race", 5000)
    assert same_key == key and new_token is not None and new_token != old_token
    old = first.set("question", "old", context=context, _lease_key=key, _lease_token=old_token)
    assert old.outcome == "REJECTED"
    first.backend.release_flight(key, old_token)
    assert second.backend.flight_active(key)
    assert (
        second.set(
            "question", "new", context=context, _lease_key=key, _lease_token=new_token
        ).outcome
        == "STORED"
    )
    assert first.get("question", context=context) == "new"
    second.backend.release_flight(key, new_token)


@pytest.mark.asyncio
async def test_async_shared_execution_and_remote_invalidation(pair):
    first, second = pair
    context = trusted_context()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.1)
        return {"answer": "one"}

    answers = await asyncio.gather(
        *(
            cache.aget_or_compute("question", compute, context=context)
            for cache in (first, second)
            for _ in range(10)
        )
    )
    assert answers == [{"answer": "one"}] * 20
    assert calls == 1
    assert second.invalidate_by_tag("doc", context=context) == 0


@pytest.mark.asyncio
async def test_remote_invalidation_fences_paused_origin(pair):
    first, second = pair
    context = trusted_context()
    started, release = asyncio.Event(), asyncio.Event()

    async def compute():
        started.set()
        await release.wait()
        return "answer from old work"

    pending = asyncio.create_task(
        first.aget_or_compute("question", compute, context=context, tags=["doc"])
    )
    await asyncio.wait_for(started.wait(), 5)
    assert second.invalidate_by_tag("doc", context=context) == 0
    release.set()
    assert await pending == "answer from old work"
    assert second.get("question", context=context) is None


def test_uncertain_generation_refuses_remote_replay(pair):
    first, second = pair
    context = trusted_context()
    assert first.set("question", "answer", context=context).outcome == "STORED"
    assert second.get("question", context=context) == "answer"
    first.backend._client.set(first.backend.DIRTY, "1")
    try:
        with pytest.raises(ValueError, match="uncertain mutation"):
            second.get("question", context=context)
    finally:
        first.backend._client.delete(first.backend.DIRTY)


def test_local_index_failure_does_not_delete_committed_shared_entry(pair, monkeypatch):
    first, second = pair
    context = trusted_context()

    def broken_add(*_):
        raise RuntimeError("local FAISS failed")

    monkeypatch.setattr(first.index, "add", broken_add)
    with pytest.raises(CacheOperationError):
        first.set("question", "answer", context=context)
    assert second.get("question", context=context) == "answer"


def test_strict_and_bypass_policy_during_coordination_outage(monkeypatch):
    import redis

    context = trusted_context()
    for policy in ("raise", "bypass"):
        config = CacheConfig.from_dict(
            _config("outage-" + uuid4().hex).to_dict() | {"cache_error_policy": policy}
        )
        with LongCache(config, embedder=HashEmbedder()) as cache:
            calls = 0

            def compute():
                nonlocal calls
                calls += 1
                return "origin"

            def unavailable():
                raise redis.exceptions.ConnectionError("unavailable")

            monkeypatch.setattr(cache.backend, "epoch", unavailable)
            if policy == "raise":
                with pytest.raises(CacheOperationError):
                    cache.get_or_compute("question", compute, context=context)
                assert calls == 0
            else:
                assert cache.get_or_compute("question", compute, context=context) == "origin"
                assert calls == 1


def test_incompatible_shared_manifest_fails_before_reuse():
    config = _config("manifest-" + uuid4().hex)
    context = trusted_context()
    with LongCache(config, embedder=HashEmbedder()) as first:
        assert first.set("question", "answer", context=context).outcome == "STORED"
        incompatible = CacheConfig.from_dict(config.to_dict() | {"max_entries": 99})
        with pytest.raises(ValueError, match="Incompatible shared cache generation"):
            LongCache(incompatible, embedder=HashEmbedder())
        first.clear(operator=True)


@pytest.mark.asyncio
async def test_remote_follower_sees_leader_failure_without_running_origin(pair):
    first, second = pair
    context = trusted_context()
    started, release = asyncio.Event(), asyncio.Event()
    follower_calls = 0

    async def leader():
        started.set()
        await release.wait()
        raise RuntimeError("provider failed")

    async def follower():
        nonlocal follower_calls
        follower_calls += 1
        return "should not run"

    pending_leader = asyncio.create_task(first.aget_or_compute("question", leader, context=context))
    await asyncio.wait_for(started.wait(), 5)
    pending_follower = asyncio.create_task(
        second.aget_or_compute("question", follower, context=context)
    )
    await asyncio.sleep(0.1)
    release.set()
    with pytest.raises(RuntimeError, match="provider failed"):
        await pending_leader
    with pytest.raises(SingleflightLeaderError):
        await pending_follower
    assert follower_calls == 0


def _burst_worker(config, gate, result):
    import redis

    client = redis.Redis.from_url(REDIS_URL)
    context = trusted_context()
    cache = LongCache(config, embedder=HashEmbedder())

    def compute():
        client.incr(config.cache_id + ":origin_calls")
        time.sleep(0.3)
        return {"answer": "one"}

    try:
        gate.wait(10)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(cache.get_or_compute, "question", compute, context=context)
                for _ in range(10)
            ]
            result.put([future.result(timeout=10) for future in futures])
    except BaseException as exc:
        result.put((type(exc).__name__, str(exc)))
    finally:
        cache.close()


def _crashing_owner(config, started):
    context = trusted_context()
    with LongCache(config, embedder=HashEmbedder()) as cache:

        def crash():
            started.set()
            os._exit(9)

        cache.get_or_compute("question", crash, context=context)


def test_owner_crash_allows_takeover_after_lease_expiry():
    config = _config("crash-" + uuid4().hex, lease=0.3, renew=0.09)
    process_context = get_context("spawn")
    started = process_context.Event()
    process = process_context.Process(target=_crashing_owner, args=(config, started))
    try:
        process.start()
        assert started.wait(5)
        process.join(timeout=5)
        assert process.exitcode == 9
        with LongCache(config, embedder=HashEmbedder()) as cache:
            answer = cache.get_or_compute(
                "question", lambda: "recovered", context=trusted_context()
            )
            assert answer == "recovered"
            assert cache.get("question", context=trusted_context()) == "recovered"
            cache.clear(operator=True)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_twenty_requests_across_two_processes_compute_once():
    import redis

    config = _config("burst-" + uuid4().hex)
    process_context = get_context("spawn")
    gate, results = process_context.Event(), process_context.Queue()
    processes = [
        process_context.Process(target=_burst_worker, args=(config, gate, results))
        for _ in range(2)
    ]
    client = redis.Redis.from_url(REDIS_URL)
    try:
        for process in processes:
            process.start()
        gate.set()
        answers = [results.get(timeout=15) for _ in processes]
        assert answers == [[{"answer": "one"}] * 10] * 2
        assert int(client.get(config.cache_id + ":origin_calls")) == 1
    finally:
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        with LongCache(config, embedder=HashEmbedder()) as cache:
            cache.clear(operator=True)
        client.delete(config.cache_id + ":origin_calls")
