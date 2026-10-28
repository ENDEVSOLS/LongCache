"""Cross-backend, sync/async and policy-failure acceptance cases."""

import asyncio
from dataclasses import replace
import os
import threading
from uuid import uuid4
import pytest
from longcache import CacheConfig, Instruction, LongCache, PolicyDecision, RequestScope
from tests.test_embedding_contract_v2 import TinyEmbedder
from tests.test_identity_v2 import context


@pytest.fixture(
    params=[
        (backend, index) for backend in ("memory", "sqlite", "redis") for index in ("flat", "hnsw")
    ],
    ids=lambda p: "-".join(p),
)
def cache(request, tmp_path):
    backend, index = request.param
    config = CacheConfig(
        backend=backend,
        faiss_index_type=index,
        db_path=str(tmp_path / "v2.db"),
        redis_url=os.environ.get("LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0"),
        cache_id="acceptance-" + uuid4().hex,
        ttl_seconds=0,
    )
    if backend == "redis":
        import redis

        try:
            redis.Redis.from_url(config.redis_url, socket_connect_timeout=1).ping()
        except redis.RedisError:
            pytest.skip("Real Redis service is required")
    value = LongCache(config, embedder=TinyEmbedder())
    try:
        yield value
    finally:
        if not value._closed:
            value.clear(operator=True)
            value.close()


@pytest.mark.asyncio
async def test_sync_async_parity_and_scoped_maintenance(cache):
    ctx = context()
    assert (
        await cache.aset("policy", {"text": "answer"}, context=ctx, tags=["document"])
    ).outcome == "STORED"
    assert cache.get("policy", context=ctx) == {"text": "answer"}
    assert (await cache.alookup("policy", context=ctx)).outcome == "HIT"
    assert cache.entries(context=ctx)[0].hit_count == 2
    cache.inspect("policy", context=ctx)
    assert cache.entries(context=ctx)[0].hit_count == 2
    other = context(scope=RequestScope("other-tenant", "alice"))
    assert (await cache.alookup("policy", context=other)).outcome == "MISS"
    assert cache.invalidate_by_tag("document", context=other) == 0
    assert cache.invalidate_by_tag("document", context=ctx) == 1
    assert await cache.aget("policy", context=ctx) is None


@pytest.mark.asyncio
async def test_interleaved_users_never_share_answers(cache):
    alice = context()
    bob = context(scope=RequestScope("tenant", "bob"))
    await asyncio.gather(
        cache.aset("policy", "alice", context=alice), cache.aset("policy", "bob", context=bob)
    )
    results = await asyncio.gather(
        *[cache.aget("policy", context=ctx) for ctx in (alice, bob) * 10]
    )
    assert results == ["alice", "bob"] * 10
    assert cache.stats(context=alice)["hits"] == 10
    assert cache.stats(context=bob)["hits"] == 10


@pytest.mark.parametrize(
    "field,value",
    [
        ("provider", "different"),
        ("scope", RequestScope("other", "alice")),
        ("instructions", (Instruction("developer", "new rules"),)),
        ("generation", {"temperature": None}),
        ("corpus_version", "changed"),
        ("document_versions", {"source": "changed"}),
        ("codec", "openai-dict-v1"),
    ],
)
def test_context_mutations_isolate_each_backend(cache, field, value):
    ctx = context()
    cache.set("policy", "original", context=ctx)
    changed = replace(ctx, **{field: value})
    assert cache.lookup("policy", context=changed).outcome != "HIT"
    assert cache.inspect("policy", context=changed)["similarity"] is None
    assert cache.invalidate(context=changed) == 0
    assert cache.get("policy", context=ctx) == "original"


@pytest.mark.asyncio
async def test_cancelled_storage_cannot_commit_after_hook_returns(cache):
    entered, proceed, finished = threading.Event(), threading.Event(), threading.Event()

    def hook(*args):
        entered.set()
        assert proceed.wait(5)
        return PolicyDecision("accept")

    original_set = cache.set

    def observed_set(*args, **kwargs):
        try:
            return original_set(*args, **kwargs)
        finally:
            finished.set()

    cache.set = observed_set
    cache._admit = hook
    task = asyncio.create_task(cache.aset("policy", "answer", context=context()))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        proceed.set()
    assert await asyncio.to_thread(finished.wait, 5)
    # finished is signaled by the complete set operation, after its commit check.
    with cache._mutation_lock:
        assert cache.entries(operator=True) == []
    assert cache.stats(operator=True)["hits"] == 0


@pytest.mark.parametrize("kind", ["invalid", "awaitable", "abstain", "raises"])
@pytest.mark.parametrize("hook_name", ["admit", "verifier"])
@pytest.mark.asyncio
async def test_policy_hook_failures_fail_closed_in_both_call_modes(hook_name, kind):
    async def pending():
        return PolicyDecision("accept")

    def hook(*args):
        if kind == "invalid":
            return True
        if kind == "awaitable":
            return pending()
        if kind == "abstain":
            return PolicyDecision("abstain")
        raise RuntimeError("synthetic private hook failure")

    with LongCache(
        CacheConfig(backend="memory"), embedder=TinyEmbedder(), **{hook_name: hook}
    ) as cache:
        if hook_name == "admit":
            assert cache.set("policy", "answer", context=context()).outcome == "REJECTED"
            assert (await cache.aset("policy", "answer", context=context())).outcome == "REJECTED"
            assert cache.entries(operator=True) == []
        else:
            cache.set("policy", "answer", context=context())
            assert cache.lookup("policy", context=context()).outcome == "MISS"
            assert (await cache.alookup("policy", context=context())).outcome == "MISS"
            assert cache.entries(context=context())[0].hit_count == 0


@pytest.mark.asyncio
async def test_generation_mapping_is_snapshotted_across_inflight_provider():
    parameters = {"settings": {"revision": 1}}
    ctx = context(generation=parameters)
    entered, release = asyncio.Event(), asyncio.Event()
    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:

        @cache.cached(context=ctx)
        async def provider(query):
            entered.set()
            await release.wait()
            return "old revision answer"

        task = asyncio.create_task(provider("policy"))
        await asyncio.wait_for(entered.wait(), 5)
        parameters["settings"]["revision"] = 2
        release.set()
        assert await task == "old revision answer"
        original = provider.cache_context_for(ctx)
        changed = provider.cache_context_for(context(generation=parameters))
        assert cache.get("policy", context=original) == "old revision answer"
        assert cache.get("policy", context=changed) is None


def test_default_decisions_and_logs_do_not_leak_payload_or_identity(caplog):
    caplog.set_level("DEBUG", logger="longcache")
    ctx = context(
        scope=RequestScope("tenant-private-marker", "user-private-marker"),
        instructions=(Instruction("system", "instruction-private-marker"),),
    )
    with LongCache(
        CacheConfig(backend="memory"),
        embedder=TinyEmbedder(),
        verifier=lambda *args: (_ for _ in ()).throw(RuntimeError("credential-private-marker")),
    ) as cache:
        cache.set("prompt-private-marker", "answer-private-marker", context=ctx)
        result = cache.lookup("prompt-private-marker", context=ctx)
        inspection = cache.inspect("prompt-private-marker", context=ctx)
        diagnostics = caplog.text + repr(result) + repr(inspection) + repr(cache)
        for marker in (
            "tenant-private-marker",
            "user-private-marker",
            "instruction-private-marker",
            "prompt-private-marker",
            "answer-private-marker",
            "credential-private-marker",
        ):
            assert marker not in diagnostics


def test_reopened_store_preserves_scopes_and_tag_boundaries(cache):
    alice = context()
    bob = context(scope=RequestScope("tenant", "bob"))
    cache.set("policy", "alice", context=alice, tags=["source"])
    cache.set("policy", "bob", context=bob, tags=["source"])
    config = cache.config
    cache.close()
    with LongCache(config, embedder=TinyEmbedder()) as reopened:
        try:
            if config.backend == "memory":
                assert reopened.get("policy", context=alice) is None
                return
            assert reopened.get("policy", context=alice) == "alice"
            assert reopened.get("policy", context=bob) == "bob"
            assert reopened.invalidate_by_tag("source", context=alice) == 1
            assert reopened.get("policy", context=alice) is None
            assert reopened.get("policy", context=bob) == "bob"
        finally:
            reopened.clear(operator=True)
