import asyncio
import threading
import pytest
from longcache.backends.memory import InMemoryCacheBackend
from longcache import CacheConfig, LongCache, PolicyDecision, RequestScope
from tests.test_identity_v2 import context
from tests.test_embedding_contract_v2 import TinyEmbedder


@pytest.fixture(params=["memory", "sqlite"])
def cache(request, tmp_path):
    with LongCache(
        CacheConfig(backend=request.param, db_path=str(tmp_path / "cache.db"), ttl_seconds=0),
        embedder=TinyEmbedder(),
    ) as value:
        yield value


def test_private_requires_trusted_context_before_embedding(cache):
    cache.embedder.embed = lambda _: pytest.fail("Missing scope reached embedding")
    assert cache.lookup("hello").reason == "missing_identity"
    assert cache.set("hello", "answer").outcome == "REJECTED"
    assert cache.entries(operator=True) == []


def test_scoped_hit_inspection_stats_and_invalidation(cache):
    alice, bob = context(), context(scope=RequestScope("tenant", "bob"))
    assert cache.set("refund policy", "answer", context=alice, tags=["policy"]).outcome == "STORED"
    assert cache.lookup("refund policy", context=bob).outcome == "MISS"
    assert cache.inspect("refund policy", context=bob)["similarity"] is None
    assert cache.invalidate_by_tag("policy", context=bob) == 0
    assert cache.stats(context=bob)["total_entries"] == 0
    assert cache.inspect("refund policy", context=alice)["outcome"] == "HIT"
    assert cache.entries(context=alice)[0].hit_count == 0
    assert cache.get("refund policy", context=alice) == "answer"
    assert cache.entries(context=alice)[0].hit_count == 1
    assert cache.stats(context=alice)["hits"] == 1
    assert cache.invalidate_by_tag("policy", context=alice) == 1
    assert cache.lookup("refund policy", context=alice).outcome == "MISS"


@pytest.mark.parametrize(
    "query",
    [
        "refund is not allowed",
        "refund within 40 days",
        "refund in Germany",
        "refund after purchase",
    ],
)
def test_semantic_mismatches_are_rejected(cache, query):
    original = {
        "refund is not allowed": "refund is allowed",
        "refund within 40 days": "refund within 30 days",
        "refund in Germany": "refund in France",
        "refund after purchase": "refund before purchase",
    }[query]
    cache.set(original, "answer", context=context())
    assert cache.lookup(query, context=context()).outcome == "MISS"
    assert cache.entries(context=context())[0].hit_count == 0


def test_rejected_nearest_can_fall_through(cache):
    cache.set("refund within 30 days", "wrong", context=context())
    cache.set("refund within 40 days", "right", context=context())
    assert cache.get("refund within 40 days", context=context()) == "right"
    assert sorted(e.hit_count for e in cache.entries(context=context())) == [0, 1]


def test_admission_failures_and_strict_serialization(cache):
    for bad in [None, object(), {"number": float("nan")}, {1: "not a string key"}]:
        assert cache.set("q", bad, context=context()).outcome == "REJECTED"
    assert cache.entries(operator=True) == []


def test_expiry_rechecked_after_verification(tmp_path):
    entered = threading.Event()
    proceed = threading.Event()

    def verifier(*args):
        entered.set()
        assert proceed.wait(5)
        return PolicyDecision("accept")

    with LongCache(
        CacheConfig(backend="memory", ttl_seconds=0), embedder=TinyEmbedder(), verifier=verifier
    ) as cache:
        cache.set("q", "answer", context=context())
        results = []
        worker = threading.Thread(
            target=lambda: results.append(cache.lookup("q", context=context()))
        )
        worker.start()
        assert entered.wait(5)
        cache.invalidate(context=context())
        proceed.set()
        worker.join(5)
        assert results[0].outcome == "MISS"
        assert cache.stats(context=context())["hits"] == 0


def test_persisted_generation_and_restart(tmp_path):
    config = CacheConfig(db_path=str(tmp_path / "v2.db"), ttl_seconds=0)
    with LongCache(config, embedder=TinyEmbedder()) as first:
        first.set("q", "answer", context=context())
    with LongCache(config, embedder=TinyEmbedder()) as second:
        assert second.get("q", context=context()) == "answer"
    other = TinyEmbedder()
    other.revision = "v2"
    with pytest.raises(ValueError, match="Incompatible"):
        LongCache(config, embedder=other)


def test_hooks_cannot_override_mandatory_rejection():
    with LongCache(
        CacheConfig(backend="memory"),
        embedder=TinyEmbedder(),
        admit=lambda *a: PolicyDecision("accept"),
    ) as cache:
        assert cache.set("q", object(), context=context()).outcome == "REJECTED"


@pytest.mark.asyncio
async def test_async_equivalence_and_cancellation():
    entered = threading.Event()
    proceed = threading.Event()

    def verifier(*a):
        entered.set()
        proceed.wait(5)
        return PolicyDecision("accept")

    with LongCache(
        CacheConfig(backend="memory", ttl_seconds=0), embedder=TinyEmbedder(), verifier=verifier
    ) as cache:
        await cache.aset("q", "answer", context=context())
        task = asyncio.create_task(cache.alookup("q", context=context()))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        proceed.set()
        await asyncio.sleep(0.05)
        assert cache.stats(context=context())["hits"] == 0
        assert await cache.aget("q", context=context()) == "answer"


def test_decorator_and_stream_passthrough(cache):
    calls = []

    @cache.cached(context=context())
    def answer(query):
        calls.append(query)
        return "answer"

    assert answer("q") == answer("q") == "answer"
    assert len(calls) == 1

    @cache.cached_stream()
    def stream(query):
        calls.append(query)
        yield "chunk"

    assert list(stream("q")) == ["chunk"]
    assert list(stream("q")) == ["chunk"]
    assert len(calls) == 3


def test_valid_json_payload_corruption_is_not_replayed(cache):
    stored = cache.set("q", "approved", context=context())
    if isinstance(cache.backend, InMemoryCacheBackend):
        cache.backend.retrieve(stored.entry_id).response = "corrupted"
    else:
        cache.backend._get_connection().execute(
            "UPDATE cache_entries SET response=? WHERE id=?", ('"corrupted"', stored.entry_id)
        )
        cache.backend._get_connection().commit()
    assert cache.lookup("q", context=context()).outcome == "MISS"
    assert cache.stats(context=context())["hits"] == 0


def test_two_decorated_operations_cannot_share_an_answer(cache):
    @cache.cached(context=context())
    def first(query):
        return "first"

    @cache.cached(context=context())
    def second(query):
        return "second"

    assert first("q") == "first"
    assert second("q") == "second"
    assert first("q") == "first"


@pytest.mark.parametrize(
    "policy,same", [("private", False), ("tenant_shared", True), ("public", True)]
)
def test_sharing_requires_explicit_policy(policy, same):
    with LongCache(
        CacheConfig(backend="memory", scope_policy=policy), embedder=TinyEmbedder()
    ) as cache:
        cache.set("q", "answer", context=context())
        other = context(scope=RequestScope("tenant", "bob"))
        assert (cache.lookup("q", context=other).outcome == "HIT") == same


def test_hnsw_verification_budget_and_fallback():
    with LongCache(
        CacheConfig(backend="memory", faiss_index_type="hnsw", verification_budget=2),
        embedder=TinyEmbedder(),
    ) as cache:
        cache.set("return in 30 days", "wrong", context=context())
        cache.set("return in 40 days", "right", context=context())
        assert cache.get("return in 40 days", context=context()) == "right"


def test_application_verifier_failure_cannot_count_as_hit():
    def verifier(*args):
        raise RuntimeError("secret exception")

    with LongCache(
        CacheConfig(backend="memory"), embedder=TinyEmbedder(), verifier=verifier
    ) as cache:
        cache.set("q", "answer", context=context())
        decision = cache.lookup("q", context=context())
        assert decision.reason == "hook_failed"
        assert "secret" not in repr(decision)
        assert cache.stats(context=context())["hits"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), object()])
def test_malformed_persisted_payload_becomes_miss(invalid):
    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:
        stored = cache.set("q", "approved", context=context())
        cache.backend.retrieve(stored.entry_id).response = invalid
        result = cache.lookup("q", context=context())
        assert result.outcome == "MISS"
        assert result.reason == "codec_invalid"


def test_verification_budget_exhaustion_is_not_a_hit():
    with LongCache(
        CacheConfig(backend="memory", verification_budget=2), embedder=TinyEmbedder()
    ) as cache:
        for days in (30, 40, 50):
            cache.set(f"return in {days} days", "answer", context=context())
        result = cache.lookup("return in 60 days", context=context())
        assert result.outcome == "MISS"
        assert result.reason == "verification_budget_exhausted"
        assert cache.stats(context=context())["hits"] == 0


def test_admission_hook_failure_rejects_storage():
    def broken(*args):
        raise RuntimeError("private provider metadata")

    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder(), admit=broken) as cache:
        decision = cache.set("q", "complete answer", context=context())
        assert decision.outcome == "REJECTED"
        assert decision.reason == "hook_failed"
        assert not cache.entries(operator=True)


def test_invalid_unicode_never_reaches_embedding_or_storage(cache):
    assert cache.lookup("bad\ud800", context=context()).outcome == "BYPASS"
    assert cache.set("q", "bad\ud800", context=context()).outcome == "REJECTED"


def test_decorator_maintenance_uses_the_same_derived_context(cache):
    calls = []

    @cache.cached(context_resolver=lambda query: context(), tags=["source"])
    def first(query):
        calls.append(query)
        return "first"

    @cache.cached(context_resolver=lambda query: context())
    def second(query):
        return "second"

    assert first("q") == "first"
    assert second("q") == "second"
    derived = first.cache_context_for(context())
    assert cache.inspect("q", context=derived)["outcome"] == "HIT"
    assert cache.invalidate_by_tag("source", context=derived) == 1
    assert first("q") == "first"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_decorator_exposes_maintenance_context():
    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:

        @cache.cached(context=context())
        async def answer(query):
            return "answer"

        assert await answer("q") == "answer"
        assert cache.invalidate(context=answer.cache_context_for(context())) == 1
