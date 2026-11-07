# Shared Redis deployment (`stable` in v1.0.0)

Qualified in v0.6.0 and promoted to `stable` in v1.0.0, shared Redis mode connects multiple processes to one writable Redis authority. It has passed multi-process churn, fail-closed Redis crash/data-loss recovery, and bounded memory index rebuilds. The default remains `single_process`.

```python
from longcache import CacheConfig, LongCache, RequestContext, RequestScope

config = CacheConfig(
    backend="redis",
    deployment_mode="shared",
    redis_url="redis://localhost:6379/0",
    cache_id="support",
    generation="shared-v1",  # choose a fresh generation for this deployment
    embedder_revision="<immutable-model-revision>",
)
context = RequestContext(
    scope=RequestScope("trusted-tenant", "trusted-user"),
    operation="support.answer",
    provider="my-provider",
    provider_scope="my-account",
    model="my-model",
    prompt_version="support-v1",
    corpus_version="corpus-revision-1",
    document_versions={"doc-id": "document-revision-1"},
)

with LongCache(config) as cache:
    answer = cache.get_or_compute(
        "What is the return window?",
        lambda: generate_from_immutable_snapshot(),
        context=context,
        tags=("doc-id",),
    )
```

Each process has its own FAISS index. Redis stores entries, a mutation revision/journal, and request leases. Before a lookup searches its local index, it reads the Redis revision and applies new store events. Deletion and invalidation events make old vectors ineligible through Redis guards, and a background sweep compacts the local index. A journal gap triggers a full rebuild. A hit must pass an atomic Redis check of the current entry, expiry, identity metadata, and invalidation state. A disconnected process cannot safely serve its old local index.

Tag, partition, and operator invalidations advance a shared watermark before returning. Other processes reject old entries immediately; a bounded background sweep later removes their Redis references. The call counts entries seen live during a revision-checked scan. Under very high concurrent write churn, it may fail with an invalidation-conflict error after five retries; retry after the churn subsides. Shared mode caps active leases at `shared_max_leases=1024` by default and returns a capacity error when full. The Redis generation manifest binds that limit, so changing it requires a new generation.

Shared execution uses the same exact identity, effective TTL, and tags in every process. A lease has a 15-second lifetime and is renewed every five seconds while its owner runs. A healthy overlapping burst shares one origin call; after owner failure or expiry, a new call may execute the origin again. Keep provider callbacks safe to repeat and set their own timeouts. Manual lookup → generation → store and arbitrary ASGI work do not gain distributed coalescing.

`cache_error_policy="raise"` remains the default. `"bypass"` can run the origin when cache coordination is unavailable; it cannot reuse an unverified local result. An ambiguous or partially failed Redis mutation marks the generation uncertain and requires investigation or a new generation. The development implementation has not yet passed the full outage/fault matrix.

For RAG, retrieve an immutable document snapshot first. Put its exact corpus and document versions in `RequestContext`, then generate from that same snapshot. `invalidate_by_tag(tag, context=...)` affects that exact context; it does not discover or revoke every revision of a document. An application that continues sending an old version can still request an old-version answer.

Use a dedicated Redis allocation with a non-evicting policy and capacity headroom. Replica reads, Redis failover, data rollback, Redis Cluster, and automatic server-restart recovery are outside the v0.4 design. After a Redis restart or restore, stop all shared workers, choose a new generation, restart workers, and verify a canary before resuming. Local process restarts rebuild their index from Redis.

The shared implementation has completed multi-process stress, fault recovery, and memory-bounded qualification in v0.6.0. See the [v0.6 benchmarks & cost methodology](benchmarks.md), [release process & runbook](release-process.md), and [support matrix](support-matrix.md).

Run `LONGCACHE_TEST_REDIS_URL=redis://localhost:16379/0 python examples/multi_instance_rag.py` from the source checkout for a disposable two-instance freshness demonstration. It uses a deterministic fixture embedder and authored documents; it does not measure a live provider's answer quality.
