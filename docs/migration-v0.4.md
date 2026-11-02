# Moving to the v0.4 shared Redis mode

The opt-in mode uses a separate `longcache:shared-v1:<cache_id>:<generation>` keyspace. Keep the existing single-process `longcache:v2:...` generation intact. Do not relabel a populated v2 keyspace or run v0.3 processes against shared-v1 keys.

1. Record the current v0.3 configuration, model revision, sharing policy, cache ID, generation, and rollback point.
2. Create a fresh shared generation on a dedicated writable Redis authority. Configure all workers with the same cache ID, generation, embedder descriptor, and policy settings.
3. Start two workers, then check remote write visibility, expiry, and exact-context tag invalidation with a canary before adding traffic.
4. Change supported application call paths to `get_or_compute` or `aget_or_compute` when distributed request coalescing is required. Existing manual lookup and store calls remain valid but cannot coordinate the provider call between them.
5. For RAG, keep the retrieved document snapshot, corpus version, document versions, and generation callback tied together. Resolve tenant/user from trusted application identity.

For rollback, disable shared mode and use a separate v2 generation. Verify freshness of any retained v2 entries before reuse. A v0.3 worker must never be pointed at shared-v1 keys.

The shared mode is still in development. The [acceptance map](plans/v0.4.0-acceptance.md) and [execution record](plans/v0.4.0-execution.md) distinguish passing local tests from release qualification.
