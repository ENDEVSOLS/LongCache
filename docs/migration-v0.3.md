# Upgrade and rollback: v0.2 to v0.3

v0.3.0 is an unreleased development version. Installing this branch is different
from installing whichever version is currently published on the package index.
See [installation](getting-started/installation.md).

## Storage and Python compatibility

Compatible v0.2 **format-2 stores need no reset or relabeling**. Keep the same SQLite
path or Redis `cache_id`/`generation`, embedding descriptor, codec, policy version, and
execution context. The default path remains `.longcache/v2/cache.db`; the `v2` is the
storage-format version, not the current package version. Python 3.11 and 3.12 remain
the tested interpreters; the Python minimum remains 3.11.

Preserve a consistent backup and the previous environment before rollout. Actual
v0.2 write → v0.3 read/write → v0.2 read tests passed for SQLite/Redis with flat/HNSW.
To roll back, stop the newer writers before reopening the preserved compatible store
with v0.2. This verifies persistence compatibility, not concurrent multi-instance use.

If starting from v0.1.1 or another under-scoped legacy store, follow the separate
[legacy migration](migration-v0.2.md). Those entries cannot be promoted by changing a
version label; use a fresh path/generation as documented there. Changing embedding
models/revisions or preprocessing also requires a compatible new generation.

## Request execution changes

- Query-only decorators and supported OpenAI calls now share identical concurrent
  executions. Responses can contain the same generated completion rather than
  independently sampled results. Set `singleflight_enabled=False` to let concurrent
  misses execute independently. Ordinary cache hits still reuse stored responses.
- Separate manual lookup → call → store code remains independent. Use
  `get_or_compute(query, compute, context=...)` or its async equivalent to coalesce it.
- Exact query, complete scope/context, effective TTL, and tags determine flight joining.
  Missing identity and unsupported shapes still bypass both cache and flights.
- Blocking shared execution is rejected on an event-loop thread. Use
  `await aget_or_compute(...)` or move a synchronous caller to an application thread.
- A follower timeout/cancellation does not cancel the leader. A leader failure releases
  followers without automatic retries. Rejected/oversized responses are not shared.
- Explicit invalidation fences older in-flight writes/results, even when the store
  was empty. Different cache instances/processes still do not coordinate flights.

## Failure policy, resources, and measurements

Strict `cache_error_policy="raise"` remains the default. Opt-in `"bypass"` permits
origin execution during recognized runtime cache outages, with no unverified reuse.
A strict store failure can happen after the origin executed; blindly retrying the
whole operation can incur another provider call. Maintenance failures remain explicit.

Backend I/O defaults to a 2-second per-I/O timeout, follower waits to 30 seconds,
and shutdown grace to 5 seconds. These are different deadlines; set finite provider
timeouts yourself. Coordination capacity errors never become hidden duplicate calls.
See [defaults and bounds](reliability.md#defaults-and-bounds) before sizing a deployment.

Use `aclose()` / `async with` for async applications. SQLite now owns one serialized
connection, and running/queued cache work stays tracked through cancellation. A
shutdown timeout means cleanup may still be pending; do not treat it as successful
resource release. Borrowed OpenAI clients, embedders, and supplied caches remain
caller-owned. Middleware-created caches need explicit application-lifespan cleanup.

`metrics(operator=True)` exposes bounded in-memory process measurements, including
after closure. Scoped `stats()` retention is capped at 1,024 partitions; window/reset
metadata discloses eviction, while aggregate process totals remain exact. Cost fields
remain unmeasured estimates. New writes over `max_capture_bytes` are rejected.

## Verify before release

The recorded local matrix passed 457 tests per supported interpreter with zero skips.
It does not replace full controlled-host performance qualification, hosted CI,
domain-representative evaluation, or release-candidate feedback. Keep v0.3.0 unreleased
until the [release checklist](plans/v0.3.0-acceptance.md#release-readiness-checklist) passes.

Prometheus, OpenTelemetry, new schema/Pydantic caching, distributed coordination, and
generic HTTP invocation coalescing remain deferred.
