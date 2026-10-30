# Reliability in v0.3

LongCache coordinates reusable requests within **one cache instance in one process**.
Python 3.11 and 3.12, memory/SQLite/synchronous Redis, and flat/HNSW indexes remain
the supported matrix. A Redis store does not synchronize FAISS indexes or flights
between instances or processes.

## Shared execution

```python
answer = cache.get_or_compute(
    query,
    lambda: call_provider(query),
    context=context,
    namespace="support",
    ttl=300,
    tags=("support-docs",),
)

answer = await cache.aget_or_compute(
    query,
    lambda: async_call_provider(query),
    context=context,
)
```

Use the same trusted `RequestContext` contract as lookup/storage. `operation` must
describe the actual computation; different callback behavior needs a different
operation identity. Additional function arguments are still unsupported by the
query-only decorator and bypass caching. Callbacks must be safe to reuse: followers
do not execute their callbacks when a result is shared.

The synchronous method accepts a synchronous zero-argument callable. The asynchronous
method accepts a zero-argument factory returning an awaitable. Use async methods on
an event-loop thread; blocking shared execution is rejected there.

Twenty overlapping identical admissible requests execute the origin once. Joining
requires exact query text, complete execution scope, effective TTL, and tags. Similar
wording alone cannot join a flight. Missing identity and unsupported shapes bypass
both cache access and shared execution.

Query-only decorators and the supported OpenAI subset use this same module. Generic
ASGI middleware retains cache behavior but does not coalesce arbitrary downstream
HTTP invocations. Opaque framework wrappers and streaming remain pass-through.

The cache is rechecked after registration. Initialized async calls share the leader's
post-join lookup rather than searching independently. Shared replay rechecks the
stored entry directly and validates admission, codec/type, identity, integrity,
verification, expiry, and invalidation state. Mutable responses are decoded separately
for each caller. A cache hit and a shared live result are measured separately.

## Waiting and failures

| Event | Behavior |
| --- | --- |
| Follower timeout | That caller receives `SingleflightTimeoutError`; shared work continues. |
| Follower cancellation | Only that caller is cancelled; the leader and other followers continue. |
| Provider failure | Original error reaches the leader; followers receive `SingleflightLeaderError` chained to it. |
| Async leader cancellation | Followers terminate with a leader failure; a later explicit request can retry. |
| Rejected/refused/oversized response | No shared replay or storage. Followers invoke their own callbacks once. |
| Capacity reached | `SingleflightCapacityError` or `CacheCapacityError`; no hidden duplicate execution. |
| Recursive same-key computation | `SingleflightReentrancyError`; no self-waiting. |
| Invalidation during generation | Older work cannot repopulate or publish a shared response. Followers receive `SingleflightInvalidatedError`. |

No automatic provider retries, negative caching, distributed leases, or forcible
thread termination are introduced. Set provider/client timeouts yourself; the
follower wait timeout does not constrain the provider's execution time.

Explicit partition/tag invalidation and clearing advance an instance-wide epoch,
even when storage is empty. This deliberately fences all older in-flight writes,
including unrelated partitions. Their original callers may receive their own origin
responses, but those results cannot be stored or shared afterward. This conservative
choice does not replace application-supplied source versions.

## Cache error policy

`cache_error_policy="raise"` is the default. Recognized runtime embedding/index/storage
failures raise `CacheOperationError` with a bounded `stage`. An origin may already
have executed when a strict write fails; retrying the whole operation can incur
another provider call.

Choose `cache_error_policy="bypass"` to continue with origin execution during a cache
outage. A failed write returns the successful origin result and can share an admitted
transient result. It does not repeat the origin. Bypass never serves an unverified
candidate or suppresses invalid configuration, incompatible formats/models, invalid
vector contracts, provider errors, cancellation, or failed maintenance.

Partial mutation or uncertain index consistency quarantines the instance. Bypass mode
continues origin execution; strict mode raises. Close and recreate the cache to
validate/rebuild its state. There is no automatic repair or circuit-breaker framework.
Incidental persistent hit-count failure does not discard an otherwise verified answer.

After ASGI has forwarded an upstream response, cache persistence failure is reported
without retracting the response or invoking the app again. Before response start,
normal strict/bypass behavior applies. Application error mapping remains the host
framework's responsibility.

## Defaults and bounds

| Configuration | Default |
| --- | --- |
| `singleflight_enabled` | `True` |
| `singleflight_wait_timeout_seconds` | `30.0` |
| `singleflight_max_inflight` | `1024` |
| `singleflight_max_waiters_per_key` | `256` |
| `cache_error_policy` | `raise` |
| `backend_timeout_seconds` | `2.0` |
| `shutdown_timeout_seconds` | `5.0` |

The internal cache executor has four workers and at most 64 queued/running jobs.
Cancellation does not free queue capacity until the work item is actually dequeued.
Flights retain at most `max_capture_bytes` (default 1 MiB) per replay envelope and
32 MiB of registered replay bytes. Exceeding a replay budget makes the result
non-shareable. Worker arguments, decoded responses handed to callers, origin results,
embedding models, and index contents have their own memory costs.

These limits are instance-wide, not per tenant. Applications needing workload fairness
should apply their own admission/rate limits. Separate instances provide separate
budgets but also separate singleflight guarantees.

Finite SQLite busy and Redis connect/socket deadlines apply per I/O, not to an entire
multi-step request. Model initialization and arbitrary synchronous hooks/functions
may take longer. Hooks must be synchronous, safe for concurrent invocation, and
deterministic over their supplied inputs. Do not mutate configuration during requests.

## Ownership and shutdown

```python
with LongCache(config, context=context) as cache:
    answer = cache.get_or_compute(query, lambda: call_provider(query))

async with LongCache(config, context=context) as cache:
    answer = await cache.aget_or_compute(query, lambda: async_call_provider(query))
```

`close(timeout=None)` and `aclose(timeout=None)` stop new work, stop the sweeper,
and drain registered operations within the configured grace period. Async callers
use `aclose`; blocking close on an event-loop thread is permitted only when no work
needs draining. Calls from the cache's own active callback/hook are rejected.

At the deadline, publication permissions are revoked, followers are released, and
owned async leaders are cancelled cooperatively. A still-running thread remains
tracked. `CacheShutdownTimeoutError` means cleanup is incomplete: the cache remains
`CLOSING`, and connections in use remain open until their owners exit. A final
resource-close failure also remains visibly `CLOSING` with an explicit error.

Repeated/concurrent close calls join the same cleanup. Cancellation of an `aclose`
caller does not abandon the owned cleanup worker. A joining caller can use a shorter
wait timeout without abandoning the original cleanup grace period. A synchronous provider that never
returns cannot be forcibly stopped; configure provider deadlines. Depending on the
application's thread ownership, such code can prevent application process exit.

SQLite uses one backend-owned connection with serialized access and closes it after
workers/sweeper drain. Redis connections belong to the cache's client. Borrowed
embedders, supplied cache instances, and supplied OpenAI clients remain caller-owned.
`CachedOpenAI.close/aclose` close only its internally created cache. Close the provider
client yourself. If middleware creates its cache, wire `await middleware.aclose()`
into the application's lifespan; no global lifespan is installed automatically.

## Logs and measurements

```python
snapshot = cache.metrics(operator=True)
scoped = cache.stats(context=context)
```

`metrics` is a process-wide, redacted in-memory snapshot requiring explicit operator
access. It does not query the backend and remains available during failure and after
closure. It includes counters, cumulative fixed-bucket latency histograms, lifecycle
state, cache jobs, active flights/followers, replay bytes, and SQLite connection count.
Redis pool client ownership is not represented as an exact active-socket count.

Counters distinguish external lookups/stores/executions, origin invocations, cache
hits, shared deliveries, admission, verification, errors, and coordination outcomes.
Internal lookup rechecks do not count as extra external requests. `shared_deliveries`
is separate from `singleflight` leader/follower registrations. Histogram estimates
are not exact benchmark percentiles. No new money/token savings claims are made.

Scoped statistics retain up to 1,024 partition records. Eviction resets a partition's
retained window; `stats_window_started_at` and `stats_retention_evictions` disclose
that limitation. Aggregate process hit/miss totals remain exact. `track_stats=False`
disables nonessential counters/histograms while lifecycle coordination still runs.

Standard logging uses `logging.getLogger("longcache")` and installs no application
handlers. Decision events are DEBUG; dependency failures use bounded, rate-limited
WARNING events. Structured fields are available in `record.longcache`. Default
events omit prompts, responses, instructions, credentials, raw exception text, identity
values, namespaces, operation/model names, and full request digests. Metric labels
come from fixed allowlists; application hook reasons collapse to a fixed category.

Logging/measurement failure cannot change a valid cache decision. Slow application
handlers can still add latency; use `logging.handlers.QueueHandler` with an
application-owned `QueueListener` when emitting detailed events. No Prometheus,
OpenTelemetry, metrics HTTP endpoint, or new schema/Pydantic caching is included.

## Upgrade and evidence

The format-2 identity/storage generation and Python minimum are unchanged. Existing
compatible v0.2 stores need no relabeling or reset. New writes exceeding the capture
limit are rejected. Concurrent identical calls can now share the same completion;
disable singleflight if independent samples are required for your cacheable workload.
Separate manual lookup → origin → store code must migrate to the shared execution
methods to receive coalescing.

Run `python examples/reliability.py` for an offline public-interface demonstration.
See [benchmarks](benchmarks.md), [execution](plans/v0.3.0-execution.md), and
[acceptance evidence](plans/v0.3.0-acceptance.md) for measured results and release gates.
