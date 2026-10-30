# Request lifecycle

## Identity and verified reuse

1. Resolve trusted scope and reject unsupported request shapes before embedding.
2. Canonicalize execution context. The current query stays outside the semantic
   partition but participates in the exact-request key.
3. Validate the storage manifest and embedding generation.
4. Retrieve scored, live candidates from the exact partition without recording hits.
   An empty index returns a miss without computing a query embedding.
5. Apply current admission policy, verification, and strict response decoding.
6. Recheck identity, payload integrity, expiry, and deletion under mutation coordination.
7. Account once for an accepted hit, or return a reasoned miss/bypass.

Identity includes sharing policy, tenant/user, logical namespace, operation,
provider/deployment, model, instructions, generation parameter states, source versions,
codec, policy version, and embedding fingerprint. Full SHA-256 digests are identifiers,
not encryption or authentication.

## Shared execution in v0.3

`get_or_compute` / `aget_or_compute`, query-only decorators, and supported OpenAI calls
use the shared execution module. Identical fully scoped requests can join a flight;
similarity alone never authorizes joining. Effective TTL and tags also separate flights.

After registration the cache is rechecked. Initialized async calls let the leader
perform that lookup once; followers observe its result through completion. On a miss,
the leader runs the origin and admits/stores an eligible result before publishing a
bounded replay envelope. Followers revalidate policy, identity, integrity, and freshness,
checking the stored entry directly instead of repeating vector search. Each caller
receives an independently decoded response.

Sync waiters and detachable loop-local async subscriptions share a per-instance
registry. Timeout/cancellation removes a follower without cancelling shared work.
Provider failures propagate without hidden retries. Rejected or oversized results
are not shared; followers execute their own callbacks once.

Separate manual lookup → origin → store calls and generic downstream ASGI invocations
do not receive singleflight automatically. Multiple instances/processes do not share
flights or synchronize their local FAISS indexes, even with Redis persistence.

## Failure, invalidation, and resources

Recognized runtime cache failures follow strict/default `raise` or opt-in `bypass`
policy. Bypass runs the origin without trusting a failed cache path. Invalid configuration,
incompatible generations, policy rejection, provider errors, and failed maintenance
are not silently converted into accepted hits. Partial mutation quarantines the instance.

Explicit invalidation/clear advances an instance-wide epoch and fences older in-flight
writes and shared responses, including an empty-store miss. Source changes still need
accurate corpus/document versions supplied by the application.

Provider callbacks and policy hooks execute outside storage/index locks. Async cache
work uses a bounded, tracked executor. Cancellation observed before commit prevents
late storage/accounting; an already-started commit can complete. Shutdown drains owners
before closing resources and reports incomplete cleanup explicitly. SQLite owns one
serialized connection; concurrent async close callers share one cleanup worker.

Format-2 manifests prevent incompatible vectors from entering the index. Restart and
maintenance rebuild from compatible live entries while preserving flat/HNSW settings.
Redacted events and bounded measurements distinguish cache hits from shared origin
results. See [reliability](../reliability.md) for the exact contracts and limits.
