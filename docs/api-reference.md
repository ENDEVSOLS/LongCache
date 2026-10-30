# Public interface

## Cache and identity

`LongCache(config=None, *, embedder=None, admit=None, verifier=None, context=None, metadata_only=False)` accepts explicit dependencies. Runtime hooks and scope resolvers are not serialized configuration.

`RequestScope(tenant_id, user_id=None)` represents identity supplied by a trusted application. `RequestContext` is immutable and snapshots nested parameter mappings. Required fields are `scope`, `operation`, `provider`, `provider_scope`, `model`, and `prompt_version`. Optional fields include ordered `instructions`, `generation`, `corpus_version`, `document_versions`, `protected_entities`, `language`, `request_shape`, and `codec`.

`Instruction(role, content)` accepts system/developer roles. Context instructions and generation options must describe the actual execution. The OpenAI adapter derives those fields from its supported request subset.

## Operations

| Method | Result and behavior |
| ------ | ------------------- |
| `get_or_compute(query, compute, *, context=None, namespace=None, ttl=None, tags=None)` | Verified reuse or one exact-scoped execution; returns the response |
| `aget_or_compute(...)` | Async equivalent; callback factory returns an awaitable |
| `metrics(operator=True)` | Bounded process-wide redacted in-memory snapshot, including after closure |
| `lookup(query, namespace=None, *, context=None)` | `LookupDecision(outcome, reason, response, similarity, entry_id, threshold)` |
| `alookup(...)` | Async decision-equivalent lookup |
| `get(...)`, `aget(...)` | Decoded response or `None`; use lookup for reasons |
| `set(query, response, namespace=None, ttl=None, tags=None, *, context=None)` | `StoreDecision(STORED or REJECTED, reason, entry_id)` |
| `aset(...)` | Async storage with the same admission contract |
| `cached(..., context=None, context_resolver=None)` | Query-only sync/async decorator; extra arguments bypass |
| `cached_stream(...)` | Generator pass-through; no cache operations |
| `inspect(query, *, context=None)` | Redacted decision dictionary; no hit accounting |
| `invalidate(..., context=...)`, `invalidate_by_tag(tag, ..., context=...)` | Delete within one exact semantic partition |
| `entries(..., context=..., operator=False)` | Owned scoped entries, or explicit trusted operator selection |
| `stats(context=..., operator=False)` | Scoped process counters and stored-entry count |
| `clear(operator=True)` | Explicitly clear the owned store |
| `close(timeout=None)` / sync context manager | Drain tracked work and close all owned resources; explicit timeout on incomplete drain |
| `aclose(timeout=None)` / async context manager | Async cleanup with the same ownership contract |

`metadata_only=True` supports counts/export/maintenance without downloading an embedder; vector lookup/storage is rejected in this mode. CLI uses this for operator commands.

## Configuration

`CacheConfig` rejects unknown dictionary keys and invalid safety settings. Key defaults:

| Field | Default |
| ----- | ------- |
| `scope_policy` | `private` (`tenant_shared` / `public` require explicit opt-in) |
| `backend` | `sqlite` (`memory` / synchronous `redis` supported) |
| `db_path` | `.longcache/v2/cache.db` |
| `cache_id`, `generation` | `default`, `default` |
| `policy_version` | `safe-reuse-v1` |
| `similarity_threshold` | `0.92` |
| `ttl_seconds`, `max_entries` | `3600`, `10000` per semantic partition |
| `verification_budget` | `8` candidates |
| `max_query_chars`, `max_capture_bytes` | `16384`, `1048576` |
| `faiss_index_type` | `flat` (`hnsw` also supported) |
| `embedder_model` | `BAAI/bge-small-en-v1.5` |
| `embedder_revision` | Resolved downloaded commit; set explicitly for deployments |
| `singleflight_enabled` | `True` |
| `singleflight_wait_timeout_seconds` | `30.0` |
| `singleflight_max_inflight`, `singleflight_max_waiters_per_key` | `1024`, `256` |
| `cache_error_policy` | `raise` (`bypass` is opt-in) |
| `backend_timeout_seconds`, `shutdown_timeout_seconds` | `2.0`, `5.0` |

Legacy model/prompt/temperature fields are insufficient to establish trusted request identity. Explicit-context calls reject conflicting legacy keywords. Calls with a fixed cache context may map compatible legacy per-call keywords into that context; use `RequestContext` for new code.

## Policy and embedding interfaces

`PolicyDecision(action, reason)` has action `accept`, `reject`, or `abstain`. Admission receives `(context, normalized_response)`; verification receives `(query, candidate_query, normalized_response, similarity, context)`. Hook exceptions/invalid return values reject reuse or storage. Mandatory checks cannot be overridden.

`BaseEmbedder` requires `dimension`, `embed(text)`, and a stable `EmbeddingDescriptor(model, revision, dimension, preprocessing, normalization)`. Single vectors and batches must be contiguous float32 arrays with finite, L2-normalized rows and the declared dimension. Empty batches must have shape `(0, dimension)` and float32 dtype; they do not bypass dtype validation. The default batch implementation supports empty batches. Calls are serialized per embedder; malformed or incompatible vectors are rejected.

See [architecture](how-it-works/architecture.md), [v0.3 migration](migration-v0.3.md), and the [support matrix](index.md#supported-integrations) for limits.

For a decorated function, `answer.cache_context_for(base_context)` returns its immutable derived operation context. Use it for inspection, exports, or invalidation of that function's entries. Fixed contexts and per-call resolver contexts use the same derivation.

## Reliability settings and errors

See [reliability](reliability.md) for singleflight/failure settings, exception classes, lifecycle limits, statistic windows, and migration behavior. New v0.3 values are also available through validated `CacheConfig.from_dict` and environment factories.
