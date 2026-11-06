# LongCache 1.x Semantic Versioning & Support Contract

This document defines the official public API stability guarantee, backward-compatibility contracts, and support policies for **LongCache 1.x**.

---

## 1. Semantic Versioning Commitment

LongCache strictly adheres to [Semantic Versioning 2.0.0](https://semver.org/):

$$\text{Format: } \mathbf{MAJOR.MINOR.PATCH}$$

* **MAJOR (e.g., 2.0.0):** Incompatible public API changes, schema format upgrades requiring offline migration, or removal of deprecated interfaces.
* **MINOR (e.g., 1.1.0, 1.2.0):** Additive, backward-compatible features (e.g., multi-turn conversation caching, telemetry exporters, new embedder presets, external vector store adapters).
* **PATCH (e.g., 1.0.1, 1.0.2):** Backward-compatible defect fixes, performance optimizations, and security patches.

---

## 2. Public Contract Boundaries (`PUBLIC_API_STABILITY`)

LongCache programmatically declares the stability tier of every exported symbol in `longcache.PUBLIC_API_STABILITY`:

### 2.1 The Frozen "Stable" Surface (`tier == "stable"`)
Every symbol marked `"stable"` is frozen under the **1.x backward-compatibility guarantee**. Their signatures, parameter kinds, default values, return annotations, and behavioral semantics will not change incompatibly across any `1.x` release:

1. **Cache Core & Config:**
   * `LongCache`, `CacheConfig`, `__version__`, `API_CONTRACT_VERSION`
2. **Request Identity & Context:**
   * `RequestScope`, `RequestContext`, `Instruction`
3. **Decisions & Storage Entries:**
   * `LookupDecision`, `StoreDecision`, `PolicyDecision`, `CacheEntry`
4. **Embedder Interface:**
   * `BaseEmbedder`, `EmbeddingDescriptor`
5. **Storage Backends:**
   * `AbstractCacheBackend`, `SQLiteCacheBackend`, `RedisCacheBackend`, `SharedRedisCacheBackend`, `SharedCoordinationBackend`, `InMemoryCacheBackend`, `get_redis_backend`
6. **Active Integrations & Middleware:**
   * `CachedOpenAI`, `get_openai_wrapper`, `LongCacheMiddleware`
7. **Error & Warning Hierarchy:**
   * `CacheError`, `CacheClosedError`, `CacheShutdownTimeoutError`, `CacheCapacityError`, `CacheOperationError`
   * `SingleflightError`, `SingleflightTimeoutError`, `SingleflightCapacityError`, `SingleflightLeaderError`, `SingleflightReentrancyError`, `SingleflightInvalidatedError`
   * `LongCacheWarning`, `LongCachePassThroughWarning`, `LongCacheExperimentalWarning`
8. **Protocols & Typed Snapshots:**
   * `AdmissionHook`, `VerificationHook`, `ContextResolver`, `ScopeResolver`, `CachedCallable`
   * `InspectionReport`, `StatsSnapshot`, `ResourceSnapshot`, `MetricsSnapshot`
   * `ScopePolicy`, `BackendName`, `DeploymentMode`, `ErrorPolicy`, `FaissIndexType`, `InstructionRole`, `ResponseCodec`, `LookupOutcome`, `StoreOutcome`, `PolicyAction`, `HealthStatus`, `Float32Vector`

### 2.2 Experimental Surface (`tier == "experimental"`)
* `AsyncRedisCacheBackend`: Standalone low-level async storage backend. While stable in wire format, it is not orchestrated through `LongCache(config)` and may receive non-breaking enhancements. Instantiation emits a `LongCacheExperimentalWarning`.

### 2.3 Pass-Through Compatibility Surface (`tier == "pass-through"`)
* `CachedLLM`, `CachedChain`, `CachedQueryEngine`, `get_llamaindex_wrapper`, `CachedState`, `StreamingCacheAdapter`, `LongCache.cached_stream`
* These adapters maintain structural interface compatibility with external frameworks (LangChain, LlamaIndex, LangGraph) whose opaque graphs cannot guarantee execution provenance.
* They **never cache or reuse responses** and emit a filterable `LongCachePassThroughWarning`.

---

## 3. Persistence Wire & Schema Guarantee (Format-2)

1. **SQLite (`.longcache/v2/cache.db`):**
   * The Format-2 schema (tables `cache_entries` and `cache_manifest`) is frozen. Any `1.x` release will read and write existing `0.2.0`–`0.6.0` and `1.0.0` databases without data loss or silent table resets.
2. **Redis (`longcache:v2:<cache_id>:<generation>:*`):**
   * The Redis keyspace, manifest payload structure, and journal serialization are frozen for single-process and multi-instance shared coordination.
3. **Partition & Exact-Key Digests (`v2:<sha256>`):**
   * The canonical JSON hashing rules governing `RequestContext.partition()` and `RequestContext.exact_key()` are frozen. Upgrading LongCache within `1.x` will never cause cache misses due to changed partition digests.

---

## 4. Static Typing & Tooling Guarantee (PEP 561)

* LongCache ships with the `py.typed` marker.
* All public exports pass `mypy --strict`.
* Consuming applications running `mypy --strict` or `pyright` will never encounter missing import type errors or unannotated public parameters.
* Typed return dictionaries (`InspectionReport`, `StatsSnapshot`, etc.) may receive additive optional keys in minor releases, but existing keys will never be removed or change types.

---

## 5. Security & Isolation Patch Policy

* Confirmed tenant-isolation leaks, cross-user privacy issues, or cache-poisoning vulnerabilities receive **immediate patch releases** (`1.0.x`).
* Patches never wait for a minor release milestone.
* In the event of a critical persistence flaw, operators will be provided with explicit `generation` rotation instructions to quarantine affected data safely without service interruption.

