# Support Matrix & API Stability

This document outlines the official platform support, runtime dependencies, and API stability classifications for LongCache **v1.0.0**.

---

## Python & Platform Support

LongCache runs on standard Python environments with no external C compilation required at install time (using prebuilt wheels for `numpy` and `faiss-cpu`).

| Environment | Version / Platform | Support Tier | Notes |
|:---|:---|:---|:---|
| **Python** | 3.11 | Tier 1 (CI Tested) | Baseline supported Python runtime. |
| **Python** | 3.12 | Tier 1 (CI Tested) | Primary development and documentation build runtime. |
| **Python** | 3.13 | Tier 2 (Supported) | Compatible with pure-python core and modern wheels. |
| **Linux** | Ubuntu 22.04+, Debian 12+, RHEL 9+ (x86_64, aarch64) | Tier 1 (Primary) | Full multi-process, SQLite, and Redis support. |
| **macOS** | macOS 13+ (Apple Silicon M-series, Intel) | Tier 1 (Supported) | Accelerated FAISS vector indexing. |
| **Windows** | Windows 10/11, Server 2022 | Tier 2 (Supported) | Embedded SQLite and Memory backends; Redis via external host or WSL. |

---

## Storage Backends

| Backend Identifier | Persistence | Multi-Process / Multi-Worker | Production Readiness | Notes |
|:---|:---|:---|:---|:---|
| `sqlite` | Persistent (`.longcache/v2/cache.db`) | Single-node file locking (WAL mode) | **Stable** | Default zero-dependency local backend. Uses schema format 2. |
| `memory` | Ephemeral (RAM) | Single process only | **Stable** | Fast in-memory backend for unit testing and ephemeral workers. |
| `redis` | Persistent / Networked | Multi-process standalone instance | **Stable** | Full single-instance Redis backend with TTL and tag invalidation. |
| `redis_shared` | Persistent / Networked | Multi-tenant shared cluster / leases | **Stable** | Multi-instance singleflight coordination and epoch validation (`SharedRedisCacheBackend`, `SharedCoordinationBackend`). |
| `async_redis` | Persistent / Networked | Asyncio native | **Experimental** | Direct async event-loop Redis connections (`AsyncRedisCacheBackend`). |

---

## Framework Integrations

LongCache provides both active caching wrappers and explicit pass-through compatibility adapters for popular LLM orchestration frameworks:

| Integration | Class / Entry Point | Behavior | Status |
|:---|:---|:---|:---|
| **OpenAI** | `CachedOpenAI`, `get_openai_wrapper` | Inspects, scopes, caches, and verifies single-turn text completions | **Stable** |
| **FastAPI / ASGI** | `LongCacheMiddleware` | Transparent HTTP request caching with header-based decision tracing | **Stable** |
| **LangChain** | `CachedLLM`, `CachedChain` | Preserves LCEL interface contracts; unproven graph state passes through with `LongCachePassThroughWarning` | **Pass-Through** |
| **LangGraph** | `CachedState` | Checkpointer compatibility pass-through; opaque state bypasses cache with `LongCachePassThroughWarning` | **Pass-Through** |
| **LlamaIndex** | `CachedQueryEngine`, `get_llamaindex_wrapper` | Query engine proxy pass-through; opaque retrieval bypasses cache with `LongCachePassThroughWarning` | **Pass-Through** |

---

## API Stability Classification (v1.0.0)

Every public export from `longcache` is formally classified into one of three stability tiers via `longcache.PUBLIC_API_STABILITY`:

### 1. Stable (`stable`)
Core types, error and warning hierarchies, decision objects, storage backends, and qualified caching interfaces. These interfaces adhere to strict semantic versioning under the `1.x` contract; backward compatibility is strictly maintained, and breaking changes require a major version bump.

- **Cache Engine:** `LongCache`, `CacheConfig`, `__version__`, `API_CONTRACT_VERSION`
- **Identity & Context:** `Instruction`, `RequestContext`, `RequestScope`
- **Decisions & Contracts:** `LookupDecision`, `StoreDecision`, `PolicyDecision`, `CacheEntry`, `PUBLIC_API_STABILITY`, `StabilityTier`
- **Embedder Interface:** `BaseEmbedder`, `EmbeddingDescriptor`
- **Error & Warning Hierarchy:**
  - `CacheError`, `CacheClosedError`, `CacheShutdownTimeoutError`, `CacheCapacityError`, `CacheOperationError`
  - `SingleflightError`, `SingleflightTimeoutError`, `SingleflightCapacityError`, `SingleflightLeaderError`, `SingleflightReentrancyError`, `SingleflightInvalidatedError`
  - `LongCacheWarning`, `LongCachePassThroughWarning`, `LongCacheExperimentalWarning`
- **Protocols & Snapshots:** `AdmissionHook`, `VerificationHook`, `ContextResolver`, `ScopeResolver`, `CachedCallable`, `InspectionReport`, `StatsSnapshot`, `ResourceSnapshot`, `MetricsSnapshot`, `ScopePolicy`, `BackendName`, `DeploymentMode`, `ErrorPolicy`, `FaissIndexType`, `InstructionRole`, `ResponseCodec`, `LookupOutcome`, `StoreOutcome`, `PolicyAction`, `HealthStatus`, `Float32Vector`
- **Storage Backends:** `AbstractCacheBackend`, `InMemoryCacheBackend`, `SQLiteCacheBackend`, `RedisCacheBackend`, `SharedRedisCacheBackend`, `SharedCoordinationBackend`, `get_redis_backend`
- **Integrations & Middleware:** `CachedOpenAI`, `get_openai_wrapper`, `LongCacheMiddleware`

### 2. Experimental (`experimental`)
Features or standalone backends undergoing active evolution. Signatures may receive enhancements or adjustments in minor releases:
- `AsyncRedisCacheBackend`

### 3. Pass-Through (`pass-through`)
Compatibility shims designed for drop-in interoperability with third-party frameworks whose internal execution graphs cannot provide deterministic provenance. In LongCache 1.0, instantiating or invoking these adapters emits a filterable `LongCachePassThroughWarning`:
- `CachedLLM`
- `CachedChain`
- `CachedQueryEngine`
- `get_llamaindex_wrapper`
- `CachedState`
- `StreamingCacheAdapter`
- `LongCache.cached_stream`

---

## Static Typing & `py.typed` (PEP 561)

Starting with **v0.5.0**, LongCache is fully typed according to PEP 561 and includes the `src/longcache/py.typed` marker file.
Every public function, method, and protocol is checked under strict type checking (`mypy --strict`). Consuming applications running `mypy --strict` will receive complete type definitions, generic parameter preservation on `@cache.cached`, and IDE autocompletion for all decisions and configuration options.

