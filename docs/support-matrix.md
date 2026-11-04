# Support Matrix & API Stability

This document outlines the official platform support, runtime dependencies, and API stability classifications for LongCache **v0.5.0**.

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
| `sqlite` | Persistent (`.longcache/v2/cache.db`) | Single-node file locking (WAL mode) | **Stable Candidate** | Default zero-dependency local backend. Uses schema format 2. |
| `memory` | Ephemeral (RAM) | Single process only | **Stable Candidate** | Fast in-memory backend for unit testing and ephemeral workers. |
| `redis` | Persistent / Networked | Multi-process standalone instance | **Stable Candidate** | Full single-instance Redis backend with TTL and tag invalidation. |
| `redis_shared` | Persistent / Networked | Multi-tenant shared cluster / leases | **Experimental** | Multi-instance singleflight coordination and epoch validation. |
| `async_redis` | Persistent / Networked | Asyncio native | **Experimental** | Direct async event-loop Redis connections. |

---

## Framework Integrations

LongCache provides both active caching wrappers and explicit pass-through compatibility adapters for popular LLM orchestration frameworks:

| Integration | Class / Entry Point | Behavior | Status |
|:---|:---|:---|:---|
| **OpenAI** | `CachedOpenAI` | Inspects, scopes, caches, and verifies single-turn text completions | Experimental |
| **FastAPI / ASGI** | `LongCacheMiddleware` | Transparent HTTP request caching with header-based decision tracing | Experimental |
| **LangChain** | `CachedLLM`, `CachedChain` | Preserves LCEL interface contracts; unproven graph state passes through | Pass-Through |
| **LangGraph** | `CachedState` | Checkpointer compatibility pass-through; opaque state bypasses cache | Pass-Through |
| **LlamaIndex** | `CachedQueryEngine` | Query engine proxy pass-through; opaque retrieval bypasses cache | Pass-Through |

---

## API Stability Classification (v0.5.0)

Every public export from `longcache` is formally classified into one of three stability tiers via `longcache.PUBLIC_API_STABILITY`:

### 1. Stable Candidate (`stable-candidate`)
Core types, error hierarchies, decision objects, and top-level cache management interfaces. These interfaces adhere to strict semantic versioning; backward compatibility is strictly maintained, and breaking changes require a major version bump.

- **Cache Engine:** `LongCache`, `CacheConfig`
- **Identity & Context:** `Instruction`, `RequestContext`, `RequestScope`
- **Decisions & Contracts:** `LookupDecision`, `StoreDecision`, `PolicyDecision`, `PUBLIC_API_STABILITY`
- **Embedder Interface:** `BaseEmbedder`, `EmbeddingDescriptor`
- **Exceptions:**
  - `CacheError`, `CacheClosedError`, `CacheShutdownTimeoutError`, `CacheCapacityError`, `CacheOperationError`
  - `SingleflightError`, `SingleflightTimeoutError`, `SingleflightCapacityError`, `SingleflightLeaderError`, `SingleflightReentrancyError`, `SingleflightInvalidatedError`

### 2. Experimental (`experimental`)
Features, backends, or middleware adapters undergoing active evolution. Signatures may receive minor enhancements or signature additions in minor releases:
- `CachedOpenAI`
- `LongCacheMiddleware`
- `StreamingCacheAdapter`
- `SharedRedisCacheBackend`, `AsyncRedisCacheBackend`

### 3. Pass-Through (`pass-through`)
Compatibility shims designed for drop-in interoperability with third-party frameworks whose internal execution graphs cannot provide deterministic provenance:
- `CachedLLM`
- `CachedChain`
- `CachedQueryEngine`
- `CachedState`

---

## Static Typing & `py.typed` (PEP 561)

Starting with **v0.5.0**, LongCache is fully typed according to PEP 561 and includes the `src/longcache/py.typed` marker file.
Every public function, method, and protocol is checked under strict type checking (`mypy --strict`). Consuming applications running `mypy --strict` will receive complete type definitions, generic parameter preservation on `@cache.cached`, and IDE autocompletion for all decisions and configuration options.

