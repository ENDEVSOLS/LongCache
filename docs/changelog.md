# Changelog

All notable changes are recorded here using [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - Unreleased

### Added

- Scoped sync/async `get_or_compute` methods and singleflight for supported decorators/OpenAI calls, with bounded flights, followers, replay bytes, and worker queues.
- Explicit follower timeouts, leader error/cancellation outcomes, reentrancy protection, and in-process invalidation fences.
- Strict (`raise`, default) and opt-in `bypass` runtime cache-error policies. Partial mutation quarantines the instance without retrying the origin.
- Async close/context managers, tracked cancellation, bounded shutdown, and redacted decision logs plus bounded process metrics.
- Repeatable paired reliability benchmarks, scheduled-arrival workloads, measurement-overhead checks, and installed-wheel concurrency smoke checks.

### Changed

- SQLite owns one serialized connection; worker cancellation cannot leak thread-local connections or permit late writes. Redis/SQLite I/O deadlines are explicit.
- Concurrent identical cacheable calls may share a completion. Disable singleflight when independent origin samples are needed; manual lookup/store pairs remain independent.
- Blocking shared execution is rejected on an event-loop thread. Provider clients and supplied caches remain caller-owned.
- Scoped statistics have bounded retained windows with reset metadata; aggregate process counts stay exact. Oversized new responses are not admitted.
- ASGI preserves an already-sent response if optional cache persistence fails afterward; generic HTTP invocation coalescing remains deferred.

### Compatibility and validation

- Python minimum and format-2 identity/storage remain unchanged. Compatible v0.2 stores require no reset; streaming, opaque framework paths and standalone async Redis retain their support limits.
- Prometheus, OpenTelemetry, and new JSON-schema/Pydantic caching are deferred.
- Local implementation evidence and remaining controlled-performance/hosted-CI/candidate gates are recorded in `docs/plans/v0.3.0-execution.md`. No push, tag, or publication is performed by this implementation.

## [0.2.0] - Unreleased

### Added

- Immutable `RequestScope`, `RequestContext`, and ordered instructions, with canonical full-length identity digests across lookup, writes, maintenance, and restart.
- Explicit private, tenant-shared, and public sharing policies. Missing private identity bypasses cache access before embedding.
- Typed lookup/store decisions, same-scope similarity scores, redacted inspection, and bounded ASGI decision headers.
- Mandatory response admission, strict JSON/OpenAI codecs, optional synchronous admission/verifier hooks, and conservative English mismatch checks.
- Injected `BaseEmbedder` support with model/revision/preprocessing descriptors, vector validation, and immutable storage-generation manifests.
- Format-2 storage with isolated Redis keys and a new SQLite path; payload integrity checks and cold-cache migration/rollback guidance.
- A frozen 4,800-case synthetic evaluation, pinned-model provenance, and flat/HNSW 1K/10K warm-hit comparisons against the committed v0.1.1 baseline.

### Changed

- Python minimum is now 3.11; supported CI interpreters are 3.11 and 3.12. v0.1.1 retains Python 3.10 support.
- `set`/`aset` return `StoreDecision`. `lookup`/`alookup` expose decisions; `get`/`aget` remain response-or-None conveniences.
- Default SQLite path is `.longcache/v2/cache.db`; legacy files are refused without modification. Existing under-scoped entries are not promoted.
- Scoped inspection and maintenance replace namespace-only assumptions. Raw exports and whole-store clearing require explicit scope/operator access.
- Generic LangChain/LangGraph/LlamaIndex wrappers and all streaming helpers are compatibility pass-through until their identity and replay contracts are proven.
- Query-only decorators bind a qualified function operation; additional arguments bypass. Unknown configuration keys raise.
- Unmeasured cost fields remain zero; process counters reset on restart while entry hit counts persist. Manual hit/miss accounting methods were removed.

### Fixed

- Unsafe ASGI responses with repeated cache directives, non-JSON media types, or observed disconnects are never admitted.
- Empty embedding batches enforce the same float32/contiguous contract as populated batches.
- SQLite schema initialization is transactional, including process termination; incompatible partial schemas and missing manifests in populated SQLite/Redis stores fail closed.

### Validation

- Full local Python 3.11/3.12 runs include real Redis and cached BGE weights. See the [execution record](plans/v0.2.0-execution.md) for exact evidence and remaining external release gates.
- Synthetic results are template-expanded and do not establish correctness on unrelated workloads. End-to-end timing is reported separately from precomputed-vector scenario timing.

## [0.1.1] - 2026-10-27

### Fixed

- Similarity lookup now considers other eligible entries when the closest vector is in another namespace, expired, or missing.
- Rebuilds restore all live namespaces after restart and invalidation, and preserve configured HNSW indexes.
- Cache inspection no longer records a hit. In-memory entry hit counts now increase once per served hit.
- The documented OpenAI `chat.completions.create` path works for supported sync and async clients. Supported requests are scoped by model, instructions, and temperature, and SDK response objects can survive a SQLite restart.
- Additional function arguments bypass decorator caching until those arguments can be represented in a stable cache identity.
- The ASGI middleware bypasses unsupported chat shapes and requests bearing Authorization, Cookie, or X-API-Key headers.
- Cache instances can be closed explicitly or used as context managers to stop background work.
- The package, CLI, and wheel report the same version.

### Compatibility and migration

- The OpenAI and ASGI integrations use new internal namespaces. Entries written by the old under-scoped integrations are not reused; the first eligible request after upgrade is a miss.
- Unsupported `backend="async_redis"` through `LongCache` now fails immediately. `AsyncRedisCacheBackend` remains available as a standalone backend while the full orchestration contract is developed.
- Redis stores responses but does not synchronize in-memory FAISS indexes between processes. Use one LongCache process per Redis-backed cache until multi-instance consistency is implemented.
- Python 3.10 remains supported in 0.1.1. The planned minimum of Python 3.11 is scheduled for 0.2.0.

### Validation

- Run the unit and integration suite with `uv run pytest -q`; Redis integration tests require a reachable service at `LONGCACHE_TEST_REDIS_URL`.
- Run `uv run ruff check src tests` and `uv build` from the package directory.
- Run real-model integration tests explicitly with `LONGCACHE_TEST_MODEL=1 uv run pytest tests/test_embedder.py -q`; these may download model weights.

## [0.1.0] - 2026-10-26

### Added

- **Core Semantic Cache Engine:**
  - Fast local embeddings using `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors) running fully offline.
  - FAISS index integration with support for exact flat cosine similarity and scalable HNSW indexing.
  - Transparent sync and async decorator `@cache.cached` with parameter hashing, TTL expiry, and namespace scoping.
  - Composite cache keys to prevent cross-model poisoning (scoped by model, system prompt, temperature, and query).
- **Pluggable Storage Backends:**
  - `SQLiteBackend`: Zero-configuration local disk persistence with transactional safety and WAL mode.
  - `RedisBackend` & `AsyncRedisBackend`: High-throughput distributed caching for multi-container and cluster deployments.
  - `MemoryBackend`: High-speed in-memory storage for unit tests, ephemeral workers, and local experiments.
- **Framework Integrations:**
  - `langchain`: Semantic caching adapter for LangChain LLMs and runnables.
  - `langgraph`: Contextual state cache node for multi-agent workflows.
  - `llamaindex`: Semantic cache wrapper for LlamaIndex query engines.
  - `openai`: Drop-in wrapper for `openai.OpenAI` and `openai.AsyncOpenAI` chat completions.
  - `fastapi`: ASGI HTTP middleware for automatic semantic caching of REST / SSE endpoints.
- **CLI & Diagnostics:**
  - `longcache stats`: Real-time cache hit rate, token savings, and cost metrics.
  - `longcache clear`: Scoped namespace and total cache eviction.
  - `longcache inspect`: Query lookup, similarity breakdown, and payload inspection.
  - `longcache export`: JSON and CSV dataset exporting for cache analytics.
- **Developer Tools & CI/CD:**
  - CI workflow with Python 3.10, 3.11, and 3.12 test matrix.
  - Automated GitHub Pages documentation deployment with Material for MkDocs.
  - Auto-tagging and GitHub release generation on main branch merge.
  - Automated PyPI publishing pipeline with Trusted Publisher authentication.
