<p align="center">
  <img src="https://raw.githubusercontent.com/ENDEVSOLS/LongCache/main/assets/logo.png" alt="LongCache Logo" width="320"/>
</p>

<p align="center"><strong>Semantic similarity cache for LLM and RAG pipelines .</strong></p>

<p align="center">
<a href="https://pypi.org/project/longcache/"><img src="https://img.shields.io/pypi/v/longcache" alt="PyPI Version"></a>
<a href="https://pepy.tech/project/longcache"><img src="https://static.pepy.tech/badge/longcache" alt="Total Downloads"></a>
<a href="https://pepy.tech/project/longcache"><img src="https://static.pepy.tech/badge/longcache/month" alt="Monthly Downloads"></a>
<a href="https://github.com/ENDEVSOLS/LongCache/stargazers"><img src="https://img.shields.io/github/stars/ENDEVSOLS/LongCache?style=flat" alt="GitHub Stars"></a>
<a href="https://github.com/ENDEVSOLS/LongCache/actions/workflows/ci.yml"><img src="https://github.com/ENDEVSOLS/LongCache/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<img src="https://img.shields.io/pypi/pyversions/longcache" alt="Python Versions">
<a href="https://github.com/ENDEVSOLS/LongCache/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ENDEVSOLS/LongCache" alt="License"></a>
</p>

<p align="center">
<a href="https://endevsols.github.io/LongCache/"><strong>📖 Documentation</strong></a>  · 
<a href="https://endevsols.github.io/LongCache/getting-started/quickstart/"><strong>Quick Start</strong></a>  · 
<a href="https://endevsols.github.io/LongCache/api-reference/"><strong>API Reference</strong></a>  · 
<a href="https://github.com/ENDEVSOLS/LongCache/blob/main/CHANGELOG.md"><strong>Changelog</strong></a>
</p>

**LongCache** reuses stored responses when trusted scope, execution context, response admission, and verification permit reuse. Python 3.11 and 3.12 are tested and  remains the Python 3.10-compatible line. Measure response quality, latency, and savings on your own workload.

---

## Why LongCache?

| Problem                                      | Solution                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Repeated queries consume LLM tokens          | Semantic cache intercepts similar queries before they reach the model                                        |
| Repeated LLM requests waste time and money   | LongCache can reuse a stored answer when the request and context qualify                                     |
| String-match caches miss paraphrased queries | FAISS + BGE-small understands semantic meaning, not just exact tokens                                        |
| Most caches are coupled to one LLM provider  | Scoped caching for supported OpenAI requests and raw functions; generic framework adapters pass through      |
| Cross-model cache poisoning is a silent bug  | Immutable request identity isolates tenant/user, provider/model, instructions, settings, and source versions |

**Results depend on the workload, hit quality, embedding cost, and cache configuration; measure savings in your own application.**

---

## Quick Start

### Installation

```bash
# Core — SQLite backend + FAISS + BGE-small (no external services needed)
uv pip install longcache

# With LangChain + LangGraph compatibility adapters (pass-through in v0.2)
uv pip install longcache[langchain]

# With Redis response storage (FAISS index remains local to this process)
uv pip install longcache[redis]

# Everything
uv pip install longcache[all]
```

> **Offline-friendly**: LongCache embeds queries locally. After the BGE-small model weights are downloaded on first run, no internet connection is required.

### 30-Second Example

```python
from longcache import LongCache, RequestContext, RequestScope

context = RequestContext(
    scope=RequestScope(tenant_id="your-tenant", user_id="your-user"),
    operation="support.answer",
    provider="your-provider",
    provider_scope="your-deployment-v1",
    model="your-model",
    prompt_version="support-v1",
    corpus_version="knowledge-v1",
)
cache = LongCache(context=context)


@cache.cached
def my_rag_chain(query: str) -> str:
    return expensive_llm_call(query)  # Your retrieval + LLM pipeline


# First call: MISS → runs the full pipeline
result1 = my_rag_chain("What is your refund policy?")

# Second call — exact query: HIT → cached response
result2 = my_rag_chain("What is your refund policy?")

# Third call — paraphrase: HIT only if similarity and verification both pass
result3 = my_rag_chain("How do I get a refund?")
```

Obtain scope from trusted authentication state and use context values matching the actual provider execution. Missing identity bypasses caching. When RAG documents change, snapshot their revisions in `corpus_version` / `document_versions` before lookup and generation; see the [changing-document RAG guide](docs/integrations/versioned-rag.md).

Async functions are detected automatically and use the same explicit context:

```python
@cache.cached(namespace="faq", ttl=7200)
async def my_async_chain(query: str) -> str:
    return await async_llm_call(query)
```

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                        Query Lifecycle                       │
│                                                              │
│  User Query                                                  │
│       │                                                      │
│       ▼                                                      │
│  ┌───────────────────┐                                       │
│  │   LocalEmbedder   │  BAAI/bge-small-en-v1.5 — no API     │
│  │   (local model)   │  call, runs offline after download   │
│  └────────┬──────────┘                                       │
│           │ 384-dim vector                                   │
│           ▼                                                  │
│  ┌───────────────────┐    HIT (scope + policy verified)       │
│  │  SimilarityIndex  │ ─────────────────────────────────▶   │
│  │  FAISS + Backend  │         Return cached response        │
│  └────────┬──────────┘                                       │
│           │    MISS                                          │
│           ▼                                                  │
│  ┌───────────────────┐                                       │
│  │  Full RAG Chain   │  Retrieval + LLM            │
│  └────────┬──────────┘                                       │
│           │                                                  │
│           ▼                                                  │
│  ┌───────────────────┐                                       │
│  │   CacheWriter     │  Stores vector + response + TTL       │
│  │                   │  + namespace + tags + context         │
│  └───────────────────┘                                       │
└──────────────────────────────────────────────────────────────┘
```

1. **Intercept** — validate trusted scope and supported request shape before embedding
2. **Embed** — the query is embedded locally with BGE-small (no embedding API call)
3. **Search** — FAISS searches for cosine similarity against cached vectors
4. **HIT**: verify the candidate and decode an eligible response before serving it
5. **MISS**: run the full pipeline, admit an eligible result, and return it; unsupported requests **BYPASS** caching

---

## Configuration

### CacheConfig

All settings are controlled via the `CacheConfig` dataclass:

```python
from longcache import LongCache, CacheConfig

config = CacheConfig(
    similarity_threshold=0.92,  # 0.0–1.0; higher = stricter matching
    ttl_seconds=3600,  # Entry lifetime in seconds (0 = never expire)
    max_entries=10_000,  # Max entries per semantic partition (oldest evicted first)
    namespace="default",  # Logical isolation group
    embedder_model="BAAI/bge-small-en-v1.5",  # Local embedding model
    backend="sqlite",  # "sqlite" | "redis" | "memory"
    db_path=".longcache/v2/cache.db",  # SQLite path (auto-creates parent dirs)
    redis_url="redis://localhost:6379/0",  # Redis connection URL
    track_stats=True,  # Track process-local hit/miss counters
    faiss_index_type="flat",  # "flat" (exact) | "hnsw" (approximate, faster at scale)
    hnsw_m=32,  # HNSW connections per node
    hnsw_ef_search=64,  # HNSW search depth
    scope_policy="private",  # Trusted tenant + user required; sharing is explicit
    # Legacy execution fields below do not replace RequestContext
    model="",  # LLM model identifier (e.g. "gpt-4o")
    system_prompt="",  # System prompt string
    temperature=-1.0,  # Sampling temperature (-1.0 = unset)
)

cache = LongCache(config, context=context)
```

### From a Dictionary

```python
config = CacheConfig.from_dict(
    {
        "similarity_threshold": 0.90,
        "backend": "redis",
        "ttl_seconds": 7200,
    }
)
```

### From Environment Variables

```bash
export LONGCACHE_SIMILARITY_THRESHOLD=0.90
export LONGCACHE_BACKEND=redis
export LONGCACHE_TTL_SECONDS=7200
export LONGCACHE_REDIS_URL=redis://prod-redis:6379/0
export LONGCACHE_MODEL=gpt-4o
```

```python
config = CacheConfig.from_env()  # Reads all LONGCACHE_* env vars automatically
cache = LongCache(config, context=context)
```

### Similarity Threshold Guide

| Threshold      | Behavior                      | Recommended Use Case            |
| -------------- | ----------------------------- | ------------------------------- |
| 0.99           | Near-exact match only         | Precise FAQ lookup              |
| 0.95           | Very similar phrasing         | Customer support                |
| **0.92** | **Default — balanced** | **General RAG pipelines** |
| 0.85           | Looser matching               | Exploratory search              |
| 0.75           | Topic-level clustering        | Broad domain matching           |

**Tip:** Start at 0.92 and adjust based on your observed hit rate vs. false-positive rate.

---

## Features ✨

### Core

- ✅ **Semantic Similarity Cache:** Finds paraphrase candidates, then checks scope and reuse policy before serving an answer.
- ✅ **Local Embedding:** BAAI/bge-small-en-v1.5 runs on-device after its first download; latency depends on hardware.
- ✅ **`@cache.cached` Decorator:** Scoped caching for query-only sync or async functions; additional arguments bypass
- ✅ **`@cache.cached_stream` Decorator:** Preserves sync/async generator interfaces as pass-through; no cache reads or writes in v0.2
- ✅ **Composite Cache Keys:** Full SHA-256 identity includes trusted scope, execution settings, versions, codec, policy, and embedding generation
- ✅ **Tag-Based Invalidation:** Attach tags to entries at write time; invalidate by tag within the exact scoped execution context
- ✅ **Per-Entry TTL:** Override TTL per `set()` call or decorator — supports never-expire (`ttl=0`) and short-lived entries
- ✅ **LRU Eviction:** Configurable `max_entries` per semantic partition; oldest entries evicted when the limit is reached

### Reliability

- ✅ **Thread-Safe Statistics:** Hit/miss counts protected by a dedicated lock; safe under 8+ concurrent threads
- ✅ **Accepted-Hit Accounting:** Process counters reset on restart; stored entry hit counts persist, and unmeasured savings remain zero
- ✅ **Rebuild-Safe FAISS Index:** Index is fully rebuilt after TTL sweeps or evictions — prevents stale vector accumulation
- ✅ **Background TTL Sweeper:** Daemon thread runs every 5 minutes to delete expired entries; stops when the cache is closed
- ✅ **Correct Miss Counting:** Supported eligible lookups count hits only after successful verification and decoding
- ✅ **Config Validation:** `CacheConfig` validates all fields on construction with clear error messages

### Integrations & Backends

- ✅ **SQLite Backend:** No external service, WAL mode, versioned manifests, transactional initialization; default backend
- ✅ **Redis Backend:** Versioned durable response storage with TTL; one LongCache process per generation
- ✅ **Async Redis Backend:** Experimental standalone storage interface; unsupported through `LongCache`
- ✅ **In-Memory Backend:** Zero I/O, ideal for unit tests and CI/CD pipelines
- ✅ **LangChain Integration:** `CachedLLM` and `CachedChain` compatibility wrappers; pass-through in v0.2
- ✅ **LangGraph Integration:** `CachedState` compatibility interface; graph state is not cached in v0.2
- ✅ **LlamaIndex Integration:** `CachedQueryEngine` preserves the engine interface without caching opaque state
- ✅ **OpenAI Integration:** Supported single-turn text requests with explicit context; sync/async dict and SDK replay
- ✅ **FastAPI Middleware:** Bounded supported chat-completion JSON with trusted scope resolution and diagnostic headers
- ✅ **CLI:** `longcache stats`, `clear`, `inspect`, `export` — manage the cache without writing code

---

## Backends

### SQLite (Default — Zero External Dependencies)

```python
cache = LongCache(
    CacheConfig(
        backend="sqlite",
        db_path=".longcache/v2/cache.db",  # Auto-creates parent directories
    )
)
```

|                    |                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| **Pros**     | No external service, persistent entries/hit counts, WAL mode, compatible-generation validation |
| **Cons**     | Single-machine only                                                                            |
| **Best for** | Development, single-process applications, CLI tools                                            |

### Redis (Shared Storage)

```bash
uv pip install longcache[redis]
```

```python
cache = LongCache(
    CacheConfig(
        backend="redis",
        redis_url="redis://localhost:6379/0",
        max_entries=100_000,
    )
)
```

|                    |                                                                                                                |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| **Pros**     | Shared response storage and Redis key expiry                                                                   |
| **Cons**     | Requires Redis; each LongCache process keeps its own FAISS index, so cross-process writes are not synchronized |
| **Best for** | A single cache process using Redis for durable response storage                                                |

### Async Redis backend status

`AsyncRedisCacheBackend` can be used and tested as a standalone storage backend. It is not supported through `LongCache` in v0.2.0; selecting `backend="async_redis"` fails immediately. End-to-end async Redis integration is planned for a later release.

### In-Memory (Testing & CI)

```python
cache = LongCache(CacheConfig(backend="memory"))
```

|                    |                                              |
| ------------------ | -------------------------------------------- |
| **Pros**     | Fastest possible, zero I/O, no files created |
| **Cons**     | Lost on process restart, single-process only |
| **Best for** | Unit tests, CI/CD pipelines, benchmarks      |

---

## Integrations

**v0.2 support:** generic LangChain, LangGraph, and LlamaIndex adapters below preserve their interfaces but bypass cache reads/writes. Use the explicit-context manual/decorator path for caching proven application workflows. All streaming helpers also pass through.

### LangChain

```bash
uv pip install longcache[langchain]
```

#### `CachedLLM` — Wrap Any LLM

```python
from langchain_openai import ChatOpenAI
from longcache.integrations.langchain import CachedLLM

llm = CachedLLM(ChatOpenAI(model="gpt-4o"), cache=LongCache())

response = llm.invoke("What is the refund policy?")  # sync
response = await llm.ainvoke("What is the refund policy?")  # async

for chunk in llm.stream("What is RAG?"):  # sync streaming
    print(chunk, end="")

async for chunk in llm.astream("What is RAG?"):  # async streaming
    print(chunk, end="")
```

#### `CachedChain` — Wrap Any LCEL Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from longcache.integrations.langchain import CachedChain

chain = ChatPromptTemplate.from_template("Answer: {question}") | ChatOpenAI() | StrOutputParser()
cached_chain = CachedChain(chain, cache=LongCache(), query_key="question")

result = cached_chain.invoke({"question": "What is RAG?"})
result = await cached_chain.ainvoke({"question": "What is RAG?"})
```

### LangGraph

```python
from langgraph.graph import StateGraph
from longcache import LongCache
from longcache.integrations.langgraph import CachedState

cache = LongCache()


def cache_node(state: dict) -> dict:
    return CachedState.check(state, cache, query_key="messages")


def llm_node(state: dict) -> dict:
    response = my_llm_call(state["messages"])
    CachedState.save(state, response, cache)
    return {"response": response}


def should_use_llm(state: dict) -> str:
    return "llm" if not state.get("__cache_hit") else "end"


graph = StateGraph(dict)
graph.add_node("cache", cache_node)
graph.add_node("llm", llm_node)
graph.add_conditional_edges("cache", should_use_llm)
graph.set_entry_point("cache")
```

Async variants: `CachedState.acheck()` and `CachedState.asave()`.

### LlamaIndex

```bash
uv pip install longcache[llamaindex]
```

```python
from llama_index.core import VectorStoreIndex
from longcache.integrations.llamaindex import CachedQueryEngine

engine = VectorStoreIndex.from_documents(documents).as_query_engine()
cached_engine = CachedQueryEngine(engine, cache=LongCache())
response = cached_engine.query("What is the refund policy?")
```

### OpenAI (Direct)

```bash
uv pip install longcache[openai]
```

```python
from openai import OpenAI
from longcache.integrations.openai import CachedOpenAI

client = CachedOpenAI(OpenAI(), cache=LongCache(), context=context)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is RAG?"}],
)
```

In v0.2.0, OpenAI caching requires a complete trusted context and the supported single-turn text subset. Multi-turn, tool, multimodal, structured-output, streaming, and unknown-parameter requests bypass. Complete responses require `finish_reason="stop"`. For varying identities, pass `cache_context=per_request_context`; namespaces alone do not establish tenant isolation.

---

## FastAPI Middleware

```bash
uv pip install longcache[server]
```

```python
from fastapi import FastAPI
from longcache import LongCache
from longcache.middleware import LongCacheMiddleware

app = FastAPI()
app.add_middleware(
    LongCacheMiddleware,
    cache=LongCache(),
    context=context,  # Execution template from the quick start
    scope_resolver=lambda scope: scope["state"]["request_scope"],
    cached_routes={"/v1/chat/completions"},
)

# Authentication middleware must populate state["request_scope"] with a trusted RequestScope.
# Resolver failure bypasses; only complete eligible bounded JSON responses are stored.
# X-LongCache: HIT  →  cached response served
# X-LongCache: MISS/BYPASS → upstream response; admission is checked afterward
```

### Streaming Cache Adapter

```python
from longcache.middleware import StreamingCacheAdapter


def my_stream(query: str):
    for chunk in llm.stream(query):
        yield chunk


adapter = StreamingCacheAdapter(cache, my_stream)

# Pass-through — streams from LLM without cache storage
for token in adapter.stream("What is RAG?"):
    print(token, end="")

# Repeated calls also reach the LLM; streaming replay is unsupported in v0.2
for token in adapter.stream("What is RAG?"):
    print(token, end="")
```

---

## CLI

LongCache ships a command-line tool for cache management and inspection. Supply a JSON `RequestContext` via `--context-file`; see the [CLI guide](docs/cli.md). Whole-store operations require explicit operator mode:

```bash
# View cache statistics (hits, misses, hit rate, estimated savings)
longcache stats --context-file context.json
longcache stats --namespace faq --context-file context.json

# Clear a namespace
longcache clear --namespace faq --context-file context.json --yes

# Clear the entire cache
longcache clear --all --operator --yes

# Inspect the nearest cached entry for a query
longcache inspect --query "refund policy" --context-file context.json

# Export cache entries as JSON
longcache export --context-file context.json --format json
longcache export --namespace faq --context-file context.json -o faq_cache.json
```

---

## API Reference

### `LongCache(config=None, *, context=None, embedder=None, admit=None, verifier=None, metadata_only=False)`

Main facade class. Accepts an optional `CacheConfig` and explicit request context, embedder, and synchronous policy hooks. Missing identity bypasses caching.

#### Core Methods

| Method                                                                         | Description                                                                           |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `lookup(query, ..., context=...)`, `alookup(...)`                          | Typed `HIT` / `MISS` / `BYPASS` decision with reason and decoded response.      |
| `get(query, ..., context=...)`, `aget(...)`                                | Decoded response or `None`; same lookup policy.                                     |
| `set(query, response, ..., context=...)`, `aset(...)`                      | `StoreDecision` with `STORED` / `REJECTED`; strict admission and serialization. |
| `cached(..., context=..., context_resolver=...)`                             | Query-only sync/async decorator; extra arguments bypass.                              |
| `cached_stream(...)`                                                         | Sync/async generator pass-through; no caching.                                        |
| `invalidate(..., context=...)`, `invalidate_by_tag(tag, ..., context=...)` | Delete only within the exact scoped execution context.                                |
| `clear(operator=True)`                                                       | Explicit whole-store clearing.                                                        |
| `stats(context=...)`                                                         | Scoped process counters and stored-entry count.                                       |
| `inspect(query, ..., context=...)`                                           | Redacted decision/score dictionary without hit accounting.                            |
| `close()` / context manager                                                  | Release owned background work and backend resources.                                  |

#### `@cache.cached` Decorator Options

```python
@cache.cached(
    namespace="faq",  # Logical isolation group
    ttl=7200,  # Entry lifetime override (seconds)
    tags=["billing"],  # Tags for selective invalidation
    context=context,  # Complete trusted execution context
)
def my_chain(query: str) -> str: ...
```

Use a per-call `context_resolver` when trusted identity varies. For matching inspection/invalidation, derive the decorator context with `my_chain.cache_context_for(context)`. Streaming decorators do not cache in v0.2.

#### `stats()` Return Value

```python
{
    "hits": 420,
    "misses": 180,
    "hit_rate": 0.70,
    "total_saved_ms": 0.0,
    "estimated_tokens_saved": 0,
    "estimated_cost_saved_usd": 0.0,
    "total_entries": 95,
}
```

Process hit/miss counters reset on restart; stored entry hit counts persist. Unmeasured timing/token/cost savings remain zero.

---

### `CacheConfig` Field Reference

| Field                        | Type      | Default                        | Description                                                                         |
| ---------------------------- | --------- | ------------------------------ | ----------------------------------------------------------------------------------- |
| `similarity_threshold`     | `float` | `0.92`                       | Cosine similarity threshold for cache hits (0.0–1.0)                               |
| `ttl_seconds`              | `int`   | `3600`                       | Entry lifetime in seconds. `0` = never expire                                     |
| `max_entries`              | `int`   | `10_000`                     | Max entries per semantic partition; oldest entries evicted first                    |
| `namespace`                | `str`   | `"default"`                  | Default namespace                                                                   |
| `embedder_model`           | `str`   | `"BAAI/bge-small-en-v1.5"`   | Local sentence-transformers model                                                   |
| `backend`                  | `str`   | `"sqlite"`                   | `"sqlite"`, `"redis"`, `"memory"` through `LongCache`                       |
| `db_path`                  | `str`   | `".longcache/v2/cache.db"`   | SQLite database path                                                                |
| `redis_url`                | `str`   | `"redis://localhost:6379/0"` | Redis connection URL                                                                |
| `track_stats`              | `bool`  | `True`                       | Track process-local hit/miss counters                                               |
| `faiss_index_type`         | `str`   | `"flat"`                     | `"flat"` (exact) or `"hnsw"` (approximate)                                      |
| `hnsw_m`                   | `int`   | `32`                         | HNSW connections per node (higher = more accurate, more memory)                     |
| `hnsw_ef_search`           | `int`   | `64`                         | HNSW search depth (higher = more accurate, slower)                                  |
| `scope_policy`             | `str`   | `"private"`                  | Trusted tenant/user scope; `tenant_shared` and `public` require explicit opt-in |
| `cache_id`, `generation` | `str`   | `"default"`                  | Owned Redis keyspace and storage generation                                         |
| `verification_budget`      | `int`   | `8`                          | Maximum candidates checked per lookup                                               |
| `model`                    | `str`   | `""`                         | Legacy execution field; does not replace trusted context                            |
| `system_prompt`            | `str`   | `""`                         | Global system prompt for composite key scoping                                      |
| `temperature`              | `float` | `-1.0`                       | Global temperature for composite key scoping (`-1.0` = unset)                     |

---

### `CacheEntry` Field Reference

Used internally by backends and available through scoped `entries(context=...)`. `inspect()` returns a redacted decision dictionary, not a `CacheEntry`.

| Field             | Type             | Description                                     |
| ----------------- | ---------------- | ----------------------------------------------- |
| `id`            | `int \| None`   | Backend-assigned unique identifier              |
| `namespace`     | `str`          | Canonical scoped execution partition            |
| `query`         | `str`          | Original query text                             |
| `vector`        | `np.ndarray`   | 384-dim L2-normalized float32 embedding         |
| `response`      | `Any`          | Cached LLM response (must be JSON-serializable) |
| `created_at`    | `float`        | Unix timestamp of entry creation                |
| `expires_at`    | `float \| None` | Unix timestamp of expiry (`None` = no expiry) |
| `hit_count`     | `int`          | Number of times this entry has been served      |
| `model`         | `str`          | Model identifier at write time                  |
| `system_prompt` | `str`          | System prompt at write time                     |
| `temperature`   | `float`        | Temperature at write time (`-1.0` = unset)    |
| `tags`          | `list[str]`    | Tags for selective invalidation                 |

---

## Advanced Usage

### Per-Namespace Threshold Tuning

```python
faq_cache = LongCache(CacheConfig(namespace="faq", similarity_threshold=0.95))
chat_cache = LongCache(CacheConfig(namespace="general", similarity_threshold=0.85))
```

### Per-Entry TTL Override

```python
cache.set("volatile_query", response, ttl=300, context=context)  # 5-minute TTL
cache.set("stable_query", response, ttl=86400, context=context)  # 24-hour TTL
cache.set("permanent", response, ttl=0, context=context)  # Never expires
```

### Cache Invalidation Patterns

```python
# Clear entries within the exact context and namespace
cache.invalidate("faq", context=context)

# Invalidate a tag within the same context
cache.invalidate_by_tag("doc:refund_v2", context=context)

# Explicit operator access to the whole owned store
cache.clear(operator=True)
```

### Async Manual Control

```python
cached = await cache.aget("What is your refund policy?", context=context)
if cached is None:
    result = await my_pipeline("What is your refund policy?")
    await cache.aset("What is your refund policy?", result, tags=["policy"], context=context)
```

### HNSW Index for Large Caches (100K+ Entries)

```python
cache = LongCache(
    CacheConfig(
        faiss_index_type="hnsw",
        hnsw_m=32,  # More connections → more accurate, more memory
        hnsw_ef_search=64,  # Higher → more accurate, slower
        max_entries=500_000,
    )
)
```

### Inspecting the Cache

```python
decision = cache.inspect("refund policy", context=context)
# Redacted keys: outcome, reason, similarity, id, threshold, format_version.
# No raw query/response, and no hit increment.
```

### Custom Integration: Decision-Based Control

Hit accounting is automatic; `record_hit`, `record_miss`, and `track_miss` were removed in v0.2.

```python
def answer(query):
    decision = cache.lookup(query, context=context)
    if decision.outcome == "HIT":
        return decision.response
    result = my_pipeline(query)
    cache.set(query, result, context=context)  # Returns STORED or REJECTED.
    return result
```

---

## Architecture

### Component Overview

```
src/longcache/
├── __init__.py              # Public API: LongCache, CacheConfig
├── config.py                # CacheConfig dataclass + from_dict/from_env
├── cli.py                   # CLI: stats, clear, inspect, export
├── middleware.py             # FastAPI/Starlette ASGI middleware + StreamingCacheAdapter
├── core/
│   ├── entry.py             # CacheEntry dataclass (query, vector, response, tags, context)
│   ├── embedder.py          # LocalEmbedder — BGE-small singleton, lazy-loaded
│   ├── index.py             # SimilarityIndex — FAISS Flat/HNSW, rebuild-safe
│   └── cache.py             # LongCache — main orchestrator, thread-safe stats
├── backends/
│   ├── base.py              # AbstractCacheBackend ABC
│   ├── sqlite.py            # SQLiteCacheBackend — WAL mode, tag table, stats persistence
│   ├── redis.py             # RedisCacheBackend — sync, tag support, atomic TTL
│   ├── async_redis.py       # AsyncRedisCacheBackend — async, non-blocking
│   └── memory.py            # InMemoryCacheBackend — testing, zero I/O
└── integrations/
    ├── langchain.py         # CachedLLM, CachedChain
    ├── langgraph.py         # CachedState
    ├── llamaindex.py        # CachedQueryEngine
    └── openai.py            # CachedOpenAI
```

### Embedding Model

LongCache uses **BAAI/bge-small-en-v1.5** by default:

- **384-dimensional** vectors — compact and fast
- Local embedding after the model is downloaded; measure latency on your hardware
- **Fully offline** after initial weight download — no API calls ever
- **L2-normalized** — inner product equals cosine similarity in FAISS
- **Singleton pattern** — the model is loaded once and reused across all instances

### FAISS Index

| Index             | Config Value | Characteristics                                                 |
| ----------------- | ------------ | --------------------------------------------------------------- |
| `IndexFlatIP`   | `"flat"`   | Exact search, deterministic, best for ≤ 100K entries           |
| `IndexHNSWFlat` | `"hnsw"`   | Approximate search, sub-ms at any scale, best for 100K+ entries |

After TTL sweeps, namespace invalidations, or evictions, the local FAISS index is rebuilt from live backend entries. Redis storage alone does not synchronize indexes between processes.

### Thread Safety

- **Stats** (`hits`, `misses`, `total_saved_ms`) are protected by a dedicated `threading.Lock()`.
- **SQLite backend** uses WAL mode and a per-connection `threading.RLock()`.
- **The TTL sweeper** runs as a daemon thread and signals shutdown cleanly via `threading.Event`.
- **`get()`/`aget()`** correctly record misses in a thread-safe manner.

### Background TTL Sweeper

A daemon thread checks for expired entries every 5 minutes when `ttl_seconds > 0`. Call `cache.close()` or use `with LongCache(...) as cache:` to stop it and release backend connections.

### Persistent Statistics

Stored entry hit counts persist; process-level hit/miss counters reset on restart. Rejected candidates and inspection do not count as hits. Unmeasured savings remain zero.

---

## Performance

Cache hit rate and savings vary with traffic, threshold, embedding model, and safety rules. Benchmark your own workload. Record hit rate, incorrect-hit rate, bypass rate, p50/p95/p99 latency, and actual provider usage before claiming cost savings.

---

## Project Structure

```
longcache/
├── pyproject.toml
├── README.md
├── src/
│   └── longcache/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── middleware.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── entry.py
│       │   ├── embedder.py
│       │   ├── index.py
│       │   └── cache.py
│       ├── backends/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── sqlite.py
│       │   ├── redis.py
│       │   ├── async_redis.py
│       │   └── memory.py
│       └── integrations/
│           ├── __init__.py
│           ├── langchain.py
│           ├── langgraph.py
│           ├── llamaindex.py
│           └── openai.py
├── tests/
│   ├── conftest.py
│   ├── test_embedder.py
│   ├── test_index.py
│   ├── test_cache.py
│   ├── test_backends.py
│   ├── test_langchain.py
│   ├── test_middleware.py
│   └── test_cli.py
└── examples/
    ├── basic_langchain.py
    ├── fastapi_middleware.py
    └── redis_production.py
```

---

## Dependencies

### Core (Required)

| Package                   | Version             | Purpose                         |
| ------------------------- | ------------------- | ------------------------------- |
| `sentence-transformers` | `>=2.3.0`         | Local embedding — BGE-small    |
| `faiss-cpu`             | `>=1.7.4`         | Similarity search (Flat & HNSW) |
| `numpy`                 | `>=1.24.0,<2.0.0` | Vector math                     |
| `torch`                 | CPU (via uv)        | sentence-transformers runtime   |
| `click`                 | `>=8.1.0`         | CLI framework                   |

> **Note**: `numpy<2.0.0` is required — numpy 2.0 breaks ABI compatibility with `faiss-cpu` and `torch`.

### Optional Extras

| Extra            | Package(s)                                                       | Purpose                      |
| ---------------- | ---------------------------------------------------------------- | ---------------------------- |
| `[redis]`      | `redis>=5.0.0,<6.0.0`                                          | Redis & async Redis backends |
| `[langchain]`  | `langchain-core>=1.0.0`, `langgraph>=1.0.0`                  | LangChain + LangGraph        |
| `[llamaindex]` | `llama-index-core>=0.10.0`                                     | LlamaIndex integration       |
| `[openai]`     | `openai>=1.0.0,<2.0.0`                                         | Direct OpenAI integration    |
| `[server]`     | `fastapi>=0.109.0`, `starlette>=0.36.0`, `uvicorn>=0.27.0` | ASGI middleware              |
| `[all]`        | All of the above                                                 | Everything                   |

### Development

| Package            | Version      | Purpose            |
| ------------------ | ------------ | ------------------ |
| `pytest`         | `>=8.0.0`  | Test runner        |
| `pytest-asyncio` | `>=0.23.0` | Async test support |
| `pytest-cov`     | `>=5.0.0`  | Coverage reporting |
| `httpx`          | `>=0.27.0` | ASGI test client   |
| `ruff`           | `>=0.4.0`  | Linter / formatter |

---

## Running Tests

```bash
# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=longcache --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_cache.py

# Verbose output
uv run pytest -v

# Optional real-model tests (downloads model weights on first run)
LONGCACHE_TEST_MODEL=1 uv run pytest tests/test_embedder.py -q
```

---

## License

MIT
