# LongCache

Semantic response reuse for Python LLM and RAG applications. **v0.3.0 is unreleased; Python 3.11 and 3.12 are the tested interpreters.**

LongCache finds similar requests with local embeddings and FAISS, then checks whether their execution context and reuse policy permit the same answer. Similarity alone does not establish correctness. Measure false hits and useful reuse on your own workload.

## Install

v0.3.0 is currently a development checkout. Package-index commands install the available published release. See [installation](getting-started/installation.md) for installing and validating this branch.

```bash
uv pip install longcache
uv pip install 'longcache[openai,server,redis]'
```

The default embedder uses `BAAI/bge-small-en-v1.5`. Model weights must be downloaded once; subsequent inference can run offline. PyTorch, sentence-transformers, NumPy, and FAISS remain required dependencies, including when you inject a custom embedder.

## Explicit request context

```python
from longcache import CacheConfig, Instruction, LongCache, RequestContext, RequestScope

context = RequestContext(
    scope=RequestScope(tenant_id="example-tenant", user_id="example-user"),
    operation="support.answer",
    provider="application",
    provider_scope="support-deployment-v1",
    model="configured-model",
    instructions=(Instruction("system", "Answer using the approved support policy."),),
    generation={"temperature": 0.0},
    prompt_version="support-v3",
    corpus_version="knowledge-v8",
)

with LongCache(CacheConfig(backend="sqlite")) as cache:
    cache.set("What is the return policy?", "Returns are accepted within 30 days.", context=context)
    decision = cache.lookup("What is the return policy?", context=context)
    print(decision.outcome, decision.reason, decision.similarity)
    if decision.outcome == "HIT":
        print(decision.response)
```

The identity values above are examples. In a private application, obtain them from trusted authentication state. A user-supplied header, prompt, or namespace does not authenticate a user. Snapshot a context per request; do not mutate a shared wrapper to select the current user.

Omitted generation parameters, explicit `None`, and explicit values have different identities. Provider/deployment, model, ordered instructions, operation, versions, source versions, response codec, policy version, and embedding generation must match. Model aliases and hidden provider settings need an operator-managed deployment/version token when their behavior changes.

## Scope and policy

| Setting | Required identity | Sharing |
| ------- | ----------------- | ------- |
| `scope_policy="private"` — default | Trusted tenant and user | Within that user and execution context |
| `scope_policy="tenant_shared"` | Trusted tenant | Explicitly shared within that tenant |
| `scope_policy="public"` | Explicit public application/operation context | Public non-personalized answers only |

Missing private identity bypasses both lookup and storage, before embedding. There is no implicit shared tenant. Calls without a complete `RequestContext` remain callable but do not cache.

Lookup outcomes are `HIT`, `MISS`, and `BYPASS`; writes return `StoreDecision` with `STORED` or `REJECTED`. `get`/`aget` are convenience methods returning the response or `None`. `lookup`/`alookup` expose reasons and scores. `None` responses are not stored.

Mandatory admission rejects unsupported response types and incomplete provider responses. Generic responses use strict JSON-compatible primitives; arbitrary objects are not converted to strings. Optional local hooks can restrict admission or verification:

```python
from longcache import PolicyDecision


def admit_response(context, response):
    return PolicyDecision("accept") if response else PolicyDecision("reject", "empty_answer")


def verify_match(query, cached_query, response, similarity, context):
    return PolicyDecision("accept")  # Add domain-specific restrictions here.
```

Pass these as `LongCache(..., admit=admit_response, verifier=verify_match)`. Mandatory checks run first and cannot be overridden. Exceptions, invalid returns, or abstention cannot authorize reuse. Hooks must be synchronous, bounded, local, and side-effect free. Async cache calls offload blocking work; cancellation observed before a commit prevents it, while an already-started commit may finish.

Built-in checks cover supported English negation, quantities/units, protected entities, simple capitalized entity forms, and selected temporal/comparison relations. They reduce risk and are not a proof of semantic equivalence. The current built-in policy abstains for non-ASCII queries or a declared language other than `en`. Configure `protected_entities` for identifiers important to your workload. Verification examines at most eight candidates by default.

## Decorators and async calls

```python
with LongCache(context=context) as cache:

    @cache.cached(ttl=600)
    def answer(query):
        return "An approved complete answer"

    assert answer("return policy") == answer("return policy")
```

A decorator adds the qualified function identity to the explicit operation. For matching inspection or invalidation, use `derived = answer.cache_context_for(context)` and pass `derived` to the maintenance method. Extra function arguments bypass caching. Use an explicit per-call context resolver for a query-only function when trusted identity varies. Resolver failure bypasses caching.

Async applications use `await cache.alookup(...)`, `await cache.aget(...)`, and `await cache.aset(...)`, or decorate an async query-only function. Sync and async calls follow the same policy. These methods do not make Redis index state consistent between processes.

## Supported integrations

| Path | v0.3 status |
| ---- | ----------- |
| Manual lookup/store | Supported with explicit trusted context; independent origin calls |
| `get_or_compute` / `aget_or_compute` and query-only decorators | Supported with exact scoped singleflight |
| OpenAI sync/async chat completions | Supported single-turn text subset with dict/SDK response replay |
| ASGI middleware | Supported bounded single-turn JSON subset with trusted scope resolution; no HTTP invocation coalescing |
| Memory, SQLite, synchronous Redis storage | Supported within one cache process |
| Standalone async Redis backend | Experimental storage interface; unsupported through `LongCache` |
| Generic LangChain, LangGraph, and LlamaIndex adapters | Compatibility pass-through; opaque execution state is not cached |
| All streaming helpers | Pass-through; no cache reads or writes |
| Multi-turn, tools, multimodal, structured-output replay | Bypass until their identity and replay contracts are supported |

### OpenAI

```python
from openai import OpenAI
from longcache.integrations.openai import CachedOpenAI

with OpenAI(timeout=20.0, max_retries=0) as provider, LongCache() as cache:
    client = CachedOpenAI(provider, cache=cache, context=context)
    response = client.chat.completions.create(
        model="your-model",
        messages=[{"role": "user", "content": "What is the return policy?"}],
    )
```

Use `AsyncOpenAI` and await the same `create` method for an async client. Supply `cache_context=per_request_context` when identity varies; this cache-only argument is not forwarded to the provider. The adapter derives actual instructions, model, and allowed parameters from the provider call. The initial allowlist is `model`, `messages`, `temperature`, and `stream=False`; other options bypass. Completed text responses require one valid assistant choice with `finish_reason="stop"`, without refusal/tool output. Unknown mock/custom clients default to the dictionary codec; declare `response_codec="openai-sdk-v1"` when needed.

### ASGI

Configure `LongCacheMiddleware(..., context=context_template, scope_resolver=resolve_scope)` after trusted authentication has populated the ASGI scope. The synchronous resolver returns `RequestScope` from that authenticated principal. Requests with Authorization, Cookie, or X-API-Key headers bypass when no trusted resolver is configured.

The middleware bounds request and response capture to 1 MiB by default. Unsupported streaming and oversized requests pass through. It preserves upstream bodies/statuses and stores only complete eligible status-200 `application/json` responses. Cookies, `Vary`, private/no-store directives in any repeated Cache-Control header, and observed disconnects prevent admission. Headers expose `X-LongCache`, `X-LongCache-Reason`, and same-scope `X-LongCache-Similarity` when applicable. A response-start header cannot report admission that happens after the response body finishes.

## Storage and custom embeddings

SQLite defaults to `.longcache/v2/cache.db`. Redis uses a fully versioned `longcache:v2:<cache_id>:<generation>:...` keyspace. Redis persists responses but does not synchronize in-memory FAISS indexes: use one LongCache process per generation.

Inject `LongCache(..., embedder=my_embedder)` using `BaseEmbedder`. Supply an `EmbeddingDescriptor(model, revision, dimension, preprocessing, normalization)` with immutable revision identity. Vectors must be contiguous, finite, L2-normalized `float32` arrays; batches include shape `(0, dimension)` for empty input. Calls are serialized per embedder. The default model resolves its downloaded commit hash, or accepts `embedder_revision` explicitly.

One store generation has one embedding fingerprint. Equal dimensions do not make two models compatible. A revision or preprocessing change requires a fresh path/generation; mismatched stores fail before vector search or writes.

## Upgrade from v0.2

Compatible format-2 stores need no reset. See [upgrading to v0.3](migration-v0.3.md) for new execution, timeout, failure, and ownership behavior. Separate lookup/store code must opt into shared execution explicitly.

## Legacy upgrade from v0.1.1

1. Preserve the v0.1.1 environment and its store using a consistent backup method.
2. Install v0.3 on Python 3.11/3.12 and configure explicit trusted scope and execution identity.
3. Use a fresh SQLite path or format-2 Redis keyspace. Existing under-scoped entries are not imported or relabeled.
4. Expect initial misses. Verify decisions and isolation with application fixtures before rollout.
5. Roll back by stopping the newer writers and restoring the original environment/store. Do not point v0.1.1 at a format-2 store.

Opening a legacy SQLite file fails without upgrading or deleting it. Raw copies of live SQLite WAL files are not a complete backup procedure. Cleanup of old stores is an explicit operator decision.

## Inspection and maintenance

Python inspection is scoped and redacted: `cache.inspect(query, context=context)` exposes a decision and score without a hit increment or payload. Invalidation/tag invalidation targets one exact context. `clear(operator=True)` explicitly clears the owned store; database/process access remains trusted application access.

The CLI accepts `--context-file context.json` containing the JSON fields of `RequestContext` and a nested scope. Read counts and export data without loading a model:

```bash
longcache stats --context-file context.json
longcache inspect --query 'return policy' --context-file context.json
longcache export --context-file context.json -o my-scope.json
longcache clear --context-file context.json --yes
longcache export --operator -o owned-store.json
longcache clear --all --operator --yes
```

Operator mode is explicit local access to the owned format-2 store, not an authentication mechanism. Exports intentionally include raw payloads. CLI vector inspection supports the configured local embedder; use Python inspection for an injected custom embedder. Hit/miss counters are process-local; entry hit counts persist. Cost fields remain zero estimates until measured accounting is implemented.

## Development and evaluation

```bash
uv sync --locked --extra all --extra dev --extra docs
export LONGCACHE_TEST_REDIS_URL=redis://localhost:16379/0
LONGCACHE_TEST_MODEL=1 uv run --no-sync pytest -q -rs
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mkdocs build --strict
uv run --no-sync python -m evaluation.run --config evaluation/configs/v0_3_safety.json
uv run --no-sync python -m evaluation.rag --output evaluation/reports/v0_3_rag.json
```

The versioned synthetic evaluation includes safe paraphrases, near misses, isolation changes, unsupported shapes, and freshness cases. It reports incorrect hits, accepted-hit counts, useful coverage, bypasses, latency, and confidence bounds. Its expanded templates are correlated and do not establish correctness for unrelated real workloads. Scenario timing uses precomputed real-model vectors; model inference and end-to-end performance are separate measurements. See [evaluation and benchmarks](benchmarks.md) for reproducible commands and [v0.3 acceptance evidence](plans/v0.3.0-acceptance.md) for the test matrix.

The recorded v0.3 wheel validation passed **457 tests with zero skips on each of Python 3.11 and 3.12**, including real Redis/model tests and actual v0.2 upgrade/rollback. The frozen 4,800-case safety evaluation and all six backend/index RAG combinations passed locally. Warm-hit latency and measurement-overhead checks passed in the recorded local benchmark; full miss/burst/overload and singleflight-overhead qualification remain pending. Hosted CI, domain review, and release-candidate feedback also remain release gates. See the [execution record](plans/v0.3.0-execution.md) and [release checklist](plans/v0.3.0-acceptance.md#release-readiness-checklist).

## License

MIT. See [MIT license](https://github.com/ENDEVSOLS/LongCache/blob/main/LICENSE).

## v0.3 reliability

Shared execution, strict/bypass failure policy, owned shutdown and bounded private measurements are documented in [reliability](reliability.md). See the [v0.3 execution record](plans/v0.3.0-execution.md) for new validation; the earlier v0.2 records are retained separately as historical evidence.
