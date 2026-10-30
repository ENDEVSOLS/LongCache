# Quick start

Start with the [explicit request-context example](../index.md#explicit-request-context). A private request needs trusted tenant/user identity, an operation, provider/deployment, model, and prompt version. Namespace alone is insufficient.

Use `cache.lookup(query, context=context)` for a typed decision. `HIT` carries a decoded response; `MISS` needs a new provider result; `BYPASS` means the request is not eligible for caching. `cache.set(..., context=context)` returns `STORED` or `REJECTED`. Async equivalents are `alookup` and `aset`.

For shared execution of identical concurrent requests, use:

```python
with LongCache(context=context) as cache:
    answer = cache.get_or_compute(query, lambda: call_provider(query))

async with LongCache(context=context) as cache:
    answer = await cache.aget_or_compute(query, lambda: async_call_provider(query))
```

These snippets use the trusted context from the linked example and application-owned
provider functions. Sync callbacks return responses; async factories return awaitables.
Separate lookup/call/store operations do not coalesce the intervening computation.
Use the async interface on an event-loop thread.

Release resources with `with` / `close()` in sync code or `async with` / `aclose()` in
async code. `examples/safe_reuse.py` and `examples/reliability.py` are executable offline
examples with deterministic injected embedders and no external service/model download.
For changing documents, follow the [versioned RAG guide](../integrations/versioned-rag.md).
Before upgrading, read [v0.3 migration](../migration-v0.3.md) and the
[reliability contracts](../reliability.md).
