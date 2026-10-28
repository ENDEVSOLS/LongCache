# Quick start

Start with the [explicit request-context example](../index.md#explicit-request-context). A private request needs trusted tenant/user identity, an operation, provider/deployment, model, and prompt version. Namespace alone is insufficient.

Use `cache.lookup(query, context=context)` for a typed decision. `HIT` carries a decoded response; `MISS` needs a new provider result; `BYPASS` means the request is not eligible for caching. `cache.set(..., context=context)` returns `STORED` or `REJECTED`. Async equivalents are `alookup` and `aset`.

Use a context manager or `close()` to release the cache. The executable `examples/safe_reuse.py` uses a deterministic injected embedder and no external service or model download. For source documents that change, follow the [versioned RAG guide](../integrations/versioned-rag.md).
