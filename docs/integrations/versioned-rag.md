# RAG with changing documents

A repeated question can need a new answer after its source documents change. Retrieve an immutable document snapshot before cache lookup. Include its corpus and document revisions in `RequestContext`, then use the same snapshot and context for generation and storage.

1. Resolve tenant and user from trusted authentication state.
2. Retrieve the documents and their immutable revision identifiers together.
3. Build a context containing `corpus_version`, `document_versions`, and the actual provider/model/prompt settings.
4. Look up the question in that context. On a miss, generate from that exact snapshot and store with the same context.
5. When documents change, use their new revisions. In-flight results from an old snapshot remain in its old partition.

Do not read a mutable "current revision" again after generation and attach it to an answer generated from older documents. Application-managed versions are required; generic retrieval adapters do not discover this metadata automatically in v0.3.

## Runnable repository example

From a source checkout, the following offline demonstration uses the maintained application example and deterministic fixture embedder. It needs no API key, model download, or external service.

```python
import asyncio
from examples.safe_reuse import ExampleEmbedder
from examples.versioned_rag import DocumentSnapshot, answer
from longcache import CacheConfig, LongCache


async def main():
    old = DocumentSnapshot("revision-1", "Returns accepted within 30 days.")
    new = DocumentSnapshot("revision-2", "Returns accepted within 14 days.")
    question = "What is the return window?"
    with LongCache(CacheConfig(backend="memory"), embedder=ExampleEmbedder()) as cache:
        assert await answer(cache, question, old) == old.text
        assert await answer(cache, question, old) == old.text
        assert await answer(cache, question, new) == new.text
        assert await answer(cache, question, new) == new.text
        assert cache.stats(context=new.context("shop", "alice"))["hits"] == 1


asyncio.run(main())
```

`examples/versioned_rag.py` uses an extractive fixture generator. Its optional async `generate(snapshot)` callback demonstrates where a provider call belongs. When adapting it, change the example's fixture provider/model identity to the actual deployment and represent all response-affecting instructions and settings. Preserve the immutable snapshot boundary.

## Invalidation and isolation

The example tags entries with the document ID. `cache.invalidate_by_tag(document_id, context=old_context)` removes entries only in that exact context; it does not delete another revision's or tenant's entries. Do not treat this as global document invalidation. New document versions partition responses even when old entries remain until expiry or explicit maintenance.

## Validation

`tests/test_versioned_rag_v2.py` checks version changes, repeated hits, tenant isolation, scoped tag invalidation, and document changes while generation is in flight. It runs against memory, SQLite, and real Redis with both flat and HNSW indexes.

The separate real-embedding workflow evaluation is:

```bash
export LONGCACHE_TEST_REDIS_URL=redis://localhost:16379/0
uv run --no-sync python -m evaluation.rag
```

A disposable Redis service and the pinned model weights are required. The recorded run passed all six backend/index combinations. This verifies application integration and freshness with authored documents and extractive answers. Customer-domain representativeness and live-provider answer quality still require separate evaluation. See [benchmarks](../benchmarks.md) and [acceptance evidence](../plans/v0.2.0-acceptance.md).
