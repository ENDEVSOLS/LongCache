"""Run a versioned RAG freshness example with two independent Redis caches.

This uses authored documents and an extractive fixture, not a live LLM.
Set LONGCACHE_TEST_REDIS_URL to an isolated Redis service before running.
"""

import asyncio
import os
from uuid import uuid4

from safe_reuse import ExampleEmbedder
from versioned_rag import DocumentSnapshot
from longcache import CacheConfig, LongCache


async def answer(cache, question, snapshot):
    context = snapshot.context("shop", "alice")

    async def generate():
        return snapshot.text

    return await cache.aget_or_compute(
        question,
        generate,
        context=context,
        tags=(snapshot.document_id,),
    )


async def main():
    redis_url = os.environ["LONGCACHE_TEST_REDIS_URL"]
    config = CacheConfig(
        backend="redis",
        deployment_mode="shared",
        redis_url=redis_url,
        cache_id="rag-example-" + uuid4().hex,
        generation="v04",
    )
    old = DocumentSnapshot("revision-1", "Returns accepted within 30 days.")
    new = DocumentSnapshot("revision-2", "Returns accepted within 14 days.")
    question = "What is the return window?"
    async with LongCache(config, embedder=ExampleEmbedder()) as first:
        async with LongCache(config, embedder=ExampleEmbedder()) as second:
            assert await answer(first, question, old) == old.text
            assert await answer(second, question, old) == old.text
            assert await answer(second, question, new) == new.text
            assert await answer(first, question, new) == new.text
            assert (
                first.invalidate_by_tag(old.document_id, context=old.context("shop", "alice")) == 1
            )
            assert second.get(question, context=old.context("shop", "alice")) is None
            first.clear(operator=True)
    print("Two-instance versioned RAG freshness check passed.")


if __name__ == "__main__":
    asyncio.run(main())
