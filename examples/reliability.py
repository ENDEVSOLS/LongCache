"""Offline demonstration: 20 callers, one origin, explicit scope and safe cleanup."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import threading
import time

import numpy as np
from longcache import (
    BaseEmbedder,
    CacheConfig,
    EmbeddingDescriptor,
    LongCache,
    RequestContext,
    RequestScope,
)


class DemoEmbedder(BaseEmbedder):
    dimension = 384
    descriptor = EmbeddingDescriptor("offline-demo", "sha256-v1", 384)

    def embed(self, text):
        seed = int.from_bytes(sha256(text.encode()).digest()[:8], "big")
        vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
        return vector / np.linalg.norm(vector)


def main():
    context = RequestContext(
        scope=RequestScope("demo-tenant", "demo-user"),
        operation="support.answer",
        provider="offline-fixture",
        provider_scope="demo-v1",
        model="fixture",
        prompt_version="v1",
    )
    barrier, lock = threading.Barrier(20), threading.Lock()
    calls = 0

    def origin():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.1)  # Stand-in for retrieval or an LLM; no external request.
        return {"answer": "Returns are accepted within the documented policy."}

    with LongCache(
        CacheConfig(backend="memory", ttl_seconds=0), embedder=DemoEmbedder(), context=context
    ) as cache:

        def request(_):
            barrier.wait(timeout=5)
            return cache.get_or_compute("What is the return policy?", origin)

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(request, range(20)))
        assert len(results) == 20 and calls == 1
        assert all(answer == results[0] for answer in results)
        print(f"Requests: {len(results)}; origin executions: {calls}")
        print(
            f"Active flights after completion: {cache.metrics(operator=True)['resources']['active_flights']}"
        )


if __name__ == "__main__":
    main()
