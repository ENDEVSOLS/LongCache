"""Offline smoke example using an explicit, deterministic fixture embedder."""

import numpy as np
from longcache import (
    BaseEmbedder,
    CacheConfig,
    EmbeddingDescriptor,
    LongCache,
    RequestContext,
    RequestScope,
)


class ExampleEmbedder(BaseEmbedder):
    dimension = 2
    descriptor = EmbeddingDescriptor("example-fixture", "v1", 2)

    def embed(self, text):
        # A constant vector demonstrates policy enforcement, not semantic quality.
        return np.array([1, 0], dtype=np.float32)


def main():
    context = RequestContext(
        scope=RequestScope("example-tenant", "example-user"),
        operation="example.answer",
        provider="fixture",
        provider_scope="fixture-v1",
        model="fixture",
        prompt_version="v1",
    )
    with LongCache(CacheConfig(backend="memory"), embedder=ExampleEmbedder()) as cache:
        assert cache.set("return within 30 days", "answer", context=context).outcome == "STORED"
        assert cache.get("return within 30 days", context=context) == "answer"
        assert cache.lookup("return within 40 days", context=context).outcome == "MISS"
        assert cache.lookup("return within 30 days").outcome == "BYPASS"
        print("Scoped hit, numeric mismatch rejection, and missing-identity bypass passed.")


if __name__ == "__main__":
    main()
