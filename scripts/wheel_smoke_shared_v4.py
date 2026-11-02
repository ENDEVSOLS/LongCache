"""Exercise the installed shared Redis path without models or provider calls."""

from importlib.metadata import version
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import longcache
from longcache import (
    BaseEmbedder,
    CacheConfig,
    EmbeddingDescriptor,
    LongCache,
    RequestContext,
    RequestScope,
)


class FixtureEmbedder(BaseEmbedder):
    dimension = 2
    descriptor = EmbeddingDescriptor("installed-wheel-shared", "v1", 2)

    def embed(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def main():
    assert version("longcache") == longcache.__version__
    assert "site-packages" in str(Path(longcache.__file__).resolve()), longcache.__file__
    redis_url = os.environ["LONGCACHE_TEST_REDIS_URL"]
    context = RequestContext(
        scope=RequestScope("wheel-tenant", "wheel-user"),
        operation="wheel.answer",
        provider="fixture",
        provider_scope="v1",
        model="fixture",
        prompt_version="v1",
    )
    config = CacheConfig(
        backend="redis",
        deployment_mode="shared",
        redis_url=redis_url,
        cache_id="wheel-" + uuid4().hex,
        generation="shared-v1",
        ttl_seconds=0,
    )
    first = LongCache(config, embedder=FixtureEmbedder())
    second = LongCache(config, embedder=FixtureEmbedder())
    try:
        assert first.set("question", "old", context=context, tags=["document"]).outcome == "STORED"
        assert second.get("question", context=context) == "old"
        assert first.invalidate_by_tag("document", context=context) == 1
        assert second.get("question", context=context) is None
        assert second.set("question", "new", context=context, tags=["document"]).outcome == "STORED"
        assert first.get("question", context=context) == "new"
        print("Installed wheel: shared Redis write, invalidation, and refill passed")
    finally:
        first.clear(operator=True)
        first.close()
        second.close()


if __name__ == "__main__":
    main()
