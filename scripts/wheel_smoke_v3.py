"""Exercise an installed artifact without downloading models or calling providers."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
import tempfile
import threading
import time

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


class Embedder(BaseEmbedder):
    dimension = 2
    descriptor = EmbeddingDescriptor("wheel-fixture", "v1", 2)

    def embed(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)


def main():
    assert version("longcache") == longcache.__version__ == "0.3.0"
    assert "site-packages" in str(Path(longcache.__file__).resolve()), longcache.__file__
    context = RequestContext(
        scope=RequestScope("wheel", "user"),
        operation="smoke.answer",
        provider="fixture",
        provider_scope="v1",
        model="fixture",
        prompt_version="v1",
    )
    with tempfile.TemporaryDirectory() as directory:
        config = CacheConfig(db_path=str(Path(directory) / "cache.db"), ttl_seconds=0)
        cache = LongCache(config, embedder=Embedder(), context=context)
        calls, lock, barrier = [], threading.Lock(), threading.Barrier(20)

        def origin():
            with lock:
                calls.append(1)
            time.sleep(0.1)
            return {"answer": "approved"}

        def request(_):
            barrier.wait(timeout=5)
            return cache.get_or_compute("policy", origin, namespace="sync")

        with cache:
            with ThreadPoolExecutor(max_workers=20) as pool:
                results = list(pool.map(request, range(20)))
            assert results == [{"answer": "approved"}] * 20 and len(calls) == 1
        assert cache.metrics(operator=True)["resources"]["sqlite_connections"] == 0

        async def async_probe():
            calls = []
            async with LongCache(config, embedder=Embedder(), context=context) as cache:

                async def origin():
                    calls.append(1)
                    await asyncio.sleep(0.1)
                    return {"answer": "approved"}

                results = await asyncio.gather(
                    *[cache.aget_or_compute("policy", origin, namespace="async") for _ in range(20)]
                )
                assert results == [{"answer": "approved"}] * 20 and len(calls) == 1
            assert cache.metrics(operator=True)["resources"]["sqlite_connections"] == 0

        asyncio.run(async_probe())
        with LongCache(config, embedder=Embedder(), context=context) as reopened:
            assert reopened.get("policy", namespace="sync") == {"answer": "approved"}
            assert reopened.get("policy", namespace="async") == {"answer": "approved"}
    print("Installed v0.3.0: sync/async singleflight, SQLite restart and resource cleanup passed")


if __name__ == "__main__":
    main()
