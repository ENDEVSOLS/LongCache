"""Run with PYTHONPATH set to a v0.1.1 archive; synthetic semantic cases only."""

import json
from pathlib import Path
import sys
import time
import numpy as np
from longcache import CacheConfig, LongCache


def main():
    cases = json.loads(Path(sys.argv[1]).read_text())
    vectors = np.load(sys.argv[2])
    texts = json.loads(Path(sys.argv[3]).read_text())
    lookup = dict(zip(texts, vectors["vectors"]))

    class Embedder:
        def embed(self, text):
            return lookup[text]

    cache = LongCache(
        CacheConfig(backend="memory", ttl_seconds=0, similarity_threshold=float(sys.argv[4]))
    )
    cache.embedder = Embedder()
    rows = []
    for case in cases:
        if case["group"] not in ("safe", "near"):
            continue
        cache.clear()
        cache.set(case["seed"], "synthetic answer")
        start = time.perf_counter()
        response = cache.get(case["query"])
        rows.append(
            {
                "id": case["id"],
                "hit": response is not None,
                "latency_ms": (time.perf_counter() - start) * 1000,
            }
        )
    cache.close()
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
