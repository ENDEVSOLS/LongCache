"""Real-embedding application smoke evaluation for changing-document RAG.

Documents and answers are authored fixtures, and generation is extractive. This
checks application wiring and freshness, not live-provider answer accuracy.
"""

import asyncio
from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4
from examples.versioned_rag import DocumentSnapshot, answer
from longcache import CacheConfig, LongCache
from longcache.core.embedder import LocalEmbedder


async def evaluate(output=Path("evaluation/reports/versioned_rag_current.json")):
    embedder = LocalEmbedder(revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for backend in ("memory", "sqlite", "redis"):
            for index in ("flat", "hnsw"):
                config = CacheConfig(
                    backend=backend,
                    faiss_index_type=index,
                    db_path=str(Path(directory) / f"{backend}-{index}.db"),
                    redis_url=os.environ.get(
                        "LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0"
                    ),
                    cache_id="rag-evaluation-" + uuid4().hex,
                    ttl_seconds=0,
                )
                with LongCache(config, embedder=embedder) as cache:
                    try:
                        question = "What is the return window?"
                        old = DocumentSnapshot("1", "Returns accepted within 30 days.")
                        new = DocumentSnapshot("2", "Returns accepted within 14 days.")
                        assert await answer(cache, question, old) == old.text
                        assert await answer(cache, question, old) == old.text
                        assert await answer(cache, question, new) == new.text
                        assert await answer(cache, question, new) == new.text
                        assert cache.get(question, context=new.context("other", "alice")) is None
                        assert (
                            cache.invalidate_by_tag(
                                old.document_id, context=old.context("shop", "alice")
                            )
                            == 1
                        )
                        assert await answer(cache, question, new) == new.text
                        rows.append({"backend": backend, "index": index, "passed": True})
                    finally:
                        cache.clear(operator=True)
    report = {
        "protocol": "changing-document-rag-v1",
        "embedding": asdict(embedder.descriptor),
        "scenarios": rows,
        "passed": all(row["passed"] for row in rows),
        "limitations": [
            "Authored document fixtures with extractive generation; no external LLM was invoked.",
            "Customer dataset review, answer quality and hosted CI are separate release gates.",
        ],
    }
    path = output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/reports/versioned_rag_current.json")
    )
    asyncio.run(evaluate(parser.parse_args().output))
