"""An actual v0.2 runtime can read entries before and after a v0.3 upgrade."""

from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
from uuid import uuid4

import numpy as np
import pytest
from longcache import BaseEmbedder, CacheConfig, EmbeddingDescriptor, LongCache
from tests.helpers import trusted_context


class CompatibleEmbedder(BaseEmbedder):
    dimension = 2
    descriptor = EmbeddingDescriptor("compat-fixture", "v1", 2)

    def embed(self, text):
        return np.array([1, 0], dtype=np.float32)


@pytest.fixture(scope="module")
def v02_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v02-source")
    archive = subprocess.run(["git", "archive", "6a059e7", "src"], capture_output=True, timeout=10)
    if archive.returncode:
        pytest.skip("Actual v0.2 compatibility requires repository history")
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as source:
        source.extractall(directory, filter="data")
    return directory


@pytest.mark.parametrize("backend", ["sqlite", "redis"])
@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_actual_v02_upgrade_and_rollback(v02_source, tmp_path, backend, index):
    if backend == "redis" and not os.environ.get("LONGCACHE_TEST_REDIS_URL"):
        pytest.skip("Real Redis service required")
    config = CacheConfig(
        backend=backend,
        faiss_index_type=index,
        ttl_seconds=0,
        db_path=str(tmp_path / "compatible.db"),
        generation=uuid4().hex,
        redis_url=os.environ.get("LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0"),
    )
    code = textwrap.dedent("""
        import json, sys, numpy as np, longcache
        from longcache import BaseEmbedder, EmbeddingDescriptor, CacheConfig, LongCache, RequestContext, RequestScope
        assert longcache.__version__ == "0.2.0"
        class Embedder(BaseEmbedder):
            dimension = 2
            descriptor = EmbeddingDescriptor("compat-fixture", "v1", 2)
            def embed(self, text):
                return np.array([1, 0], dtype=np.float32)
        ctx = RequestContext(scope=RequestScope("test-tenant", "test-user"), operation="test.answer",
            provider="fixture", provider_scope="fixture-v1", model="fixture-model", prompt_version="v1")
        with LongCache(CacheConfig.from_dict(json.loads(sys.argv[1])), embedder=Embedder(), context=ctx) as cache:
            if sys.argv[2] == "write":
                assert cache.set("policy 1", {"answer": 1}).outcome == "STORED"
            else:
                assert cache.get("policy 1") == {"answer": 1}
                assert cache.get("policy 2") == {"answer": 2}
    """)
    # Only v0.2-supported fields are sent to the archived runtime.
    import json

    settings = {
        key: getattr(config, key)
        for key in (
            "backend",
            "faiss_index_type",
            "ttl_seconds",
            "db_path",
            "generation",
            "redis_url",
        )
    }

    def legacy(operation):
        subprocess.run(
            [sys.executable, "-c", code, json.dumps(settings), operation],
            check=True,
            cwd=tmp_path,
            env=os.environ | {"PYTHONPATH": str(Path(v02_source) / "src")},
            timeout=15,
        )

    legacy("write")
    with LongCache(config, embedder=CompatibleEmbedder(), context=trusted_context()) as current:
        try:
            assert current.get_or_compute("policy 1", lambda: pytest.fail("must reuse v0.2")) == {
                "answer": 1
            }
            assert current.set("policy 2", {"answer": 2}).outcome == "STORED"
            legacy("read")
        finally:
            current.clear(operator=True)
