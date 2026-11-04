"""Verify Format-2 persistence compatibility between v0.4.0 and v0.5.0."""

from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import textwrap
from uuid import uuid4

from click.testing import CliRunner
import numpy as np
import pytest
from longcache import (
    BaseEmbedder,
    CacheConfig,
    CacheEntry,
    EmbeddingDescriptor,
    LongCache,
)
from longcache.cli import main as cli_main
from tests.helpers import trusted_context


class CompatibleEmbedder(BaseEmbedder):
    dimension = 2
    descriptor = EmbeddingDescriptor("compat-fixture", "v1", 2)

    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_cache_entry_from_dict_ignores_legacy_keys():
    """CacheEntry.from_dict() gracefully ignores legacy model/system_prompt/temperature keys."""
    vec = np.array([1.0, 0.0], dtype=np.float32)
    legacy_blob = {
        "id": 42,
        "namespace": "v2:abc123",
        "query": "what is policy?",
        "vector": vec.tobytes(),
        "response": {"answer": "ok"},
        "created_at": 1700000000.0,
        "expires_at": None,
        "hit_count": 3,
        "model": "legacy-gpt-4",
        "system_prompt": "legacy-system",
        "temperature": 0.7,
        "tags": ["doc-1"],
        "metadata": {"format": 2, "identity": "id", "embedding": "fp", "payload": "dg"},
    }
    entry = CacheEntry.from_dict(legacy_blob, vector_dim=2)
    assert entry.id == 42
    assert entry.namespace == "v2:abc123"
    assert entry.query == "what is policy?"
    assert entry.response == {"answer": "ok"}
    assert entry.hit_count == 3
    assert entry.tags == ["doc-1"]
    assert entry.metadata["format"] == 2
    assert not hasattr(entry, "model")
    assert not hasattr(entry, "system_prompt")
    assert not hasattr(entry, "temperature")

    exported = entry.to_dict()
    assert "model" not in exported
    assert "system_prompt" not in exported
    assert "temperature" not in exported


@pytest.fixture(scope="module")
def v04_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v04-source")
    archive = subprocess.run(["git", "archive", "d2275ff", "src"], capture_output=True, timeout=10)
    if archive.returncode:
        pytest.skip("Actual v0.4 compatibility requires repository history")
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as source:
        source.extractall(directory, filter="data")
    return directory


@pytest.mark.parametrize("backend", ["sqlite", "redis"])
@pytest.mark.parametrize("index", ["flat", "hnsw"])
def test_actual_v04_upgrade_and_cli_export(v04_source, tmp_path, backend, index):
    if backend == "redis" and not os.environ.get("LONGCACHE_TEST_REDIS_URL"):
        pytest.skip("Real Redis service required")
    db_file = tmp_path / "compatible_v04.db"
    config = CacheConfig(
        backend=backend,
        faiss_index_type=index,
        ttl_seconds=0,
        db_path=str(db_file),
        generation="default" if backend == "sqlite" else uuid4().hex,
        redis_url=os.environ.get("LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0"),
    )
    code = textwrap.dedent("""
        import json, sys, numpy as np, longcache
        from longcache import BaseEmbedder, EmbeddingDescriptor, CacheConfig, LongCache, RequestContext, RequestScope
        assert longcache.__version__ == "0.4.0"
        class Embedder(BaseEmbedder):
            dimension = 2
            descriptor = EmbeddingDescriptor("compat-fixture", "v1", 2)
            def embed(self, text):
                return np.array([1.0, 0.0], dtype=np.float32)
        ctx = RequestContext(scope=RequestScope("test-tenant", "test-user"), operation="test.answer",
            provider="fixture", provider_scope="fixture-v1", model="fixture-model", prompt_version="v1")
        with LongCache(CacheConfig.from_dict(json.loads(sys.argv[1])), embedder=Embedder(), context=ctx) as cache:
            if sys.argv[2] == "write":
                assert cache.set("policy 1", {"answer": 1}, tags=["v04"]).outcome == "STORED"
            else:
                assert cache.get("policy 1") == {"answer": 1}
                assert cache.get("policy 2") == {"answer": 2}
    """)
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

    def run_v04(operation: str) -> None:
        subprocess.run(
            [sys.executable, "-c", code, json.dumps(settings), operation],
            check=True,
            cwd=tmp_path,
            env=os.environ | {"PYTHONPATH": str(Path(v04_source) / "src")},
            timeout=15,
        )

    run_v04("write")
    ctx = trusted_context()
    with LongCache(config, embedder=CompatibleEmbedder(), context=ctx) as current:
        try:
            # 1 & 2. Manifest validation succeeds without reset and warm lookup/get_or_compute hit
            decision = current.lookup("policy 1")
            assert decision.outcome == "HIT"
            assert decision.response == {"answer": 1}
            assert current.get_or_compute(
                "policy 1", lambda: pytest.fail("must reuse v0.4 entry")
            ) == {"answer": 1}

            # 3. entries() round-trips cleanly with v2:<sha256> partition digest
            entries = current.entries()
            assert len(entries) == 1
            assert entries[0].namespace.startswith("v2:")
            assert entries[0].tags == ["v04"]

            # Write new entry in v0.5.0 and verify v0.4.0 rollback can read it
            assert current.set("policy 2", {"answer": 2}).outcome == "STORED"
            run_v04("read")

            # Verify CLI export round-trips cleanly for SQLite
            if backend == "sqlite":
                runner = CliRunner()
                result = runner.invoke(
                    cli_main,
                    ["export", "--db-path", str(db_file), "--operator"],
                )
                assert result.exit_code == 0, result.output
                exported_records = json.loads(result.output)
                assert len(exported_records) == 2
        finally:
            if backend == "redis":
                current.clear(operator=True)
