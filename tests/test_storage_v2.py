import sqlite3
import numpy as np
import pytest
from longcache.backends.sqlite import SQLiteCacheBackend
from longcache.backends.memory import InMemoryCacheBackend
from longcache.core.entry import CacheEntry


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_manifest_and_metadata_roundtrip(kind, tmp_path):
    backend = (
        InMemoryCacheBackend() if kind == "memory" else SQLiteCacheBackend(str(tmp_path / "v2.db"))
    )
    backend.ensure_manifest({"version": 2, "embedding": "model-a"})
    backend.ensure_manifest({"version": 2, "embedding": "model-a"})
    with pytest.raises(ValueError):
        backend.ensure_manifest({"version": 2, "embedding": "model-b"})
    entry = CacheEntry(
        query="q",
        vector=np.array([1, 0], dtype=np.float32),
        response={"text": "answer"},
        metadata={"format": 2, "identity": "exact"},
    )
    identifier = backend.store(entry)
    assert backend.retrieve(identifier).metadata == entry.metadata
    if kind == "sqlite":
        backend.close()
        reopened = SQLiteCacheBackend(str(tmp_path / "v2.db"))
        assert reopened.retrieve(identifier).metadata == entry.metadata
        reopened.close()


def test_legacy_database_is_unchanged(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE cache_entries (id INTEGER PRIMARY KEY, response TEXT)")
        connection.execute("INSERT INTO cache_entries VALUES (1, 'legacy')")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Legacy cache format"):
        SQLiteCacheBackend(str(path))
    assert path.read_bytes() == before
