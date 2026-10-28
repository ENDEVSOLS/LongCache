"""Cold-cache upgrade, interrupted initialization and real legacy rollback."""

from io import BytesIO
import os
import sqlite3
import subprocess
import sys
import tarfile
import textwrap
from uuid import uuid4
import pytest
from longcache import CacheConfig, LongCache
from longcache.backends.sqlite import SQLiteCacheBackend
from tests.test_embedding_contract_v2 import TinyEmbedder
from tests.test_identity_v2 import context


def test_initialization_error_rolls_back_all_schema_changes(monkeypatch, tmp_path):
    path = tmp_path / "interrupted.db"
    original = sqlite3.connect

    class Interrupted(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("CREATE TABLE IF NOT EXISTS cache_manifest"):
                raise RuntimeError("injected initialization interruption")
            return super().execute(sql, *args, **kwargs)

    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: original(*a, **(kw | {"factory": Interrupted}))
    )
    with pytest.raises(RuntimeError, match="interruption"):
        SQLiteCacheBackend(str(path))
    with original(path) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        assert tables == []
    monkeypatch.setattr(sqlite3, "connect", original)
    backend = SQLiteCacheBackend(str(path))
    backend.close()


def test_incompatible_partial_schema_is_refused_without_modification(tmp_path):
    path = tmp_path / "partial.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE cache_entries (id INTEGER PRIMARY KEY, metadata TEXT)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        SQLiteCacheBackend(str(path))
    assert path.read_bytes() == before


def test_missing_manifest_in_nonempty_store_cannot_be_recreated(tmp_path):
    path = tmp_path / "cache.db"
    cfg = CacheConfig(db_path=str(path), ttl_seconds=0)
    with LongCache(cfg, embedder=TinyEmbedder()) as cache:
        cache.set("policy", "approved", context=context())
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE cache_manifest")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="manifest"):
        LongCache(cfg, embedder=TinyEmbedder())
    assert path.read_bytes() == before


@pytest.fixture(scope="module")
def legacy_source(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v011-rollback")
    result = subprocess.run(["git", "archive", "8a47ebd", "src"], capture_output=True)
    if result.returncode:
        pytest.skip("Legacy runtime rollback requires repository history; CI fetches it explicitly")
    with tarfile.open(fileobj=BytesIO(result.stdout)) as archive:
        archive.extractall(directory, filter="data")
    return directory


def legacy_call(root, path, operation):
    code = textwrap.dedent("""
        import sys,numpy as np,longcache
        from longcache import LongCache,CacheConfig
        assert longcache.__version__ == '0.1.1'
        class Embedder:
            def embed(self,text):return np.array([1,0],dtype=np.float32)
        with LongCache(CacheConfig(db_path=sys.argv[1],ttl_seconds=0)) as cache:
            cache.embedder=Embedder()
            if sys.argv[2]=='write':cache.set('policy','legacy answer')
            else:assert cache.get('policy')=='legacy answer'
    """)
    subprocess.run(
        [sys.executable, "-c", code, str(path), operation],
        cwd=root,
        env=os.environ | {"PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )


def test_v011_upgrade_and_rollback_with_actual_legacy_runtime(legacy_source, tmp_path):
    old = tmp_path / "v011.db"
    legacy_call(legacy_source, old, "write")
    backup = tmp_path / "backup.db"
    with sqlite3.connect(old) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    before = old.read_bytes()
    with pytest.raises(ValueError, match="Legacy cache format"):
        LongCache(CacheConfig(db_path=str(old)), embedder=TinyEmbedder())
    assert old.read_bytes() == before
    with LongCache(CacheConfig(db_path=str(tmp_path / "v02.db")), embedder=TinyEmbedder()) as cache:
        assert cache.lookup("policy", context=context()).outcome == "MISS"
        cache.set("policy", "v02 answer", context=context())
    restored = tmp_path / "restored-v011.db"
    with sqlite3.connect(backup) as source, sqlite3.connect(restored) as target:
        source.backup(target)
    legacy_call(legacy_source, restored, "read")
    with sqlite3.connect(old) as conn:
        assert "metadata" not in {
            row[1] for row in conn.execute("PRAGMA table_info(cache_entries)")
        }


@pytest.mark.asyncio
async def test_real_redis_generation_manifest_loss_fails_closed():
    import redis

    url = os.environ.get("LONGCACHE_TEST_REDIS_URL", "redis://localhost:16379/0")
    client = redis.Redis.from_url(url, socket_connect_timeout=1)
    try:
        client.ping()
    except redis.RedisError:
        pytest.skip("Real Redis service required")
    cfg = CacheConfig(
        backend="redis", redis_url=url, cache_id="recovery-" + uuid4().hex, ttl_seconds=0
    )
    with LongCache(cfg, embedder=TinyEmbedder()) as cache:
        await cache.aset("policy", "answer", context=context())
        root = cache.backend.ROOT
        cache.backend._client.delete(root + ":manifest")
    try:
        with pytest.raises(ValueError, match="manifest"):
            LongCache(cfg, embedder=TinyEmbedder())
    finally:
        for key in client.scan_iter(root + ":*"):
            client.delete(key)
        client.close()


def test_process_termination_rolls_back_schema_initialization(tmp_path):
    """Terminate after schema creation but before manifest creation/commit."""
    path = tmp_path / "killed.db"
    code = textwrap.dedent("""
        import sqlite3, sys, time
        from longcache.backends.sqlite import SQLiteCacheBackend
        original = sqlite3.connect
        class Interrupted(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):
                if sql.startswith('CREATE TABLE IF NOT EXISTS cache_manifest'):
                    print('schema-created', flush=True)
                    time.sleep(60)
                return super().execute(sql, *args, **kwargs)
        sqlite3.connect = lambda *a, **kw: original(*a, **(kw | {'factory': Interrupted}))
        SQLiteCacheBackend(sys.argv[1])
    """)
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, text=True
    )
    try:
        import select

        assert select.select([process.stdout], [], [], 10)[0], "initializer did not reach schema"
        assert process.stdout.readline().strip() == "schema-created"
    finally:
        process.terminate()
        process.wait(timeout=10)
        process.stdout.close()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
    backend = SQLiteCacheBackend(str(path))
    backend.close()
