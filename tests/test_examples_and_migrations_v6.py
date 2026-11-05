"""Migration and format compatibility tests for v0.6.0 qualification."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4

import pytest
import redis

from longcache import CacheConfig, LongCache, RequestContext, RequestScope
from tests.helpers import HashEmbedder, trusted_context

REDIS_URL = os.environ.get("LONGCACHE_TEST_REDIS_URL")


def test_format1_sqlite_database_fails_closed(tmp_path: Path) -> None:
    """Format-1 database (without metadata column) fails closed on initialization."""
    db_file = tmp_path / "format1_legacy.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE cache_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            query TEXT NOT NULL,
            vector BLOB,
            response TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL,
            hit_count INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO cache_entries (namespace, query, response, created_at) VALUES ('default', 'test', 'ans', 1.0)"
    )
    conn.commit()
    conn.close()

    config = CacheConfig(backend="sqlite", db_path=str(db_file))
    with pytest.raises(ValueError, match="Legacy cache format"):
        LongCache(config, embedder=HashEmbedder())


def test_nonempty_sqlite_without_manifest_fails_closed(tmp_path: Path) -> None:
    """Non-empty SQLite database without cache_manifest table fails closed."""
    db_file = tmp_path / "missing_manifest.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE cache_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            query TEXT NOT NULL,
            vector BLOB,
            response TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL,
            hit_count INTEGER DEFAULT 0,
            model TEXT DEFAULT '',
            system_prompt TEXT DEFAULT '',
            temperature REAL DEFAULT -1.0,
            metadata TEXT DEFAULT '{}'
        )
        """
    )
    conn.execute(
        "INSERT INTO cache_entries (namespace, query, response, created_at, metadata) VALUES ('default', 'test', 'ans', 1.0, '{}')"
    )
    conn.commit()
    conn.close()

    config = CacheConfig(backend="sqlite", db_path=str(db_file))
    with pytest.raises(ValueError, match="Missing storage manifest in nonempty cache"):
        LongCache(config, embedder=HashEmbedder())


@pytest.mark.skipif(not REDIS_URL, reason="Real Redis service required")
def test_format1_redis_keyspace_isolation_and_dirty_manifest_rejection() -> None:
    """Redis Format-2 keyspace is isolated from legacy v0.1 unprefixed keys."""
    client = redis.Redis.from_url(REDIS_URL or "redis://localhost:6379/0")
    cache_id = "compat-" + uuid4().hex

    # Plant a legacy v0.1 unprefixed entry
    client.set("longcache:entry:99999", "legacy-unprefixed-data")

    config = CacheConfig(
        backend="redis",
        redis_url=REDIS_URL or "redis://localhost:6379/0",
        cache_id=cache_id,
        generation="v06",
    )
    with LongCache(config, embedder=HashEmbedder()) as cache:
        context = trusted_context()
        assert cache.set("test-query", "ans", context=context).outcome == "STORED"
        assert cache.lookup("test-query", context=context).outcome == "HIT"

        # Verify Format-2 backend never enumerated or touched legacy key
        assert client.get("longcache:entry:99999") == b"legacy-unprefixed-data"
        cache.clear(operator=True)

    client.delete("longcache:entry:99999")


@pytest.mark.parametrize("codec", ["json-v1", "openai-dict-v1", "openai-sdk-v1"])
def test_format2_codecs_roundtrip_compatibility(tmp_path: Path, codec: str) -> None:
    """Test that json-v1, openai-dict-v1, and openai-sdk-v1 codecs roundtrip in Format-2."""
    db_file = tmp_path / f"codec_{codec}.db"
    config = CacheConfig(backend="sqlite", db_path=str(db_file))

    context = RequestContext(
        scope=RequestScope("codec-tenant", "codec-user"),
        operation="test.codec",
        provider="fixture",
        provider_scope="fixture-v1",
        model="demo",
        prompt_version="v1",
        codec=codec,  # type: ignore[arg-type]
    )

    if codec == "openai-sdk-v1":
        pytest.importorskip("openai")
        from openai.types.chat import ChatCompletion

        expected: Any = ChatCompletion.model_validate(
            {
                "id": "chatcmpl-sdk",
                "created": 1234567890,
                "model": "gpt-4o",
                "object": "chat.completion",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"role": "assistant", "content": "sdk-response"},
                    }
                ],
            }
        )
    elif codec == "openai-dict-v1":
        expected = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                }
            ],
        }
    else:
        expected = {"answer": "pure-json", "confidence": 0.99}

    with LongCache(config, embedder=HashEmbedder()) as cache:
        decision = cache.set("greeting", expected, context=context)
        assert decision.outcome == "STORED"

        lookup = cache.lookup("greeting", context=context)
        assert lookup.outcome == "HIT"
        if codec == "openai-sdk-v1":
            from openai.types.chat import ChatCompletion

            assert isinstance(lookup.response, ChatCompletion)
            assert lookup.response.choices[0].message.content == "sdk-response"
        else:
            assert lookup.response == expected
