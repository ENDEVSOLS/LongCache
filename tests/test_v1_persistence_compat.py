"""Persistence compatibility tests for LongCache v1.0.0.

Guarantees:
1. Pinned golden SHA-256 digests for partitions, exact keys, and embedding descriptors.
2. Pre-populated SQLite Format-2 databases upgrade to v1.0 with zero data loss or invalidation.
3. Pre-populated Redis Format-2 single-process and shared stores upgrade cleanly.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import pytest
import redis

from longcache import (
    CacheConfig,
    EmbeddingDescriptor,
    Instruction,
    LongCache,
    RequestContext,
    RequestScope,
)
from longcache.core.identity import canonical, digest
from tests.helpers import HashEmbedder

REDIS_URL = os.environ.get("LONGCACHE_TEST_REDIS_URL")


def test_golden_embedding_and_partition_digests():
    """Verify that canonical digest algorithms match frozen golden vectors exactly."""
    desc = EmbeddingDescriptor(
        model="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        dimension=384,
    )
    assert desc.fingerprint == "2fb4a9000f6a250ea774407f9b988f3227d54666adad8ff2628a5abb33558445"

    scope = RequestScope(tenant_id="tenant-alpha", user_id="user-42")
    ctx = RequestContext(
        scope=scope,
        operation="query.answering",
        provider="openai",
        provider_scope="global",
        model="gpt-4o",
        prompt_version="2026-11-01",
        instructions=(Instruction(role="system", content="You are a helpful assistant."),),
        generation={"temperature": 0.0, "max_tokens": 512},
        corpus_version="v2.1",
        document_versions={"doc-1": "hash-abc", "doc-2": "hash-def"},
        protected_entities=("ACME", "SecretProject"),
        language="en",
        request_shape="single-turn-text",
        codec="json-v1",
    )

    query = "What is the capital of France?"

    # 1. Private policy (tenant + user scoped)
    p_private = ctx.partition("default", "private", desc.fingerprint, "safe-reuse-v1")
    assert p_private == "v2:455397080191f06b6121a1f9878836b8864d3501a2d67b2c18c527b5469d9389"
    k_private = ctx.exact_key(query, "default", "private", desc.fingerprint, "safe-reuse-v1")
    assert k_private == "71f9e36e2b79ce20b29a28f040be5011ef2548297afb5f339469b051f60a659c"

    # 2. Tenant shared policy (tenant scoped, user stripped)
    p_tenant = ctx.partition("default", "tenant_shared", desc.fingerprint, "safe-reuse-v1")
    assert p_tenant == "v2:39ab7483632212482e6760948d5c2d46a5191be6d94de3c1d587fb063d6b5e99"
    k_tenant = ctx.exact_key(query, "default", "tenant_shared", desc.fingerprint, "safe-reuse-v1")
    assert k_tenant == "62ec4f12e0bebd3168e677107cad750f7d9485edd7e735c41642a0e34f7c022d"

    # 3. Public policy (scope stripped)
    p_public = ctx.partition("default", "public", desc.fingerprint, "safe-reuse-v1")
    assert p_public == "v2:38864ad9877379cd6074692273c684a3609aebdc3111bcc324ee30df2045e4f4"
    k_public = ctx.exact_key(query, "default", "public", desc.fingerprint, "safe-reuse-v1")
    assert k_public == "71f9ec2b3bf8da40115385a9b5c17e62fbcc5e8e2231f96d379df012ef23de9d"


def test_sqlite_format2_prepopulated_upgrade_without_dataloss(tmp_path: Path):
    """A SQLite database created with 0.6.0 Format-2 tables opens and hits cleanly in 1.0.0."""
    db_file = tmp_path / "prepopulated_v06.db"
    conn = sqlite3.connect(str(db_file))

    # Construct the exact Format-2 schema
    conn.execute(
        """
        CREATE TABLE cache_manifest (
            id INTEGER PRIMARY KEY CHECK(id=1),
            data TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE cache_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            namespace TEXT NOT NULL,
            query TEXT NOT NULL,
            vector BLOB NOT NULL,
            response TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL,
            hit_count INTEGER DEFAULT 0,
            model TEXT DEFAULT '',
            system_prompt TEXT DEFAULT '',
            temperature REAL DEFAULT -1.0,
            metadata TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE cache_tags (
            entry_id INTEGER NOT NULL REFERENCES cache_entries(id) ON DELETE CASCADE,
            tag TEXT NOT NULL
        )
        """
    )

    embedder = HashEmbedder()
    manifest_payload = {
        "format": 2,
        "cache_id": "default",
        "generation": "default",
        "embedding": asdict(embedder.descriptor),
    }
    conn.execute(
        "INSERT INTO cache_manifest (id, data) VALUES (1, ?)", (canonical(manifest_payload),)
    )

    scope = RequestScope("acme-corp", "alice")
    ctx = RequestContext(
        scope=scope,
        operation="finance.summary",
        provider="fixture",
        provider_scope="global",
        model="test-model",
        prompt_version="v1",
        codec="json-v1",
    )

    query = "annual revenue report"
    vec = embedder.embed(query)
    vec_blob = vec.tobytes()
    response_obj = {"revenue": 1000000, "status": "audited"}
    encoded_response = canonical(response_obj)
    partition = ctx.partition(
        "default", "private", embedder.descriptor.fingerprint, "safe-reuse-v1"
    )
    identity_str = canonical(
        ctx.record("default", "private", embedder.descriptor.fingerprint, "safe-reuse-v1")
    )

    entry_meta = {
        "format": 2,
        "identity": identity_str,
        "embedding": embedder.descriptor.fingerprint,
        "payload": digest(response_obj),
    }

    conn.execute(
        """
        INSERT INTO cache_entries (id, namespace, query, vector, response, created_at, expires_at, hit_count, metadata)
        VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            partition,
            query,
            vec_blob,
            encoded_response,
            time.time(),
            time.time() + 3600,
            3,
            json.dumps(entry_meta),
        ),
    )
    conn.execute("INSERT INTO cache_tags (entry_id, tag) VALUES (1, 'finance')")
    conn.execute("INSERT INTO cache_tags (entry_id, tag) VALUES (1, 'annual')")
    conn.commit()
    conn.close()

    # Now open with LongCache 1.0.0
    config = CacheConfig(backend="sqlite", db_path=str(db_file))
    with LongCache(config, embedder=embedder) as cache:
        # Lookup must HIT with exact response
        lookup = cache.lookup(query, context=ctx)
        assert lookup.outcome == "HIT"
        assert lookup.entry_id == 1
        assert lookup.similarity is not None and lookup.similarity >= 0.99
        assert lookup.response == {"revenue": 1000000, "status": "audited"}

        # Stats must report 1 entry
        stats = cache.stats(context=ctx)
        assert stats["total_entries"] == 1

        # Tag-based invalidation must evict the pre-populated entry
        invalidated = cache.invalidate_by_tag("finance", context=ctx)
        assert invalidated == 1

        # Subsequent lookup must be a MISS
        lookup_after = cache.lookup(query, context=ctx)
        assert lookup_after.outcome == "MISS"


@pytest.mark.skipif(not REDIS_URL, reason="Live Redis service required")
def test_redis_format2_prepopulated_upgrade_without_dataloss():
    """A Redis store populated with 0.6.0 Format-2 keys opens and serves hits in 1.0.0."""
    client = redis.Redis.from_url(REDIS_URL or "redis://localhost:16379/0")
    cache_id = "v06-upg-" + uuid4().hex[:8]
    generation = "gen1"
    root_key = f"longcache:v2:{cache_id}:{generation}"

    embedder = HashEmbedder()
    manifest_data = {
        "format": 2,
        "cache_id": cache_id,
        "generation": generation,
        "embedding": asdict(embedder.descriptor),
    }
    client.set(f"{root_key}:manifest", canonical(manifest_data))

    scope = RequestScope("redis-tenant", "bob")
    ctx = RequestContext(
        scope=scope,
        operation="search.query",
        provider="fixture",
        provider_scope="global",
        model="demo-model",
        prompt_version="v1",
        codec="json-v1",
    )

    query = "cached query in redis"
    vec = embedder.embed(query)
    entry_id = 101
    response_obj = {"redis_key": "stored_in_v06"}
    encoded_response = canonical(response_obj)
    partition = ctx.partition(
        "default", "private", embedder.descriptor.fingerprint, "safe-reuse-v1"
    )
    identity_str = canonical(
        ctx.record("default", "private", embedder.descriptor.fingerprint, "safe-reuse-v1")
    )

    entry_meta = {
        "format": 2,
        "identity": identity_str,
        "embedding": embedder.descriptor.fingerprint,
        "payload": digest(response_obj),
    }

    # Populate exact Redis hash for entry
    import base64

    entry_key = f"{root_key}:entry:{entry_id}"
    client.hset(
        entry_key,
        mapping={
            "id": entry_id,
            "namespace": partition,
            "query": query,
            "response": encoded_response,
            "created_at": str(time.time()),
            "hit_count": "5",
            "vector": base64.b64encode(vec.tobytes()).decode("ascii"),
            "metadata": json.dumps(entry_meta),
        },
    )
    client.sadd(f"{root_key}:entries", entry_id)
    client.sadd(f"{root_key}:ns:{partition}", entry_id)
    client.sadd(f"{root_key}:tag:v06-tag", entry_id)
    client.sadd(f"{root_key}:namespaces", partition)

    config = CacheConfig(
        backend="redis",
        redis_url=REDIS_URL or "redis://localhost:16379/0",
        cache_id=cache_id,
        generation=generation,
    )

    with LongCache(config, embedder=embedder) as cache:
        lookup = cache.lookup(query, context=ctx)
        assert lookup.outcome == "HIT"
        assert lookup.entry_id == entry_id
        assert lookup.response == {"redis_key": "stored_in_v06"}

        # Clean up
        cache.clear(operator=True)
