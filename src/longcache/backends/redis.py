"""
Redis cache backend for LongCache.

Uses Redis hashes for namespace-scoped storage and key expiry.
LongCache's FAISS index is local to one process and is not synchronized
across instances by this backend.
"""

from __future__ import annotations

import base64
import json
import time
import threading
from typing import Optional

import numpy as np

from longcache.backends.base import AbstractCacheBackend
from longcache.core.entry import CacheEntry


class RedisCacheBackend(AbstractCacheBackend):
    """
    Redis-based cache backend for distributed deployments.

    Stores each cache entry as a Redis hash with fields for query,
    vector (base64-encoded), response (JSON), and metadata.
    Namespaces are isolated using Redis key prefixes.

    Requires the 'redis' extra: uv pip install longcache[redis]

    Usage:
        backend = RedisCacheBackend(redis_url="redis://localhost:6379/0")
        entry_id = backend.store(entry)
        result = backend.retrieve(entry_id)
    """

    ROOT = "longcache:v2:default:default"
    KEY_PREFIX = ROOT + ":entry"
    NAMESPACE_SET = ROOT + ":namespaces"
    ID_COUNTER = ROOT + ":id_counter"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        max_entries: int = 10_000,
        cache_id: str = "default",
        generation: str = "default",
    ):
        """
        Initialize the Redis backend.

        Args:
            redis_url: Redis connection URL.
            max_entries: Maximum entries per namespace.
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis backend requires the 'redis' extra. "
                "Install with: uv pip install longcache[redis]"
            )

        self.ROOT = f"longcache:v2:{cache_id}:{generation}"
        self.KEY_PREFIX = self.ROOT + ":entry"
        self.NAMESPACE_SET = self.ROOT + ":namespaces"
        self.ID_COUNTER = self.ROOT + ":id_counter"
        self._redis_url = redis_url
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)

    def read_manifest(self):
        data = self._client.get(self.ROOT + ":manifest")
        return json.loads(data) if data else None

    def ensure_manifest(self, manifest: dict) -> None:
        from longcache.core.identity import canonical

        encoded = canonical(manifest)
        key = self.ROOT + ":manifest"
        if self._client.get(key) is None and self.list_all_entries():
            raise ValueError("Missing storage manifest in nonempty cache; use a new generation")
        self._client.set(key, encoded, nx=True)
        if self._client.get(key) != encoded:
            raise ValueError("Incompatible embedding or storage generation; use a new generation")

    def _entry_key(self, entry_id: int) -> str:
        """Get the Redis key for a cache entry."""
        return f"{self.KEY_PREFIX}:{entry_id}"

    def _namespace_key(self, namespace: str) -> str:
        """Get the Redis key for a namespace's entry set."""
        return f"{self.ROOT}:ns:{namespace}"

    def close(self) -> None:
        """Release Redis connections owned by this backend."""
        self._client.close()

    def _vector_to_str(self, vector: np.ndarray) -> str:
        """Encode a numpy vector as base64 string for Redis storage."""
        return base64.b64encode(vector.tobytes()).decode("ascii")

    def _str_to_vector(self, vector_str: str) -> np.ndarray:
        """Decode a base64 string back to a numpy vector."""
        return np.frombuffer(base64.b64decode(vector_str), dtype=np.float32).copy()

    def store(self, entry: CacheEntry) -> int:
        """
        Store a cache entry in Redis.

        Args:
            entry: The CacheEntry to store.

        Returns:
            The auto-generated integer ID.
        """
        with self._lock:
            entry_id = self._client.incr(self.ID_COUNTER)

            vector_str = self._vector_to_str(entry.vector) if entry.vector is not None else ""
            response_json = json.dumps(entry.response, allow_nan=False)

            entry_data = {
                "id": entry_id,
                "namespace": entry.namespace,
                "query": entry.query,
                "vector": vector_str,
                "response": response_json,
                "created_at": entry.created_at,
                "expires_at": entry.expires_at or "",
                "hit_count": entry.hit_count,
                "model": entry.model,
                "system_prompt": entry.system_prompt,
                "temperature": entry.temperature,
                "tags": json.dumps(list(entry.tags or [])),
                "metadata": json.dumps(entry.metadata, sort_keys=True, allow_nan=False),
            }

            key = self._entry_key(entry_id)

            # Use pipeline for atomicity
            pipe = self._client.pipeline()
            pipe.hset(key, mapping=entry_data)

            # Add to namespace set
            pipe.sadd(self._namespace_key(entry.namespace), entry_id)
            pipe.sadd(self.NAMESPACE_SET, entry.namespace)

            # Add to tag sets
            for tag in entry.tags or []:
                pipe.sadd(f"{self.ROOT}:tag:{entry.namespace}:{tag}", entry_id)

            # Set TTL on the Redis key if entry has expiry
            if entry.expires_at is not None:
                ttl_seconds = max(1, int(entry.expires_at - time.time()))
                pipe.expire(key, ttl_seconds)

            pipe.execute()
            return entry_id

    def retrieve(self, entry_id: int) -> Optional[CacheEntry]:
        """
        Retrieve a cache entry by its ID.

        Args:
            entry_id: The unique ID of the entry.

        Returns:
            CacheEntry if found, None otherwise.
        """
        key = self._entry_key(entry_id)
        data = self._client.hgetall(key)

        if not data:
            return None

        entry = self._data_to_entry(data)
        if entry.is_expired():
            self.delete(entry_id)
            return None

        return entry

    def _data_to_entry(self, data: dict) -> CacheEntry:
        """Convert Redis hash data to a CacheEntry."""
        vector = None
        if data.get("vector"):
            vector = self._str_to_vector(data["vector"])

        response = data.get("response", "")
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                pass

        expires_at = data.get("expires_at")
        if expires_at == "" or expires_at is None:
            expires_at = None
        else:
            expires_at = float(expires_at)

        tags_raw = data.get("tags")
        tags = []
        if tags_raw:
            try:
                tags = json.loads(tags_raw)
            except Exception:
                pass

        temp_val = data.get("temperature")
        temperature = float(temp_val) if temp_val not in (None, "") else -1.0

        return CacheEntry(
            id=int(data["id"]),
            namespace=data["namespace"],
            query=data["query"],
            vector=vector,
            response=response,
            created_at=float(data["created_at"]),
            expires_at=expires_at,
            hit_count=int(data.get("hit_count", 0)),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt", ""),
            temperature=temperature,
            tags=tags,
            metadata=json.loads(data.get("metadata", "{}")),
        )

    def delete(self, entry_id: int) -> bool:
        """
        Delete a cache entry.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if the entry was deleted.
        """
        with self._lock:
            key = self._entry_key(entry_id)

            # Get namespace before deleting
            namespace = self._client.hget(key, "namespace")

            pipe = self._client.pipeline()
            pipe.delete(key)
            if namespace:
                pipe.srem(self._namespace_key(namespace), entry_id)
            pipe.execute()
            return True

    def delete_namespace(self, namespace: str) -> int:
        """
        Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            ns_key = self._namespace_key(namespace)
            entry_ids = self._client.smembers(ns_key)

            if not entry_ids:
                return 0

            pipe = self._client.pipeline()
            for entry_id in entry_ids:
                pipe.delete(self._entry_key(int(entry_id)))
            pipe.delete(ns_key)
            pipe.srem(self.NAMESPACE_SET, namespace)
            pipe.execute()

            return len(entry_ids)

    def delete_by_tag(self, tag: str, namespace: str) -> int:
        """
        Delete all entries in a namespace carrying the given tag.

        Args:
            tag: The tag string to match.
            namespace: Scope the deletion to this namespace.

        Returns:
            The number of entries deleted.
        """
        tag_key = f"{self.ROOT}:tag:{namespace}:{tag}"
        with self._lock:
            entry_ids = self._client.smembers(tag_key)
            if not entry_ids:
                return 0

            pipe = self._client.pipeline()
            for entry_id in entry_ids:
                pipe.delete(self._entry_key(int(entry_id)))
                pipe.srem(self._namespace_key(namespace), entry_id)
            pipe.delete(tag_key)
            pipe.execute()
            return len(entry_ids)

    def clear(self) -> int:
        """
        Clear all entries across all namespaces.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            namespaces = self._client.smembers(self.NAMESPACE_SET)
            total = 0
            for ns in namespaces:
                total += self.delete_namespace(ns)
            return total

    def count(self, namespace: str) -> int:
        """
        Count entries in a namespace.

        Args:
            namespace: The namespace to count.

        Returns:
            The number of entries in the namespace.
        """
        ns_key = self._namespace_key(namespace)
        return self._client.scard(ns_key)

    def list_entries(self, namespace: str) -> list[CacheEntry]:
        """
        List all entries in a namespace.

        Args:
            namespace: The namespace to list.

        Returns:
            List of CacheEntry objects.
        """
        ns_key = self._namespace_key(namespace)
        entry_ids = self._client.smembers(ns_key)

        entries = []
        for entry_id in entry_ids:
            entry = self.retrieve(int(entry_id))
            if entry is not None and not entry.is_expired():
                entries.append(entry)

        return entries

    def list_all_entries(self) -> list[CacheEntry]:
        """List live entries across every namespace for index reconstruction."""
        entries = []
        for namespace in self._client.smembers(self.NAMESPACE_SET):
            entries.extend(self.list_entries(namespace))
        return entries

    def update_hit_count(self, entry_id: int) -> None:
        """
        Increment the hit count for a cache entry.

        Args:
            entry_id: The ID of the entry to update.
        """
        key = self._entry_key(entry_id)
        self._client.hincrby(key, "hit_count", 1)

    def evict_oldest(self, namespace: str, count: int) -> int:
        """
        Evict the oldest entries in a namespace.

        Args:
            namespace: The namespace to evict from.
            count: The number of entries to evict.

        Returns:
            The number of entries actually evicted.
        """
        with self._lock:
            entries = self.list_entries(namespace)
            # Sort by created_at ascending (oldest first)
            entries.sort(key=lambda e: e.created_at)

            evicted = 0
            for entry in entries[:count]:
                if self.delete(entry.id):
                    evicted += 1

            return evicted

    def get_expired(self, namespace: str) -> list[int]:
        """
        Get IDs of all expired entries in a namespace.

        Args:
            namespace: The namespace to check.

        Returns:
            List of expired entry IDs.
        """
        entries = self.list_entries(namespace)
        return [e.id for e in entries if e.is_expired()]

    def __repr__(self) -> str:
        return f"RedisCacheBackend(redis_url={self._redis_url!r})"
