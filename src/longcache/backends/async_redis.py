"""
Async Redis cache backend using redis.asyncio.

Standalone async storage backend. It is not yet wired into LongCache's
index and lifecycle operations.

    backend = AsyncRedisCacheBackend(redis_url="redis://localhost:6379/0")
    entry_id = await backend.store(entry)
    result = await backend.retrieve(entry_id)
"""

from __future__ import annotations

import base64
import json
import time
import asyncio
from typing import Optional

import numpy as np

from longcache.core.entry import CacheEntry


class AsyncRedisCacheBackend:
    """
    Async Redis-based cache backend for distributed deployments.
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
        try:
            from redis.asyncio import Redis
        except ImportError:
            raise ImportError(
                "Async Redis backend requires the 'redis' extra. "
                "Install with: uv pip install longcache[redis]"
            )

        self.ROOT = f"longcache:v2:{cache_id}:{generation}"
        self.KEY_PREFIX = self.ROOT + ":entry"
        self.NAMESPACE_SET = self.ROOT + ":namespaces"
        self.ID_COUNTER = self.ROOT + ":id_counter"
        self._redis_url = redis_url
        self._max_entries = max_entries
        self._lock = asyncio.Lock()
        self._client = Redis.from_url(redis_url, decode_responses=True)

    def _entry_key(self, entry_id: int) -> str:
        return f"{self.KEY_PREFIX}:{entry_id}"

    def _namespace_key(self, namespace: str) -> str:
        return f"{self.ROOT}:ns:{namespace}"

    def _vector_to_str(self, vector: np.ndarray) -> str:
        return base64.b64encode(vector.tobytes()).decode("ascii")

    def _str_to_vector(self, vector_str: str) -> np.ndarray:
        return np.frombuffer(base64.b64decode(vector_str), dtype=np.float32).copy()

    async def store(self, entry: CacheEntry) -> int:
        async with self._lock:
            entry_id = await self._client.incr(self.ID_COUNTER)

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

            pipe = self._client.pipeline()
            pipe.hset(key, mapping=entry_data)

            pipe.sadd(self._namespace_key(entry.namespace), entry_id)
            pipe.sadd(self.NAMESPACE_SET, entry.namespace)

            for tag in entry.tags or []:
                pipe.sadd(f"{self.ROOT}:tag:{entry.namespace}:{tag}", entry_id)

            if entry.expires_at is not None:
                ttl_seconds = max(1, int(entry.expires_at - time.time()))
                pipe.expire(key, ttl_seconds)

            await pipe.execute()

        # Note: outside the lock — _enforce_max_entries() calls evict_oldest()
        # which calls delete(), and self._lock is not reentrant.
        await self._enforce_max_entries(entry.namespace)
        return entry_id

    async def retrieve(self, entry_id: int) -> Optional[CacheEntry]:
        key = self._entry_key(entry_id)
        data = await self._client.hgetall(key)

        if not data:
            return None

        entry = self._data_to_entry(data)
        if entry.is_expired():
            await self.delete(entry_id)
            return None

        return entry

    def _data_to_entry(self, data: dict) -> CacheEntry:
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

    async def delete(self, entry_id: int) -> bool:
        async with self._lock:
            key = self._entry_key(entry_id)
            namespace = await self._client.hget(key, "namespace")

            pipe = self._client.pipeline()
            pipe.delete(key)
            if namespace:
                pipe.srem(self._namespace_key(namespace), entry_id)
            await pipe.execute()
            return True

    async def delete_namespace(self, namespace: str) -> int:
        async with self._lock:
            ns_key = self._namespace_key(namespace)
            entry_ids = await self._client.smembers(ns_key)

            if not entry_ids:
                return 0

            pipe = self._client.pipeline()
            for entry_id in entry_ids:
                pipe.delete(self._entry_key(int(entry_id)))
            pipe.delete(ns_key)
            pipe.srem(self.NAMESPACE_SET, namespace)
            await pipe.execute()

            return len(entry_ids)

    async def delete_by_tag(self, tag: str, namespace: str) -> int:
        """
        Delete all entries in a namespace carrying the given tag.

        Args:
            tag: The tag string to match.
            namespace: Scope the deletion to this namespace.

        Returns:
            The number of entries deleted.
        """
        tag_key = f"{self.ROOT}:tag:{namespace}:{tag}"
        async with self._lock:
            entry_ids = await self._client.smembers(tag_key)
            if not entry_ids:
                return 0

            pipe = self._client.pipeline()
            for entry_id in entry_ids:
                pipe.delete(self._entry_key(int(entry_id)))
                pipe.srem(self._namespace_key(namespace), entry_id)
            pipe.delete(tag_key)
            await pipe.execute()
            return len(entry_ids)

    async def clear(self) -> int:
        # Note: delete_namespace() acquires self._lock itself, so we must
        # NOT hold the lock here too — asyncio.Lock is not reentrant and
        # doing so would deadlock.
        namespaces = await self._client.smembers(self.NAMESPACE_SET)
        total = 0
        for ns in namespaces:
            total += await self.delete_namespace(ns)
        return total

    async def count(self, namespace: str) -> int:
        ns_key = self._namespace_key(namespace)
        return await self._client.scard(ns_key)

    async def list_entries(self, namespace: str) -> list[CacheEntry]:
        ns_key = self._namespace_key(namespace)
        entry_ids = await self._client.smembers(ns_key)

        entries = []
        for entry_id in entry_ids:
            entry = await self.retrieve(int(entry_id))
            if entry is not None and not entry.is_expired():
                entries.append(entry)

        return entries

    async def update_hit_count(self, entry_id: int) -> None:
        key = self._entry_key(entry_id)
        await self._client.hincrby(key, "hit_count", 1)

    async def evict_oldest(self, namespace: str, count: int) -> int:
        # Note: delete() acquires self._lock itself, so we must not hold
        # the lock here too — asyncio.Lock is not reentrant.
        entries = await self.list_entries(namespace)
        entries.sort(key=lambda e: e.created_at)

        evicted = 0
        for entry in entries[:count]:
            if await self.delete(entry.id):
                evicted += 1

        return evicted

    async def get_expired(self, namespace: str) -> list[int]:
        entries = await self.list_entries(namespace)
        return [e.id for e in entries if e.is_expired()]

    async def _enforce_max_entries(self, namespace: str) -> None:
        count = await self.count(namespace)
        if count > self._max_entries:
            excess = count - self._max_entries
            await self.evict_oldest(namespace, excess)

    def __repr__(self) -> str:
        return f"AsyncRedisCacheBackend(redis_url={self._redis_url!r})"
