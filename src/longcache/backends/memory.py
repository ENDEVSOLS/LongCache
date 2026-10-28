"""
In-memory cache backend for LongCache.

Primarily used for testing and benchmarking. Entries are stored
in a Python dictionary and are lost when the process exits.
Supports TTL expiry and namespace isolation.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from longcache.backends.base import AbstractCacheBackend
from longcache.core.entry import CacheEntry


class InMemoryCacheBackend(AbstractCacheBackend):
    """
    In-memory cache backend using a Python dictionary.

    Fastest backend since there is no I/O overhead, but entries
    do not survive process restarts. Ideal for testing, CI/CD,
    and short-lived applications.

    Usage:
        backend = InMemoryCacheBackend()
        entry_id = backend.store(entry)
        result = backend.retrieve(entry_id)
    """

    def __init__(self, max_entries: int = 10_000):
        """
        Initialize the in-memory backend.

        Args:
            max_entries: Maximum entries per namespace.
        """
        self._max_entries = max_entries
        self._entries: dict[int, CacheEntry] = {}
        self._next_id = 1
        self._lock = threading.RLock()

    def read_manifest(self):
        import json

        return json.loads(self._manifest) if hasattr(self, "_manifest") else None

    def ensure_manifest(self, manifest: dict) -> None:
        from longcache.core.identity import canonical

        with self._lock:
            encoded = canonical(manifest)
            if hasattr(self, "_manifest") and self._manifest != encoded:
                raise ValueError("Incompatible embedding or storage generation")
            self._manifest = encoded

    def store(self, entry: CacheEntry) -> int:
        """
        Store a cache entry in memory.

        Args:
            entry: The CacheEntry to store.

        Returns:
            The auto-generated integer ID.
        """
        with self._lock:
            entry_id = self._next_id
            entry.id = entry_id
            self._entries[entry_id] = entry
            self._next_id += 1
            return entry_id

    def retrieve(self, entry_id: int) -> Optional[CacheEntry]:
        """
        Retrieve a cache entry by its ID.

        Args:
            entry_id: The unique ID of the entry.

        Returns:
            CacheEntry if found and not expired, None otherwise.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None

            if entry.is_expired():
                self.delete(entry_id)
                return None

            return entry

    def delete(self, entry_id: int) -> bool:
        """
        Delete a cache entry.

        Args:
            entry_id: The ID of the entry to delete.

        Returns:
            True if the entry was deleted, False if not found.
        """
        with self._lock:
            return self._entries.pop(entry_id, None) is not None

    def delete_namespace(self, namespace: str) -> int:
        """
        Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            to_delete = [
                eid for eid, entry in self._entries.items() if entry.namespace == namespace
            ]
            for eid in to_delete:
                del self._entries[eid]
            return len(to_delete)

    def delete_by_tag(self, tag: str, namespace: str) -> int:
        """
        Delete all entries in a namespace carrying the given tag.

        Args:
            tag: The tag string to match.
            namespace: Scope the deletion to this namespace.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            to_delete = [
                eid
                for eid, entry in self._entries.items()
                if entry.namespace == namespace and tag in (entry.tags or [])
            ]
            for eid in to_delete:
                del self._entries[eid]
            return len(to_delete)

    def clear(self) -> int:
        """
        Clear all entries.

        Returns:
            The number of entries deleted.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def count(self, namespace: str) -> int:
        """
        Count non-expired entries in a namespace.

        Args:
            namespace: The namespace to count.

        Returns:
            The number of non-expired entries.
        """
        with self._lock:
            now = time.time()
            return sum(
                1
                for entry in self._entries.values()
                if entry.namespace == namespace
                and (entry.expires_at is None or entry.expires_at > now)
            )

    def list_entries(self, namespace: str) -> list[CacheEntry]:
        """
        List all non-expired entries in a namespace.

        Args:
            namespace: The namespace to list.

        Returns:
            List of CacheEntry objects.
        """
        with self._lock:
            now = time.time()
            return [
                entry
                for entry in self._entries.values()
                if entry.namespace == namespace
                and (entry.expires_at is None or entry.expires_at > now)
            ]

    def list_all_entries(self) -> list[CacheEntry]:
        """List live entries across every namespace for index reconstruction."""
        with self._lock:
            return [entry for entry in self._entries.values() if not entry.is_expired()]

    def update_hit_count(self, entry_id: int) -> None:
        """
        Increment the hit count for a cache entry.

        Args:
            entry_id: The ID of the entry to update.
        """
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is not None:
                entry.increment_hit()

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
            ns_entries = [
                (eid, entry) for eid, entry in self._entries.items() if entry.namespace == namespace
            ]
            # Sort by created_at ascending (oldest first)
            ns_entries.sort(key=lambda x: x[1].created_at)

            evicted = 0
            for eid, _ in ns_entries[:count]:
                del self._entries[eid]
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
        with self._lock:
            now = time.time()
            return [
                eid
                for eid, entry in self._entries.items()
                if entry.namespace == namespace
                and entry.expires_at is not None
                and entry.expires_at <= now
            ]

    def __repr__(self) -> str:
        return f"InMemoryCacheBackend(entries={len(self._entries)})"
