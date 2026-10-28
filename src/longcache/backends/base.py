"""
Abstract base class for cache backends.

All backends must implement this interface to provide
consistent storage semantics regardless of the underlying
storage engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from longcache.core.entry import CacheEntry


class AbstractCacheBackend(ABC):
    """
    Abstract interface for LongCache storage backends.

    All methods operate on CacheEntry objects, which encapsulate
    the query, vector, response, and metadata. Backends are
    responsible for serialization/deserialization as needed.
    """

    @abstractmethod
    def store(self, entry: CacheEntry) -> int:
        """
        Store a cache entry and return its unique ID.

        Args:
            entry: The CacheEntry to store. May not have an ID assigned yet.

        Returns:
            The unique integer ID assigned to the stored entry.
        """
        ...

    @abstractmethod
    def retrieve(self, entry_id: int) -> Optional[CacheEntry]:
        """
        Retrieve a cache entry by its ID.

        Args:
            entry_id: The unique ID of the entry to retrieve.

        Returns:
            CacheEntry if found, None otherwise.
        """
        ...

    @abstractmethod
    def delete(self, entry_id: int) -> bool:
        """
        Delete a cache entry by its ID.

        Args:
            entry_id: The unique ID of the entry to delete.

        Returns:
            True if the entry was deleted, False if not found.
        """
        ...

    @abstractmethod
    def delete_namespace(self, namespace: str) -> int:
        """
        Delete all entries in a namespace.

        Args:
            namespace: The namespace to clear.

        Returns:
            The number of entries deleted.
        """
        ...

    @abstractmethod
    def delete_by_tag(self, tag: str, namespace: str) -> int:
        """
        Delete all entries in a namespace carrying the given tag.

        Args:
            tag: The tag string to match.
            namespace: Scope the deletion to this namespace.

        Returns:
            The number of entries deleted.
        """
        ...

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all entries across all namespaces.

        Returns:
            The total number of entries deleted.
        """
        ...

    @abstractmethod
    def count(self, namespace: str) -> int:
        """
        Count entries in a namespace.

        Args:
            namespace: The namespace to count entries in.

        Returns:
            The number of entries in the namespace.
        """
        ...

    @abstractmethod
    def list_entries(self, namespace: str) -> list[CacheEntry]:
        """
        List all non-expired entries in a namespace.

        Used primarily for rebuilding the FAISS index on startup.

        Args:
            namespace: The namespace to list entries from.

        Returns:
            List of CacheEntry objects with vectors populated.
        """
        ...

    @abstractmethod
    def update_hit_count(self, entry_id: int) -> None:
        """
        Increment the hit count for a cache entry.

        Args:
            entry_id: The ID of the entry to update.
        """
        ...

    @abstractmethod
    def evict_oldest(self, namespace: str, count: int) -> int:
        """
        Evict the oldest entries in a namespace.

        Used for LRU-style eviction when max_entries is exceeded.

        Args:
            namespace: The namespace to evict entries from.
            count: The number of entries to evict.

        Returns:
            The number of entries actually evicted.
        """
        ...

    @abstractmethod
    def get_expired(self, namespace: str) -> list[int]:
        """
        Get IDs of all expired entries in a namespace.

        Args:
            namespace: The namespace to check for expired entries.

        Returns:
            List of entry IDs that have expired.
        """
        ...

    @abstractmethod
    def list_all_entries(self) -> list[CacheEntry]:
        """Enumerate live entries for a validated generation rebuild."""
        ...

    @abstractmethod
    def ensure_manifest(self, manifest: dict) -> None:
        """Initialize or compare the immutable store generation descriptor."""
        ...

    @abstractmethod
    def read_manifest(self) -> dict | None:
        """Read generation metadata without loading an embedding model."""
        ...
