"""
FAISS-based vector similarity index for LongCache.

Maintains an in-memory IndexFlatIP (Inner Product = Cosine Similarity for
normalized vectors) mapping query embeddings to CacheEntry IDs.
Supports both synchronous and asynchronous operations.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import threading
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from longcache.backends.base import AbstractCacheBackend
    from longcache.config import CacheConfig
    from longcache.core.entry import CacheEntry
    from longcache.core.types import Float32Vector


class SimilarityIndex:
    """
    In-memory vector index backed by FAISS IndexFlatIP.

    Maps 0-indexed internal FAISS IDs to persistent CacheEntry IDs.
    Thread-safe via threading.RLock for synchronous calls, with
    asyncio.Lock for async entry points.

    Usage:
        index = SimilarityIndex(backend, config)
        index.add(vector, entry_id)
        result = index.find_similar(query_vector, threshold=0.92)
    """

    def __init__(self, backend: AbstractCacheBackend, config: CacheConfig) -> None:
        """
        Initialize the similarity index.

        Args:
            backend: The cache backend (used for retrieving entries by ID).
            config: CacheConfig instance with similarity_threshold and other settings.
        """
        self._backend = backend
        self._config = config
        self._lock = threading.RLock()
        self._async_lock: asyncio.Lock | None = None
        self._faiss_index: Any = None
        self._id_map: dict[int, int] = {}  # faiss_internal_id -> entry_id
        self._entry_ids: set[int] = set()
        self._next_faiss_id = 0
        self._dimension: int | None = None
        self._orphan_count = 0  # vectors in FAISS with no id_map entry
        self._REBUILD_THRESHOLD = 0.20  # rebuild when 20% of index is orphaned
        self._removed_entry_ids: set[int] = set()  # entry ids explicitly removed via remove()

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create asyncio.Lock on first async use."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _new_index(self, dimension: int) -> Any:
        """Create an empty index that honors the configured search mode."""
        import faiss

        if self._config.faiss_index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, self._config.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self._config.hnsw_ef_search
            return index
        return faiss.IndexFlatIP(dimension)

    def _ensure_index(self, dimension: int) -> None:
        """
        Lazily create the FAISS index with the given dimension.

        Called automatically on first add() with a vector of the
        appropriate dimension.

        Args:
            dimension: The dimensionality of the embedding vectors.
        """
        if self._faiss_index is not None:
            return

        self._faiss_index = self._new_index(dimension)
        self._dimension = dimension

    def add(self, vector: Float32Vector, entry_id: int) -> None:
        """
        Add an embedding vector to the index.

        Creates the FAISS index on first call, inferring dimension
        from the input vector.

        Args:
            vector: L2-normalized embedding vector of shape (dimension,).
            entry_id: The database ID of the corresponding CacheEntry.
        """
        with self._lock:
            if entry_id in self._entry_ids:
                return
            if self._faiss_index is None:
                self._ensure_index(vector.shape[0])
            elif vector.shape != (self._dimension,):
                raise ValueError("Embedding dimension changed; clear the cache before reuse")

            # FAISS expects 2D arrays
            vector_2d = vector.reshape(1, -1).astype(np.float32)
            faiss_id = self._next_faiss_id
            if self._faiss_index is None:
                self._ensure_index(vector.shape[0])
            assert self._faiss_index is not None
            self._faiss_index.add(vector_2d)
            self._id_map[faiss_id] = entry_id
            self._entry_ids.add(entry_id)
            self._next_faiss_id += 1

    async def async_add(self, vector: Float32Vector, entry_id: int) -> None:
        """Async-safe version of add() for use in async contexts."""
        async with self._get_async_lock():
            self.add(vector, entry_id)

    def candidates(
        self,
        vector: Float32Vector,
        namespace: str,
        threshold: float,
        limit: int,
        *,
        retrieve: Callable[[int], CacheEntry | None] | None = None,
    ) -> list[tuple[CacheEntry, float]]:
        """Return scored eligible snapshots without recording hits or invoking hooks."""
        retrieve_fn = retrieve or self._backend.retrieve
        with self._lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []
            if vector.shape != (self._dimension,):
                raise ValueError("Incompatible embedding dimension")
            total = int(self._faiss_index.ntotal)
            k = min(total, max(8, limit))
            seen: set[int] = set()
            result: list[tuple[CacheEntry, float]] = []
            while True:
                scores, ids = self._faiss_index.search(vector.reshape(1, -1), k)
                for score, faiss_id in zip(scores[0], ids[0]):
                    identifier = self._id_map.get(int(faiss_id))
                    if identifier is None or identifier in seen or score < threshold:
                        continue
                    seen.add(identifier)
                    entry = retrieve_fn(identifier)
                    if (
                        entry is not None
                        and entry.namespace == namespace
                        and not entry.is_expired()
                    ):
                        result.append((deepcopy(entry), float(score)))
                        if len(result) >= limit:
                            return result
                if k == total or (len(scores[0]) and scores[0][-1] < threshold):
                    return result
                k = min(total, k * 2)

    def find_similar(
        self,
        query_vector: Float32Vector,
        threshold: float | None = None,
        namespace: str = "default",
        record_hit: bool = True,
    ) -> CacheEntry | None:
        """
        Search the index for the most similar non-expired entry in the namespace.

        Eligible candidates in the requested namespace are checked in descending
        similarity order so an expired or wrong-namespace vector does not hide a hit.

        Args:
            query_vector: L2-normalized float32 array of shape (dimension,).
            threshold: Minimum cosine similarity score (0.0-1.0).
                Defaults to config.similarity_threshold if None.
            namespace: Only match entries from this namespace.
            record_hit: If True, increments hit_count on the matched entry.

        Returns:
            The best matching CacheEntry if similarity >= threshold, None otherwise.
        """
        if threshold is None:
            threshold = self._config.similarity_threshold

        with self._lock:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return None
            if self._dimension is not None and query_vector.shape != (self._dimension,):
                raise ValueError("Embedding dimension changed; clear the cache before reuse")

            total = int(self._faiss_index.ntotal)
            k = min(total, 8)
            examined = 0
            while True:
                scores, ids = self._faiss_index.search(query_vector.reshape(1, -1), k)
                for score, faiss_id in zip(scores[0][examined:], ids[0][examined:]):
                    if score < threshold or int(faiss_id) == -1:
                        return None
                    entry_id = self._id_map.get(int(faiss_id))
                    if entry_id is None:
                        continue
                    entry = self._backend.retrieve(entry_id)
                    if entry is None or entry.namespace != namespace or entry.is_expired():
                        continue
                    if record_hit:
                        self._backend.update_hit_count(entry_id)
                    return entry
                if k == total:
                    return None
                examined = k
                k = min(total, k * 2)

    def remove(self, faiss_id: int) -> None:
        """
        Remove an entry from the ID map by its FAISS internal ID.

        Note: FAISS IndexFlatIP does not support efficient removal.
        The entry is removed from the ID map, so it won't be returned
        in future searches even though the vector remains in the index.
        For production use with high churn, consider rebuilding the index
        periodically.

        Args:
            faiss_id: The internal FAISS index ID to remove.
        """
        with self._lock:
            if faiss_id in self._id_map:
                entry_id = self._id_map.pop(faiss_id)
                self._entry_ids.discard(entry_id)
                self._removed_entry_ids.add(entry_id)
                self._orphan_count += 1
                self._auto_rebuild_if_needed()

    def _auto_rebuild_if_needed(self) -> None:
        """Rebuild FAISS index if orphan ratio exceeds threshold."""
        total = self._faiss_index.ntotal if self._faiss_index else 0
        if total > 0 and self._orphan_count / total >= self._REBUILD_THRESHOLD:
            import logging

            logging.getLogger(__name__).info(
                "Auto-rebuilding FAISS index: %d orphaned vectors (%.0f%%)",
                self._orphan_count,
                (self._orphan_count / total) * 100,
            )
            entries = self._backend.list_all_entries()
            filtered_entries = [
                e for e in entries if e.id is not None and e.id not in self._removed_entry_ids
            ]
            self.rebuild(filtered_entries)
            self._orphan_count = 0

    def rebuild(self, entries: list[CacheEntry]) -> None:
        """
        Rebuild the FAISS index from a list of cache entries.

        Useful for compacting the index after many removals, or for
        loading the index from persistent storage on startup.

        Args:
            entries: List of CacheEntry objects with their vectors populated.
        """
        with self._lock:
            valid_entries: list[tuple[int, Float32Vector]] = []
            for e in entries or []:
                if e.vector is not None and e.id is not None:
                    valid_entries.append((e.id, e.vector))

            if not valid_entries:
                self._faiss_index = None
                self._id_map = {}
                self._entry_ids = set()
                self._next_faiss_id = 0
                self._orphan_count = 0
                self._removed_entry_ids.clear()
                self._dimension = None
                return

            dimension = valid_entries[0][1].shape[0]
            if any(vec.shape != (dimension,) for _, vec in valid_entries):
                raise ValueError("Persisted cache contains incompatible embedding dimensions")

            replacement = self._new_index(dimension)
            vectors_array = np.stack([vec for _, vec in valid_entries]).astype(np.float32)
            replacement.add(vectors_array)
            self._faiss_index = replacement
            self._id_map = {i: entry_id for i, (entry_id, _) in enumerate(valid_entries)}
            self._entry_ids = {entry_id for entry_id, _ in valid_entries}
            self._next_faiss_id = len(valid_entries)
            self._dimension = dimension

            self._orphan_count = 0
            self._removed_entry_ids.clear()

    @property
    def size(self) -> int:
        """Return the number of entries currently in the index."""
        with self._lock:
            return len(self._id_map)

    @property
    def dimension(self) -> int | None:
        """Return the dimensionality of the index, or None if not initialized."""
        return self._dimension
