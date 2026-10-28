"""
Cache entry data model.

Represents a single cached query-response pair with metadata
for TTL, namespace isolation, and hit tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class CacheEntry:
    """
    A single cache entry storing a query, its embedding vector, and the cached response.

    Attributes:
        id: Unique identifier for the cache entry.
        namespace: Namespace this entry belongs to (for isolation).
        query: The original query text.
        vector: The embedding vector for the query (numpy float32 array).
        response: The cached LLM response. Can be any serializable Python object.
        created_at: Unix timestamp when the entry was created.
        expires_at: Unix timestamp when the entry expires. None means no expiry.
        hit_count: Number of times this entry has been served from cache.
    """

    id: Optional[int] = None
    namespace: str = "default"
    query: str = ""
    vector: Optional[np.ndarray] = None
    response: Any = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hit_count: int = 0
    model: str = ""
    system_prompt: str = ""
    temperature: float = -1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """
        Check if this cache entry has expired.

        Returns:
            True if the entry has expired, False otherwise.
            Entries with expires_at=None never expire.
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def increment_hit(self) -> None:
        """Increment the hit count for this cache entry."""
        self.hit_count += 1

    def to_dict(self) -> dict:
        """
        Serialize the cache entry to a dictionary.

        The vector is converted to bytes for storage compatibility.
        The response is stored as-is (should be JSON-serializable by backends).

        Returns:
            Dictionary representation of the cache entry.
        """
        return {
            "id": self.id,
            "namespace": self.namespace,
            "query": self.query,
            "vector": self.vector.tobytes() if self.vector is not None else None,
            "response": self.response,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "tags": list(self.tags or []),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict, vector_dim: Optional[int] = None) -> CacheEntry:
        """
        Deserialize a cache entry from a dictionary.

        The vector field is converted from bytes back to a numpy array
        if vector_dim is provided.

        Args:
            data: Dictionary with cache entry fields.
            vector_dim: Expected dimension of the vector array.
                Required if the vector is stored as raw bytes.

        Returns:
            CacheEntry instance.
        """
        vector = data.get("vector")
        if isinstance(vector, bytes) and vector_dim is not None:
            vector = np.frombuffer(vector, dtype=np.float32).copy()
            if vector_dim and len(vector) != vector_dim:
                vector = vector.reshape(vector_dim) if vector.size == vector_dim else vector

        return cls(
            id=data.get("id"),
            namespace=data.get("namespace", "default"),
            query=data.get("query", ""),
            vector=vector,
            response=data.get("response"),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            hit_count=data.get("hit_count", 0),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt", ""),
            temperature=float(data.get("temperature", -1.0)),
            tags=list(data.get("tags") or []),
            metadata=data.get("metadata") or {},
        )
