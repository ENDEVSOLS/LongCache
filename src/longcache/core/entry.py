"""
Cache entry data model.

Represents a single cached query-response pair with metadata
for TTL, namespace isolation, and hit tracking.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import time
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .types import Float32Vector


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

    id: int | None = None
    namespace: str = "default"
    query: str = ""
    vector: Float32Vector | None = None
    response: Any = None
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    hit_count: int = 0
    model: str = ""
    system_prompt: str = ""
    temperature: float = -1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: Mapping[str, Any], vector_dim: int | None = None) -> CacheEntry:
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
        parsed_vector: Float32Vector | None = None
        if isinstance(vector, bytes) and vector_dim is not None:
            buf = np.frombuffer(vector, dtype=np.float32).copy()
            if vector_dim and len(buf) != vector_dim:
                parsed_vector = buf.reshape(vector_dim) if buf.size == vector_dim else buf
            else:
                parsed_vector = buf
        elif isinstance(vector, np.ndarray) and vector.dtype == np.float32:
            parsed_vector = vector

        raw_id = data.get("id")
        entry_id = int(raw_id) if raw_id is not None else None

        raw_created = data.get("created_at")
        created_at = float(raw_created) if raw_created is not None else time.time()

        raw_expires = data.get("expires_at")
        expires_at = float(raw_expires) if raw_expires not in (None, "") else None

        raw_hits = data.get("hit_count")
        hit_count = int(raw_hits) if raw_hits is not None else 0

        raw_temp = data.get("temperature")
        temperature = float(raw_temp) if raw_temp not in (None, "") else -1.0

        raw_meta = data.get("metadata")
        metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}

        raw_tags = data.get("tags")
        tags = [str(t) for t in raw_tags] if isinstance(raw_tags, (list, tuple)) else []

        return cls(
            id=entry_id,
            namespace=str(data.get("namespace", "default")),
            query=str(data.get("query", "")),
            vector=parsed_vector,
            response=data.get("response"),
            created_at=created_at,
            expires_at=expires_at,
            hit_count=hit_count,
            model=str(data.get("model", "")),
            system_prompt=str(data.get("system_prompt", "")),
            temperature=temperature,
            tags=tags,
            metadata=metadata,
        )
