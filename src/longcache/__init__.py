"""
LongCache — Semantic similarity cache for LLM/RAG pipelines.

Reuse responses for semantically similar queries when their context matches.
"""

from longcache.core.cache import LongCache
from longcache.config import CacheConfig

__version__ = "0.5.0"
from .core.identity import Instruction, RequestContext, RequestScope
from .core.decision import LookupDecision, StoreDecision, PolicyDecision
from .core.base_embedder import BaseEmbedder
from .core.embedding_contract import EmbeddingDescriptor
from .core.types import PUBLIC_API_STABILITY
from .core.errors import (
    CacheError,
    CacheClosedError,
    CacheShutdownTimeoutError,
    CacheCapacityError,
    CacheOperationError,
    SingleflightError,
    SingleflightTimeoutError,
    SingleflightCapacityError,
    SingleflightLeaderError,
    SingleflightReentrancyError,
    SingleflightInvalidatedError,
)

__all__ = [
    "LongCache",
    "CacheConfig",
    "Instruction",
    "RequestContext",
    "RequestScope",
    "LookupDecision",
    "StoreDecision",
    "PolicyDecision",
    "BaseEmbedder",
    "EmbeddingDescriptor",
    "PUBLIC_API_STABILITY",
    "CacheError",
    "CacheClosedError",
    "CacheShutdownTimeoutError",
    "CacheCapacityError",
    "CacheOperationError",
    "SingleflightError",
    "SingleflightTimeoutError",
    "SingleflightCapacityError",
    "SingleflightLeaderError",
    "SingleflightReentrancyError",
    "SingleflightInvalidatedError",
]
