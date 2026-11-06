"""
LongCache — Semantic similarity cache for LLM/RAG pipelines.

Reuse responses for semantically similar queries when their context matches.
"""

from longcache.core.cache import LongCache
from longcache.config import CacheConfig

__version__ = "0.6.0"
from .core.identity import Instruction, RequestContext, RequestScope
from .core.decision import LookupDecision, StoreDecision, PolicyDecision
from .core.base_embedder import BaseEmbedder
from .core.embedding_contract import EmbeddingDescriptor
from .core.entry import CacheEntry
from .core.types import (
    AdmissionHook,
    BackendName,
    CachedCallable,
    ContextResolver,
    DeploymentMode,
    ErrorPolicy,
    FaissIndexType,
    Float32Vector,
    HealthStatus,
    InspectionReport,
    InstructionRole,
    LookupOutcome,
    MetricsSnapshot,
    PUBLIC_API_STABILITY,
    PolicyAction,
    ResourceSnapshot,
    ResponseCodec,
    ScopePolicy,
    ScopeResolver,
    StatsSnapshot,
    StoreOutcome,
    VerificationHook,
)
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
    "__version__",
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
    "CacheEntry",
    "AdmissionHook",
    "VerificationHook",
    "ContextResolver",
    "ScopeResolver",
    "CachedCallable",
    "InspectionReport",
    "StatsSnapshot",
    "ResourceSnapshot",
    "MetricsSnapshot",
    "ScopePolicy",
    "BackendName",
    "DeploymentMode",
    "ErrorPolicy",
    "FaissIndexType",
    "InstructionRole",
    "ResponseCodec",
    "LookupOutcome",
    "StoreOutcome",
    "PolicyAction",
    "HealthStatus",
    "Float32Vector",
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
