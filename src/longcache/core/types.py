"""Public static type contracts, literal aliases, and protocol definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, Protocol, TypeAlias, TypedDict, TypeVar
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .decision import PolicyDecision
    from .identity import RequestContext, RequestScope

T = TypeVar("T")
P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)

API_CONTRACT_VERSION: Literal["1.0"] = "1.0"
StabilityTier: TypeAlias = Literal["stable", "experimental", "pass-through"]

Float32Vector: TypeAlias = npt.NDArray[np.float32]
ScopePolicy: TypeAlias = Literal["private", "tenant_shared", "public"]
BackendName: TypeAlias = Literal["sqlite", "redis", "memory"]
DeploymentMode: TypeAlias = Literal["single_process", "shared"]
ErrorPolicy: TypeAlias = Literal["raise", "bypass"]
FaissIndexType: TypeAlias = Literal["flat", "hnsw"]
InstructionRole: TypeAlias = Literal["system", "developer"]
ResponseCodec: TypeAlias = Literal["json-v1", "openai-dict-v1", "openai-sdk-v1"]
LookupOutcome: TypeAlias = Literal["HIT", "MISS", "BYPASS"]
StoreOutcome: TypeAlias = Literal["STORED", "REJECTED"]
PolicyAction: TypeAlias = Literal["accept", "reject", "abstain"]
HealthStatus: TypeAlias = Literal["healthy", "degraded"]


class AdmissionHook(Protocol):
    """Callback for application-level admission validation."""

    def __call__(self, context: RequestContext, normalized_response: Any, /) -> PolicyDecision: ...


class VerificationHook(Protocol):
    """Callback for domain-specific semantic verification."""

    def __call__(
        self,
        query: str,
        candidate_query: str,
        normalized_response: Any,
        similarity: float,
        context: RequestContext,
        /,
    ) -> PolicyDecision: ...


class ContextResolver(Protocol):
    """Synchronous resolver converting query or call args into RequestContext."""

    def __call__(self, *args: Any, **kwargs: Any) -> RequestContext | None: ...


class ScopeResolver(Protocol):
    """Synchronous resolver extracting RequestScope from an HTTP/ASGI connection."""

    def __call__(self, scope: Mapping[str, Any], /) -> RequestScope | None: ...


class CachedCallable(Protocol[P, R_co]):
    """Wrapped callable returned by @cache.cached preserving signature and cache_context_for."""

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...
    def cache_context_for(self, base: RequestContext | None) -> RequestContext | None: ...


class InspectionReport(TypedDict):
    """Inspection dictionary format returned by cache.inspect()."""

    outcome: LookupOutcome
    reason: str
    similarity: float | None
    id: int | None
    threshold: float | None
    format_version: int


class StatsSnapshot(TypedDict):
    """Scoped or operator statistics snapshot returned by cache.stats()."""

    hits: int
    misses: int
    hit_rate: float
    total_entries: int
    total_saved_ms: float
    estimated_tokens_saved: int
    estimated_cost_saved_usd: float
    stats_window_started_at: float | None
    stats_retention_evictions: int


class ResourceSnapshot(TypedDict):
    """Process-level resource metrics snapshot."""

    state: str
    active_operations: int
    cache_jobs: int
    cleanup_failed: bool
    cleanup_workers: int
    active_flights: int
    followers: int
    retained_replay_bytes: int
    sqlite_connections: int


class MetricsSnapshot(TypedDict):
    """Process-wide metrics returned by cache.metrics(operator=True)."""

    enabled: bool
    counters: list[dict[str, Any]]
    latency: list[dict[str, Any]]
    resources: ResourceSnapshot
    health: HealthStatus


PUBLIC_API_STABILITY: dict[str, StabilityTier] = {
    # Core stable
    "LongCache": "stable",
    "CacheConfig": "stable",
    "__version__": "stable",
    "API_CONTRACT_VERSION": "stable",
    "StabilityTier": "stable",
    "Instruction": "stable",
    "RequestContext": "stable",
    "RequestScope": "stable",
    "LookupDecision": "stable",
    "StoreDecision": "stable",
    "PolicyDecision": "stable",
    "BaseEmbedder": "stable",
    "EmbeddingDescriptor": "stable",
    "CacheEntry": "stable",
    "CacheError": "stable",
    "CacheClosedError": "stable",
    "CacheShutdownTimeoutError": "stable",
    "CacheCapacityError": "stable",
    "CacheOperationError": "stable",
    "SingleflightError": "stable",
    "SingleflightTimeoutError": "stable",
    "SingleflightCapacityError": "stable",
    "SingleflightLeaderError": "stable",
    "SingleflightReentrancyError": "stable",
    "SingleflightInvalidatedError": "stable",
    "LongCacheWarning": "stable",
    "LongCachePassThroughWarning": "stable",
    "LongCacheExperimentalWarning": "stable",
    # Public static typing contracts
    "AdmissionHook": "stable",
    "VerificationHook": "stable",
    "ContextResolver": "stable",
    "ScopeResolver": "stable",
    "CachedCallable": "stable",
    "InspectionReport": "stable",
    "StatsSnapshot": "stable",
    "ResourceSnapshot": "stable",
    "MetricsSnapshot": "stable",
    "ScopePolicy": "stable",
    "BackendName": "stable",
    "DeploymentMode": "stable",
    "ErrorPolicy": "stable",
    "FaissIndexType": "stable",
    "InstructionRole": "stable",
    "ResponseCodec": "stable",
    "LookupOutcome": "stable",
    "StoreOutcome": "stable",
    "PolicyAction": "stable",
    "HealthStatus": "stable",
    "Float32Vector": "stable",
    "PUBLIC_API_STABILITY": "stable",
    # Backends
    "AbstractCacheBackend": "stable",
    "InMemoryCacheBackend": "stable",
    "SQLiteCacheBackend": "stable",
    "get_redis_backend": "stable",
    "RedisCacheBackend": "stable",
    "SharedRedisCacheBackend": "stable",
    "SharedCoordinationBackend": "stable",
    "AsyncRedisCacheBackend": "experimental",
    # Integrations & Middleware
    "CachedOpenAI": "stable",
    "get_openai_wrapper": "stable",
    "LongCacheMiddleware": "stable",
    "LongCache.cached_stream": "pass-through",
    "StreamingCacheAdapter": "pass-through",
    "CachedLLM": "pass-through",
    "CachedChain": "pass-through",
    "CachedQueryEngine": "pass-through",
    "get_llamaindex_wrapper": "pass-through",
    "CachedState": "pass-through",
}
