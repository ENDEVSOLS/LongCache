"""Public static type contracts, literal aliases, and protocol definitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias, TypedDict, TypeVar
import numpy as np
import numpy.typing as npt

from .decision import PolicyDecision
from .identity import RequestContext, RequestScope

T = TypeVar("T")

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

    active_operations: int
    closing: bool
    closed: bool
    cleanup_failed: bool
    active_flights: int
    active_waiters: int
    sqlite_connections: int


class MetricsSnapshot(TypedDict):
    """Process-wide metrics returned by cache.metrics(operator=True)."""

    counters: dict[str, Any]
    latencies_ms: dict[str, Any]
    resources: ResourceSnapshot
    health: HealthStatus
