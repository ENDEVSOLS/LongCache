"""Scoped semantic reuse with shared policy across sync and async entry points."""

from __future__ import annotations

import asyncio
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, replace
from functools import wraps
import inspect
import threading
import time
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, cast, overload

from longcache.backends.memory import InMemoryCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend
from longcache.config import CacheConfig

from .base_embedder import BaseEmbedder
from .decision import InspectionReport, LookupDecision, StoreDecision
from .distributed_singleflight import DistributedExecution
from .embedder import LocalEmbedder
from .embedding_contract import EmbeddingGuard, validate_vector
from .entry import CacheEntry
from .errors import CacheClosedError, CacheOperationError
from .execution import Execution
from .identity import RequestContext, canonical, digest
from .index import SimilarityIndex
from .lifecycle import Lifecycle, managed
from .observability import Observability
from .policy import admit, decode_response, verify
from .reliability import decision_operation, dependency, execution_operation, operational
from .types import AdmissionHook, CachedCallable, MetricsSnapshot, StatsSnapshot, VerificationHook

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])
P = ParamSpec("P")
R = TypeVar("R")


class LongCache:
    def __init__(
        self,
        config: CacheConfig | None = None,
        *,
        embedder: BaseEmbedder | None = None,
        admit: AdmissionHook | None = None,
        verifier: VerificationHook | None = None,
        context: RequestContext | None = None,
        metadata_only: bool = False,
    ) -> None:
        self.config = config or CacheConfig()
        for hook in (admit, verifier):
            if hook is not None and (
                not callable(hook)
                or inspect.iscoroutinefunction(hook)
                or inspect.iscoroutinefunction(getattr(hook, "__call__", None))
            ):
                raise ValueError("Policy hooks must be synchronous callables")
        if context is not None and not isinstance(context, RequestContext):
            raise ValueError("context must be a RequestContext")
        self._metadata_only = metadata_only
        self.context = context
        self._admit: Callable[..., Any] | None = admit
        self._verifier: Callable[..., Any] | None = verifier
        self._mutation_lock = threading.RLock()
        self._stats: Counter[str] = Counter()
        self._partition_stats: OrderedDict[str, Counter[str]] = OrderedDict()
        self._stats_started_at = time.time()
        self._stats_evictions = 0
        self._closed = False
        self._coherent = True
        self._epoch = 0
        self._shared_revision = -1
        self._shared_compaction_pending = False
        self._background_failed = False
        self._observability = Observability(self.config.track_stats, self.config.backend)
        self._ready = False
        self._stop_event = threading.Event()
        self.embedder: BaseEmbedder = (
            embedder
            if embedder is not None
            else LocalEmbedder(self.config.embedder_model, self.config.embedder_revision)
        )
        self._embedding = EmbeddingGuard(self.embedder)
        self.backend: Any = self._init_backend()
        self.index = SimilarityIndex(self.backend, self.config)
        self._lifecycle = Lifecycle(
            self._finalize_resources, self._revoke_flights, self.config.shutdown_timeout_seconds
        )
        self._execution = Execution(self)
        self._distributed_execution: DistributedExecution | None = (
            DistributedExecution(self) if self.config.deployment_mode == "shared" else None
        )
        self._fingerprint: str = ""
        # Metadata-only construction does not load/download model weights.
        try:
            if self._dependency("backend", self.backend.list_all_entries):
                self._ensure_ready()
        except CacheOperationError:
            if self.config.cache_error_policy != "bypass":
                self.backend.close() if hasattr(self.backend, "close") else None
                raise
        except Exception:
            if hasattr(self.backend, "close"):
                self.backend.close()
            raise
        self._sweep_thread: threading.Thread | None = None
        if (
            not metadata_only
            and self.config.backend != "memory"
            and (self.config.ttl_seconds or self.config.deployment_mode == "shared")
            and hasattr(self.backend, "sweep_expired")
        ):
            self._sweep_thread = threading.Thread(
                target=self._sweep, daemon=True, name="longcache-sweeper"
            )
            self._sweep_thread.start()

    def _init_backend(self) -> Any:
        c = self.config
        if c.backend == "memory":
            return InMemoryCacheBackend(c.max_entries)
        if c.backend == "sqlite":
            return SQLiteCacheBackend(c.db_path, c.max_entries, timeout=c.backend_timeout_seconds)
        if c.backend == "redis":
            if c.deployment_mode == "shared":
                from longcache.backends.redis_shared import SharedRedisCacheBackend

                return SharedRedisCacheBackend(
                    redis_url=c.redis_url,
                    max_entries=c.max_entries,
                    cache_id=c.cache_id,
                    generation=c.generation,
                    timeout=c.backend_timeout_seconds,
                    shared_max_entries=c.shared_max_entries,
                    shared_max_leases=c.shared_max_leases,
                    shared_max_receipts=c.shared_max_receipts,
                    shared_max_partitions=c.shared_max_partitions,
                    shared_max_distinct_tags=c.shared_max_distinct_tags,
                    shared_max_tags_per_entry=c.shared_max_tags_per_entry,
                    shared_max_tag_chars=c.shared_max_tag_chars,
                    distributed_lease_seconds=c.distributed_lease_seconds,
                    distributed_renew_interval_seconds=c.distributed_renew_interval_seconds,
                )
            from longcache.backends.redis import RedisCacheBackend

            return RedisCacheBackend(
                c.redis_url,
                c.max_entries,
                c.cache_id,
                c.generation,
                timeout=c.backend_timeout_seconds,
            )
        raise ValueError("async_redis is not supported through LongCache; use a supported backend")

    def _ensure_open(self) -> None:
        self._lifecycle.ensure_open()

    def _dependency(self, stage: str, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return dependency(self, stage, function, *args, **kwargs)

    def _ensure_ready(self) -> None:
        with self._mutation_lock:
            self._ensure_open()
            if not self._coherent:
                raise CacheOperationError("coherence_uncertain")
            if self._metadata_only:
                manifest = self._dependency("backend", self.backend.read_manifest)
                if manifest and (
                    manifest.get("format") != 2
                    or manifest.get("cache_id") != self.config.cache_id
                    or manifest.get("generation") != self.config.generation
                ):
                    raise ValueError("Incompatible storage generation")
                self._fingerprint = digest(manifest["embedding"]) if manifest else "empty-store"
                self._ready = True
                return
            descriptor = self._dependency("embedding", lambda: self._embedding.descriptor)
            if self._ready:
                return
            self._dependency(
                "backend",
                self.backend.ensure_manifest,
                {
                    "format": 2,
                    "cache_id": self.config.cache_id,
                    "generation": self.config.generation,
                    "embedding": asdict(descriptor),
                },
            )
            if not self._ready:
                self._fingerprint = descriptor.fingerprint
                before = (
                    self._dependency("backend", self.backend.revision)
                    if self.config.deployment_mode == "shared"
                    else None
                )
                self._rebuild_index_from_backend()
                if before is not None and before == self._dependency(
                    "backend", self.backend.revision
                ):
                    self._shared_revision = before
                self._ready = True

    def _rebuild_index_from_backend(self) -> None:
        if self._metadata_only:
            return
        entries: list[CacheEntry] = []
        for entry in self._dependency("backend", self.backend.list_all_entries):
            metadata = entry.metadata
            if metadata.get("format") != 2 or metadata.get("embedding") != self._fingerprint:
                continue
            try:
                validate_vector(entry.vector, self._embedding.descriptor.dimension)
            except ValueError:
                continue
            entries.append(entry)
        self._dependency("index", self.index.rebuild, entries)

    def _synchronize_shared(self) -> None:
        """Never search a local index without checking authoritative Redis state."""
        if self.config.deployment_mode != "shared":
            return
        deadline = time.monotonic() + self.config.shared_sync_timeout_seconds
        with self._mutation_lock:
            while True:
                before = self._dependency("backend", self.backend.revision)
                if self._shared_revision == before:
                    return
                if 0 <= self._shared_revision < before:
                    cursor = self._shared_revision
                    complete = True
                    while cursor < before and complete:
                        events = self._dependency(
                            "backend", self.backend.events, cursor + 1, before
                        )
                        if not events:
                            complete = False
                            break
                        for sequence, event in events:
                            if sequence != cursor + 1:
                                complete = False
                                break
                            kind = event.get("type")
                            if kind == "store":
                                entry = self._dependency(
                                    "backend", self.backend.retrieve, int(event["id"])
                                )
                                if entry is not None:
                                    self._dependency(
                                        "index", self.index.add, entry.vector, entry.id
                                    )
                            elif kind in ("delete", "invalidate"):
                                # Redis validates every candidate at replay. Old vectors
                                # are harmless until the background index compacts them.
                                self._shared_compaction_pending = True
                            else:
                                complete = False
                                break
                            cursor = sequence
                        if time.monotonic() >= deadline:
                            raise CacheOperationError("synchronization")
                    if complete and cursor == before:
                        self._shared_revision = before
                        return
                self._rebuild_index_from_backend()
                after = self._dependency("backend", self.backend.revision)
                if before == after:
                    self._shared_revision = after
                    return
                if time.monotonic() >= deadline:
                    raise CacheOperationError("synchronization")

    def _prepare(
        self,
        query: str,
        context: RequestContext | None,
        namespace: str | None,
        *,
        _initialize: bool = True,
    ) -> tuple[tuple[RequestContext, str, str] | None, str | None]:
        self._ensure_open()
        ctx = context if context is not None else self.context
        if not isinstance(ctx, RequestContext):
            return None, "missing_identity"
        policy = self.config.scope_policy
        if policy != "public" and (
            ctx.scope is None or (policy == "private" and not ctx.scope.user_id)
        ):
            return None, "missing_identity"
        if (
            type(query) is not str
            or not query.strip()
            or len(query) > self.config.max_query_chars
            or ctx.request_shape != "single-turn-text"
        ):
            return None, "unsupported_request_shape"
        try:
            query.encode("utf-8")
        except UnicodeError:
            return None, "unsupported_request_shape"
        if ctx.codec not in ("json-v1", "openai-dict-v1", "openai-sdk-v1"):
            return None, "unsupported_request_shape"
        if type(namespace or self.config.namespace) is not str:
            raise ValueError("Namespace must be a string")
        if _initialize:
            self._ensure_ready()
        ns = namespace if namespace is not None else self.config.namespace
        record = ctx.record(ns, policy, self._fingerprint, self.config.policy_version)
        identity = canonical(record)
        partition = "v2:" + digest(record)
        return (ctx, partition, identity), None

    def _account(self, partition: str, outcome: str) -> None:
        if self.config.track_stats:
            self._stats[outcome] += 1
            if partition not in self._partition_stats:
                if len(self._partition_stats) >= 1024:
                    self._partition_stats.popitem(last=False)
                    self._stats_evictions += 1
                self._partition_stats[partition] = Counter()
                self._partition_stats[partition]["_started_at"] = int(time.time())
            self._partition_stats.move_to_end(partition)
            self._partition_stats[partition][outcome] += 1

    def lookup(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> LookupDecision:
        return cast(LookupDecision, self._lookup(query, namespace, context=context))

    @managed
    @decision_operation("lookup")
    def _lookup(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
        _inspect: bool = False,
        _record: bool = True,
        _cancel: Any = None,
        _internal: bool = False,
    ) -> LookupDecision:
        _cancel = _cancel or self._lifecycle.current
        if self._metadata_only:
            raise ValueError("Vector operations require a normal cache instance")
        prepared, reason = self._prepare(query, context, namespace)
        threshold = self.config.similarity_threshold
        if reason or prepared is None:
            return LookupDecision("BYPASS", reason or "missing_identity", threshold=threshold)
        ctx, partition, identity = prepared
        self._synchronize_shared()
        with self._mutation_lock:
            if self.index.size == 0:
                if not _inspect and _record and not (_cancel and _cancel.is_set()):
                    self._account(partition, "misses")
                return LookupDecision("MISS", "no_candidate", threshold=threshold)
        vector = self._dependency("embedding", self._embedding.embed, query)
        with self._mutation_lock:
            candidates = self._dependency(
                "index",
                self.index.candidates,
                vector,
                partition,
                threshold,
                self.config.verification_budget + 1,
                retrieve=lambda identifier: self._dependency(
                    "backend", self.backend.retrieve, identifier
                ),
            )
        reason, last_score = "no_candidate", None
        for entry, score in candidates[: self.config.verification_budget]:
            expected = {"format": 2, "identity": identity, "embedding": self._fingerprint}
            if not isinstance(entry.metadata, dict) or any(
                entry.metadata.get(k) != v for k, v in expected.items()
            ):
                reason = "legacy_format"
                continue
            try:
                if entry.metadata.get("payload") != digest(entry.response):
                    reason = "codec_invalid"
                    continue
                response = decode_response(entry.response, ctx.codec)
                payload, admission = admit(response, ctx, self._admit)
                if admission.action != "accept":
                    reason = admission.reason
                    continue
            except (ValueError, TypeError, ImportError, AttributeError):
                reason = "codec_invalid"
                continue
            last_score = score
            verification = self._dependency(
                "verification", verify, query, entry.query, payload, score, ctx, self._verifier
            )
            self._observability.count(
                "verification",
                outcome=verification.action,
                reason=verification.reason,
            )
            if verification.action != "accept":
                reason = verification.reason
                continue
            with self._mutation_lock:
                self._ensure_open()
                shared_epoch = (
                    self._dependency("backend", self.backend.epoch)
                    if self.config.deployment_mode == "shared"
                    else None
                )
                current = self._dependency("backend", self.backend.retrieve, entry.id)
                if (
                    current is None
                    or current.namespace != partition
                    or current.metadata != entry.metadata
                    or current.is_expired()
                ):
                    reason = "expired_or_deleted"
                    continue
                # The backend may be mutable: validate the final payload, not a stale snapshot.
                try:
                    if digest(current.response) != entry.metadata["payload"]:
                        reason = "codec_invalid"
                        continue
                except (TypeError, ValueError):
                    reason = "codec_invalid"
                    continue
                if _cancel is not None and _cancel.is_set():
                    return LookupDecision("BYPASS", "cancelled", threshold=threshold)
                if shared_epoch is not None and not self._dependency(
                    "backend", self.backend.validate_hit, current, expected_epoch=shared_epoch
                ):
                    reason = "expired_or_deleted"
                    continue
                if not _inspect and _record:
                    try:
                        self._dependency("accounting", self.backend.update_hit_count, entry.id)
                    except CacheOperationError:
                        pass  # Optional persisted accounting cannot invalidate a verified hit.
                    self._account(partition, "hits")
                return LookupDecision(
                    "HIT",
                    "accepted",
                    response,
                    score,
                    entry.id,
                    threshold,
                    _epoch=self._epoch,
                    _expires_at=current.expires_at,
                    _shared_epoch=shared_epoch,
                )
        if len(candidates) > self.config.verification_budget:
            reason = "verification_budget_exhausted"
        if not _inspect and _record:
            with self._mutation_lock:
                if _cancel is None or not _cancel.is_set():
                    self._account(partition, "misses")
        return LookupDecision("MISS", reason, similarity=last_score, threshold=threshold)

    async def _async(self, method: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await self._lifecycle.call(method, *args, **kwargs)

    async def alookup(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> LookupDecision:
        return await self._async(self.lookup, query, namespace, context=context)

    async def _alookup(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
        _inspect: bool = False,
        _record: bool = True,
        _cancel: Any = None,
        _internal: bool = False,
    ) -> LookupDecision:
        kwargs: dict[str, Any] = {
            "context": context,
            "_inspect": _inspect,
            "_record": _record,
            "_internal": _internal,
        }
        if _cancel is not None:
            kwargs["_cancel"] = _cancel
        return await self._async(self._lookup, query, namespace, **kwargs)

    def get(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> Any:
        return self.lookup(query, namespace, context=context).response

    async def aget(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> Any:
        return (await self.alookup(query, namespace, context=context)).response

    def set(
        self,
        query: str,
        response: Any,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
        *,
        context: RequestContext | None = None,
    ) -> StoreDecision:
        return cast(
            StoreDecision,
            self._set(
                query,
                response,
                namespace,
                ttl=ttl,
                tags=tags,
                context=context,
            ),
        )

    @managed
    @decision_operation("store")
    def _set(
        self,
        query: str,
        response: Any,
        namespace: str | None = None,
        *,
        ttl: int | None = None,
        tags: Any = None,
        context: RequestContext | None = None,
        _cancel: Any = None,
        _expected_epoch: int | None = None,
        _expected_shared_epoch: int | None = None,
        _lease_key: str = "",
        _lease_token: str = "",
        _internal: bool = False,
    ) -> StoreDecision:
        _cancel = _cancel or self._lifecycle.current
        if _expected_epoch is None:
            with self._mutation_lock:
                _expected_epoch = self._epoch
        if self.config.deployment_mode == "shared" and _expected_shared_epoch is None:
            _expected_shared_epoch = self._dependency("backend", self.backend.epoch)
        if self._metadata_only:
            raise ValueError("Vector operations require a normal cache instance")
        prepared, reason = self._prepare(query, context, namespace)
        if reason or prepared is None:
            return StoreDecision("REJECTED", reason or "missing_identity")
        ctx, partition, identity = prepared
        payload, decision = self._dependency("admission", admit, response, ctx, self._admit)
        self._observability.count(
            "admission",
            outcome=decision.action,
            reason=decision.reason,
        )
        if decision.action != "accept":
            return StoreDecision("REJECTED", decision.reason)
        if len(canonical(payload).encode("utf-8")) > self.config.max_capture_bytes:
            return StoreDecision("REJECTED", "response_too_large")
        effective_ttl = self.config.ttl_seconds if ttl is None else ttl
        if type(effective_ttl) is not int or effective_ttl < 0:
            raise ValueError("TTL must be a nonnegative integer")
        if tags is not None and (
            not isinstance(tags, (list, tuple)) or not all(type(t) is str for t in tags)
        ):
            raise ValueError("Tags must be strings")
        if self.config.deployment_mode == "shared" and tags is not None:
            if len(tags) > self.config.shared_max_tags_per_entry or any(
                len(tag) > self.config.shared_max_tag_chars for tag in tags
            ):
                return StoreDecision("REJECTED", "tag_capacity")
            tags = tuple(dict.fromkeys(tags))
        vector = self._dependency("embedding", self._embedding.embed, query)
        now = time.time()
        entry = CacheEntry(
            namespace=partition,
            query=query,
            response=payload,
            vector=vector,
            created_at=now,
            expires_at=now + effective_ttl if effective_ttl else None,
            tags=list(tags or []),
            metadata={
                "format": 2,
                "identity": identity,
                "embedding": self._fingerprint,
                "payload": digest(payload),
            },
        )
        with self._mutation_lock:
            self._ensure_open()
            if _expected_epoch != self._epoch:
                return StoreDecision("REJECTED", "invalidated")
            if _cancel is not None and _cancel.is_set():
                return StoreDecision("REJECTED", "cancelled")
            try:
                if self.config.deployment_mode == "shared":
                    identifier, commit_revision = self._dependency(
                        "store",
                        self.backend.store_shared,
                        entry,
                        expected_epoch=_expected_shared_epoch,
                        lease_key=_lease_key,
                        lease_token=_lease_token,
                    )
                    if identifier == 0:
                        return StoreDecision("REJECTED", "invalidated")
                    if identifier == -1:
                        return StoreDecision("REJECTED", "capacity")
                else:
                    identifier = self._dependency("store", self.backend.store, entry)
                try:
                    self._dependency("index", self.index.add, vector, identifier)
                except Exception:
                    if self.config.deployment_mode != "shared":
                        self._dependency("store", self.backend.delete, identifier)
                    raise
                if (
                    self.config.deployment_mode == "shared"
                    and commit_revision is not None
                    and self._shared_revision == commit_revision - 1
                ):
                    self._shared_revision = commit_revision
                if self.config.deployment_mode != "shared":
                    excess = (
                        self._dependency("backend", self.backend.count, partition)
                        - self.config.max_entries
                    )
                    if excess > 0:
                        self._dependency("store", self.backend.evict_oldest, partition, excess)
                        self._rebuild_index_from_backend()
            except BaseException:
                self._coherent = False
                raise
        return StoreDecision("STORED", "accepted", identifier)

    async def aset(
        self,
        query: str,
        response: Any,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
        *,
        context: RequestContext | None = None,
    ) -> StoreDecision:
        return await self._async(
            self.set,
            query,
            response,
            namespace,
            ttl=ttl,
            tags=tags,
            context=context,
        )

    async def _aset(
        self,
        query: str,
        response: Any,
        namespace: str | None = None,
        *,
        ttl: int | None = None,
        tags: Any = None,
        context: RequestContext | None = None,
        _cancel: Any = None,
        _expected_epoch: int | None = None,
        _expected_shared_epoch: int | None = None,
        _lease_key: str = "",
        _lease_token: str = "",
        _internal: bool = False,
    ) -> StoreDecision:
        kwargs: dict[str, Any] = {
            "ttl": ttl,
            "tags": tags,
            "context": context,
            "_expected_epoch": _expected_epoch,
            "_expected_shared_epoch": _expected_shared_epoch,
            "_lease_key": _lease_key,
            "_lease_token": _lease_token,
            "_internal": _internal,
        }
        if _cancel is not None:
            kwargs["_cancel"] = _cancel
        return await self._async(self._set, query, response, namespace, **kwargs)

    def get_or_compute(
        self,
        query: str,
        compute: Callable[[], T],
        *,
        context: RequestContext | None = None,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
    ) -> T:
        """Reuse a verified answer or coordinate one exact scoped computation."""
        return self._run_compute(
            query,
            compute,
            context=context,
            namespace=namespace,
            ttl=ttl,
            tags=tags,
            _entry_point="manual",
        )

    @execution_operation
    def _run_compute(
        self,
        query: str,
        compute: Callable[[], T],
        *,
        context: RequestContext | None = None,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Any = None,
        _entry_point: str = "manual",
    ) -> T:
        if self._distributed_execution is not None:
            return cast(
                T,
                self._distributed_execution.run(
                    query,
                    compute,
                    context=context,
                    namespace=namespace,
                    ttl=ttl,
                    tags=tags,
                    entry_point=_entry_point,
                ),
            )
        return cast(
            T,
            self._execution.run(
                query,
                compute,
                context=context,
                namespace=namespace,
                ttl=ttl,
                tags=tags,
                _entry_point=_entry_point,
            ),
        )

    async def aget_or_compute(
        self,
        query: str,
        compute: Callable[[], Awaitable[T]],
        *,
        context: RequestContext | None = None,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
    ) -> T:
        """Async equivalent; followers await without occupying worker threads."""
        return await self._arun_compute(
            query,
            compute,
            context=context,
            namespace=namespace,
            ttl=ttl,
            tags=tags,
            _entry_point="manual",
        )

    @execution_operation
    async def _arun_compute(
        self,
        query: str,
        compute: Callable[[], Awaitable[T]],
        *,
        context: RequestContext | None = None,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Any = None,
        _entry_point: str = "manual",
    ) -> T:
        if self._distributed_execution is not None:
            return cast(
                T,
                await self._distributed_execution.arun(
                    query,
                    compute,
                    context=context,
                    namespace=namespace,
                    ttl=ttl,
                    tags=tags,
                    entry_point=_entry_point,
                ),
            )
        return cast(
            T,
            await self._execution.arun(
                query,
                compute,
                context=context,
                namespace=namespace,
                ttl=ttl,
                tags=tags,
                _entry_point=_entry_point,
            ),
        )

    @overload
    def cached(
        self,
        func: Callable[P, R],
        *,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
        context: RequestContext | None = None,
        context_resolver: Callable[..., RequestContext | None] | None = None,
    ) -> CachedCallable[P, R]: ...

    @overload
    def cached(
        self,
        func: None = None,
        *,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
        context: RequestContext | None = None,
        context_resolver: Callable[..., RequestContext | None] | None = None,
    ) -> Callable[[Callable[P, R]], CachedCallable[P, R]]: ...

    def cached(
        self,
        func: Callable[..., Any] | None = None,
        *,
        namespace: str | None = None,
        ttl: int | None = None,
        tags: Sequence[str] | None = None,
        context: RequestContext | None = None,
        context_resolver: Callable[..., RequestContext | None] | None = None,
    ) -> Any:
        if context_resolver is not None and (
            not callable(context_resolver)
            or inspect.iscoroutinefunction(context_resolver)
            or inspect.iscoroutinefunction(getattr(context_resolver, "__call__", None))
        ):
            raise ValueError("Context resolvers must be synchronous callables")

        def decorate(fn: Callable[..., Any]) -> Any:
            def context_for(base: RequestContext | None) -> RequestContext | None:
                if not isinstance(base, RequestContext):
                    if inspect.iscoroutine(base):
                        base.close()
                    return None
                return replace(
                    base, operation=base.operation + ":" + fn.__module__ + "." + fn.__qualname__
                )

            def resolve(args: tuple[Any, ...], kwargs: dict[str, Any]) -> RequestContext | None:
                if context_resolver is None:
                    return context_for(context or self.context)
                try:
                    return context_for(context_resolver(*args, **kwargs))
                except Exception:
                    return None

            if inspect.iscoroutinefunction(fn):

                @wraps(fn)
                async def async_wrapper(query: str, *args: Any, **kwargs: Any) -> Any:
                    if args or kwargs:
                        return await self._apass_through(
                            lambda: fn(query, *args, **kwargs), "unsupported_parameter", "decorator"
                        )
                    ctx = resolve((query,), {})
                    if ctx is None:
                        return await self._apass_through(
                            lambda: fn(query), "missing_identity", "decorator"
                        )
                    return await self._arun_compute(
                        query,
                        lambda: fn(query),
                        context=ctx,
                        namespace=namespace,
                        ttl=ttl,
                        tags=tags,
                        _entry_point="decorator",
                    )

                async_wrapper.cache_context_for = context_for  # type: ignore[attr-defined]
                return async_wrapper

            @wraps(fn)
            def wrapper(query: str, *args: Any, **kwargs: Any) -> Any:
                if args or kwargs:
                    return self._pass_through(
                        lambda: fn(query, *args, **kwargs), "unsupported_parameter", "decorator"
                    )
                ctx = resolve((query,), {})
                if ctx is None:
                    return self._pass_through(lambda: fn(query), "missing_identity", "decorator")
                return self._run_compute(
                    query,
                    lambda: fn(query),
                    context=ctx,
                    namespace=namespace,
                    ttl=ttl,
                    tags=tags,
                    _entry_point="decorator",
                )

            wrapper.cache_context_for = context_for  # type: ignore[attr-defined]
            return wrapper

        return decorate(func) if func is not None else decorate

    @execution_operation
    def _pass_through(self, compute: Callable[[], Any], reason: str, entry_point: str) -> Any:
        with self._lifecycle.operation():
            result = self._execution.origin(compute)
            return self._execution.finish(
                None,
                None,
                result,
                "origin",
                "sync",
                "none",
                entry_point,
                self._epoch,
                reason=reason,
            )

    @execution_operation
    async def _apass_through(
        self, compute: Callable[[], Any], reason: str, entry_point: str
    ) -> Any:
        with self._lifecycle.operation():
            result = await self._execution.aorigin(compute)
            return await self._execution.afinish(
                None,
                None,
                result,
                "origin",
                "async",
                "none",
                entry_point,
                self._epoch,
                reason=reason,
            )

    def cached_stream(self, func: Any = None, **kwargs: Any) -> Any:
        # Preserve the generator interface while unsupported replay remains pass-through.
        return func if func is not None else lambda fn: fn

    def inspect(
        self,
        query: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> InspectionReport:
        return cast(
            InspectionReport,
            self._lookup(
                query, namespace, context=context, _inspect=True, _internal=True
            ).inspection(),
        )

    @managed
    def invalidate(
        self,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> int:
        prepared, reason = self._prepare("maintenance", context, namespace)
        if reason or prepared is None:
            return 0
        with self._mutation_lock:
            try:
                self._epoch += 1
                self._execution.registry.abort()
                count = int(self._dependency("store", self.backend.delete_namespace, prepared[1]))
                revision = (
                    self._dependency("backend", self.backend.revision)
                    if self.config.deployment_mode == "shared"
                    else None
                )
                self._rebuild_index_from_backend()
                if revision is not None and revision == self._dependency(
                    "backend", self.backend.revision
                ):
                    self._shared_revision = revision
                return count
            except BaseException:
                self._coherent = False
                raise

    @managed
    def invalidate_by_tag(
        self,
        tag: str,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
    ) -> int:
        prepared, reason = self._prepare("maintenance", context, namespace)
        if reason or prepared is None:
            return 0
        with self._mutation_lock:
            try:
                self._epoch += 1
                self._execution.registry.abort()
                count = int(self._dependency("store", self.backend.delete_by_tag, tag, prepared[1]))
                revision = (
                    self._dependency("backend", self.backend.revision)
                    if self.config.deployment_mode == "shared"
                    else None
                )
                self._rebuild_index_from_backend()
                if revision is not None and revision == self._dependency(
                    "backend", self.backend.revision
                ):
                    self._shared_revision = revision
                return count
            except BaseException:
                self._coherent = False
                raise

    @managed
    def clear(self, *, operator: bool = False) -> int:
        self._ensure_open()
        if not operator:
            raise ValueError("Clearing all partitions requires explicit operator=True")
        with self._mutation_lock:
            try:
                self._epoch += 1
                self._execution.registry.abort()
                count = int(self._dependency("store", self.backend.clear))
                revision = (
                    self._dependency("backend", self.backend.revision)
                    if self.config.deployment_mode == "shared"
                    else None
                )
                self._dependency("index", self.index.rebuild, [])
                if revision is not None and revision == self._dependency(
                    "backend", self.backend.revision
                ):
                    self._shared_revision = revision
                return count
            except BaseException:
                self._coherent = False
                raise

    @managed
    def entries(
        self,
        namespace: str | None = None,
        *,
        context: RequestContext | None = None,
        operator: bool = False,
    ) -> list[CacheEntry]:
        self._ensure_open()
        with self._mutation_lock:
            if operator:
                return list(self.backend.list_all_entries())
            prepared, reason = self._prepare("maintenance", context, namespace)
            return (
                [] if reason or prepared is None else list(self.backend.list_entries(prepared[1]))
            )

    @managed
    def stats(
        self,
        *,
        context: RequestContext | None = None,
        namespace: str | None = None,
        operator: bool = False,
    ) -> StatsSnapshot:
        self._ensure_open()
        with self._mutation_lock:
            if operator:
                counts = self._stats.copy()
                total = len(self.backend.list_all_entries())
            else:
                prepared, reason = self._prepare("maintenance", context, namespace)
                counts = (
                    Counter()
                    if reason or prepared is None
                    else self._partition_stats.get(prepared[1], Counter()).copy()
                )
                total = 0 if reason or prepared is None else self.backend.count(prepared[1])
            hits, misses = counts["hits"], counts["misses"]
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
                "total_entries": total,
                "total_saved_ms": 0.0,
                "estimated_tokens_saved": 0,
                "estimated_cost_saved_usd": 0.0,
                "stats_window_started_at": self._stats_started_at
                if operator
                else counts.get("_started_at"),
                "stats_retention_evictions": self._stats_evictions,
            }

    def _sweep(self) -> None:
        delay, failures = 60.0, 0
        while not self._stop_event.wait(delay):
            try:
                with self._lifecycle.operation(), self._mutation_lock:
                    if not self._coherent:
                        return
                    removed = self._dependency("sweeper", self.backend.sweep_expired)
                    if (removed or self._shared_compaction_pending) and self._ready:
                        try:
                            self._rebuild_index_from_backend()
                            self._shared_compaction_pending = False
                        except BaseException:
                            self._coherent = False
                            raise
            except CacheClosedError:
                return
            except Exception as exc:
                self._observability.event(warning=True, stage="sweeper", reason="background_failed")
                if not isinstance(exc, CacheOperationError) and not operational(exc, "sweeper"):
                    self._background_failed = True
                    return
                failures += 1
                delay = float(min(60, 2 ** min(failures - 1, 6)))
            else:
                delay, failures = 60.0, 0

    def metrics(self, *, operator: bool = False) -> MetricsSnapshot:
        """Return process-wide redacted measurements without backend I/O."""
        if not operator:
            raise ValueError("Process-wide metrics require operator=True")
        obs = self._observability.snapshot()
        life = self._lifecycle.snapshot()
        reg = self._execution.registry.snapshot()
        return {
            "enabled": bool(obs["enabled"]),
            "counters": obs["counters"],
            "latency": obs["latency"],
            "resources": {
                "state": str(life["state"]),
                "active_operations": int(life["active_operations"]),
                "cache_jobs": int(life["cache_jobs"]),
                "cleanup_failed": bool(life["cleanup_failed"]),
                "cleanup_workers": int(life["cleanup_workers"]),
                "active_flights": int(reg["active_flights"]),
                "followers": int(reg["followers"]),
                "retained_replay_bytes": int(reg["retained_replay_bytes"]),
                "sqlite_connections": int(getattr(self.backend, "_connection", None) is not None),
            },
            "health": "degraded"
            if not self._coherent or self._background_failed or life["cleanup_failed"]
            else "healthy",
        }

    def _revoke_flights(self) -> None:
        if hasattr(self, "_execution"):
            self._execution.shutdown()

    def _finalize_resources(self) -> None:
        if self._sweep_thread is not None and self._sweep_thread is not threading.current_thread():
            self._sweep_thread.join()
        if hasattr(self.backend, "close"):
            self.backend.close()
        self._closed = True

    def close(self, timeout: float | None = None) -> None:
        if self._lifecycle.current is not None:
            raise CacheClosedError("Cannot close the cache from its own active operation")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if self._lifecycle.snapshot()["active_operations"]:
                raise CacheClosedError("Use await cache.aclose() to drain active work")
        self._stop_event.set()
        self._lifecycle.close(timeout)

    async def aclose(self, timeout: float | None = None) -> None:
        if self._lifecycle.current is not None:
            raise CacheClosedError("Cannot close the cache from its own active operation")
        self._stop_event.set()
        await self._lifecycle.aclose(timeout)

    def __enter__(self) -> LongCache:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except Exception:
            self._observability.event(warning=True, stage="lifecycle", reason="shutdown")
            if exc is None:
                raise

    async def __aenter__(self) -> LongCache:
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            await self.aclose()
        except Exception:
            self._observability.event(warning=True, stage="lifecycle", reason="shutdown")
            if exc is None:
                raise

    def __repr__(self) -> str:
        return f"LongCache(backend={self.config.backend!r}, format=2)"
