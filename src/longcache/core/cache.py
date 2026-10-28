"""Scoped semantic reuse with shared policy across sync and async entry points."""

from __future__ import annotations
import asyncio
from collections import Counter
from dataclasses import asdict, replace
from functools import wraps
import inspect
import threading
import time

from longcache.config import CacheConfig
from longcache.backends.memory import InMemoryCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend
from .base_embedder import BaseEmbedder
from .decision import LookupDecision, StoreDecision
from .embedding_contract import EmbeddingGuard, validate_vector
from .embedder import LocalEmbedder
from .entry import CacheEntry
from .identity import Instruction, RequestContext, canonical, digest
from .index import SimilarityIndex
from .policy import admit, decode_response, verify


class LongCache:
    def __init__(
        self,
        config=None,
        *,
        embedder: BaseEmbedder | None = None,
        admit=None,
        verifier=None,
        context: RequestContext | None = None,
        metadata_only: bool = False,
    ):
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
        self._admit, self._verifier = admit, verifier
        self._mutation_lock = threading.RLock()
        self._stats = Counter()
        self._partition_stats = {}
        self._closed = False
        self._ready = False
        self._stop_event = threading.Event()
        self.embedder = (
            embedder
            if embedder is not None
            else LocalEmbedder(self.config.embedder_model, self.config.embedder_revision)
        )
        self._embedding = EmbeddingGuard(self.embedder)
        self.backend = self._init_backend()
        self.index = SimilarityIndex(self.backend, self.config)
        # Metadata-only construction does not load/download model weights.
        try:
            if self.backend.list_all_entries():
                self._ensure_ready()
        except Exception:
            if hasattr(self.backend, "close"):
                self.backend.close()
            raise
        self._sweep_thread = None
        if not metadata_only and self.config.ttl_seconds and hasattr(self.backend, "sweep_expired"):
            self._sweep_thread = threading.Thread(
                target=self._sweep, daemon=True, name="longcache-sweeper"
            )
            self._sweep_thread.start()

    def _init_backend(self):
        c = self.config
        if c.backend == "memory":
            return InMemoryCacheBackend(c.max_entries)
        if c.backend == "sqlite":
            return SQLiteCacheBackend(c.db_path, c.max_entries)
        if c.backend == "redis":
            from longcache.backends.redis import RedisCacheBackend

            return RedisCacheBackend(c.redis_url, c.max_entries, c.cache_id, c.generation)
        raise ValueError("async_redis is not supported through LongCache; use a supported backend")

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Cache is closed")

    def _ensure_ready(self):
        with self._mutation_lock:
            self._ensure_open()
            if self._metadata_only:
                manifest = self.backend.read_manifest()
                if manifest and (
                    manifest.get("format") != 2
                    or manifest.get("cache_id") != self.config.cache_id
                    or manifest.get("generation") != self.config.generation
                ):
                    raise ValueError("Incompatible storage generation")
                self._fingerprint = digest(manifest["embedding"]) if manifest else "empty-store"
                self._ready = True
                return
            descriptor = self._embedding.descriptor
            if self._ready:
                return
            self.backend.ensure_manifest(
                {
                    "format": 2,
                    "cache_id": self.config.cache_id,
                    "generation": self.config.generation,
                    "embedding": asdict(descriptor),
                }
            )
            if not self._ready:
                self._fingerprint = descriptor.fingerprint
                self._rebuild_index_from_backend()
                self._ready = True

    def _rebuild_index_from_backend(self):
        if self._metadata_only:
            return
        entries = []
        for entry in self.backend.list_all_entries():
            metadata = entry.metadata
            if metadata.get("format") != 2 or metadata.get("embedding") != self._fingerprint:
                continue
            try:
                validate_vector(entry.vector, self._embedding.descriptor.dimension)
            except ValueError:
                continue
            entries.append(entry)
        self.index.rebuild(entries)

    def _prepare(self, query, context, namespace, legacy):
        self._ensure_open()
        if context is not None and any(value is not None for value in legacy.values()):
            raise ValueError("Legacy context keywords cannot override RequestContext")
        ctx = context if context is not None else self.context
        if not isinstance(ctx, RequestContext):
            return None, "missing_identity"
        if any(value is not None for value in legacy.values()):
            updates = {}
            if legacy.get("model") is not None:
                updates["model"] = legacy["model"]
            if legacy.get("system_prompt") is not None:
                updates["instructions"] = (Instruction("system", legacy["system_prompt"]),)
            if legacy.get("temperature") is not None:
                updates["generation"] = dict(ctx.generation) | {
                    "temperature": legacy["temperature"]
                }
            ctx = replace(ctx, **updates)
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
        self._ensure_ready()
        ns = namespace if namespace is not None else self.config.namespace
        record = ctx.record(ns, policy, self._fingerprint, self.config.policy_version)
        identity = canonical(record)
        partition = "v2:" + digest(record)
        return (ctx, partition, identity), None

    def _account(self, partition, outcome):
        if self.config.track_stats:
            self._stats[outcome] += 1
            self._partition_stats.setdefault(partition, Counter())[outcome] += 1

    def lookup(
        self,
        query,
        namespace=None,
        *,
        context=None,
        model=None,
        system_prompt=None,
        temperature=None,
        _inspect=False,
        _cancel=None,
    ):
        if self._metadata_only:
            raise ValueError("Vector operations require a normal cache instance")
        prepared, reason = self._prepare(
            query,
            context,
            namespace,
            dict(model=model, system_prompt=system_prompt, temperature=temperature),
        )
        threshold = self.config.similarity_threshold
        if reason:
            return LookupDecision("BYPASS", reason, threshold=threshold)
        ctx, partition, identity = prepared
        vector = self._embedding.embed(query)
        with self._mutation_lock:
            candidates = self.index.candidates(
                vector, partition, threshold, self.config.verification_budget + 1
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
            verification = verify(query, entry.query, payload, score, ctx, self._verifier)
            if verification.action != "accept":
                reason = verification.reason
                continue
            with self._mutation_lock:
                self._ensure_open()
                current = self.backend.retrieve(entry.id)
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
                if not _inspect:
                    self.backend.update_hit_count(entry.id)
                    self._account(partition, "hits")
                return LookupDecision("HIT", "accepted", response, score, entry.id, threshold)
        if len(candidates) > self.config.verification_budget:
            reason = "verification_budget_exhausted"
        if not _inspect:
            with self._mutation_lock:
                if _cancel is None or not _cancel.is_set():
                    self._account(partition, "misses")
        return LookupDecision("MISS", reason, similarity=last_score, threshold=threshold)

    async def _async(self, method, *args, **kwargs):
        cancelled = threading.Event()
        try:
            return await asyncio.to_thread(method, *args, _cancel=cancelled, **kwargs)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def alookup(self, query, namespace=None, **kwargs):
        return await self._async(self.lookup, query, namespace, **kwargs)

    def get(
        self,
        query,
        namespace=None,
        model=None,
        system_prompt=None,
        temperature=None,
        track_miss=True,
        *,
        context=None,
    ):
        return self.lookup(
            query,
            namespace,
            context=context,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
        ).response

    async def aget(
        self,
        query,
        namespace=None,
        model=None,
        system_prompt=None,
        temperature=None,
        track_miss=True,
        *,
        context=None,
    ):
        return (
            await self.alookup(
                query,
                namespace,
                context=context,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        ).response

    def set(
        self,
        query,
        response,
        namespace=None,
        ttl=None,
        tags=None,
        model=None,
        system_prompt=None,
        temperature=None,
        *,
        context=None,
        _cancel=None,
    ):
        if self._metadata_only:
            raise ValueError("Vector operations require a normal cache instance")
        prepared, reason = self._prepare(
            query,
            context,
            namespace,
            dict(model=model, system_prompt=system_prompt, temperature=temperature),
        )
        if reason:
            return StoreDecision("REJECTED", reason)
        ctx, partition, identity = prepared
        payload, decision = admit(response, ctx, self._admit)
        if decision.action != "accept":
            return StoreDecision("REJECTED", decision.reason)
        effective_ttl = self.config.ttl_seconds if ttl is None else ttl
        if type(effective_ttl) is not int or effective_ttl < 0:
            raise ValueError("TTL must be a nonnegative integer")
        if tags is not None and (
            not isinstance(tags, (list, tuple)) or not all(type(t) is str for t in tags)
        ):
            raise ValueError("Tags must be strings")
        vector = self._embedding.embed(query)
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
            if _cancel is not None and _cancel.is_set():
                return StoreDecision("REJECTED", "cancelled")
            identifier = self.backend.store(entry)
            try:
                self.index.add(vector, identifier)
            except Exception:
                self.backend.delete(identifier)
                raise
            excess = self.backend.count(partition) - self.config.max_entries
            if excess > 0:
                self.backend.evict_oldest(partition, excess)
                self._rebuild_index_from_backend()
        return StoreDecision("STORED", "accepted", identifier)

    async def aset(self, query, response, namespace=None, **kwargs):
        return await self._async(self.set, query, response, namespace, **kwargs)

    def cached(
        self,
        func=None,
        *,
        namespace=None,
        ttl=None,
        tags=None,
        context=None,
        context_resolver=None,
        **legacy,
    ):
        if context_resolver is not None and (
            not callable(context_resolver)
            or inspect.iscoroutinefunction(context_resolver)
            or inspect.iscoroutinefunction(getattr(context_resolver, "__call__", None))
        ):
            raise ValueError("Context resolvers must be synchronous callables")

        def decorate(fn):
            def context_for(base):
                if not isinstance(base, RequestContext):
                    if inspect.iscoroutine(base):
                        base.close()
                    return None
                return replace(
                    base, operation=base.operation + ":" + fn.__module__ + "." + fn.__qualname__
                )

            def resolve(args, kwargs):
                if context_resolver is None:
                    return context_for(context or self.context)
                try:
                    return context_for(context_resolver(*args, **kwargs))
                except Exception:
                    return None

            if inspect.iscoroutinefunction(fn):

                @wraps(fn)
                async def async_wrapper(query, *args, **kwargs):
                    if args or kwargs or legacy:
                        return await fn(query, *args, **kwargs)
                    ctx = resolve((query,), {})
                    if ctx is None:
                        return await fn(query)
                    decision = await self.alookup(query, namespace, context=ctx)
                    if decision.outcome == "HIT":
                        return decision.response
                    result = await fn(query)
                    await self.aset(query, result, namespace, context=ctx, ttl=ttl, tags=tags)
                    return result

                async_wrapper.cache_context_for = context_for
                return async_wrapper

            @wraps(fn)
            def wrapper(query, *args, **kwargs):
                if args or kwargs or legacy:
                    return fn(query, *args, **kwargs)
                ctx = resolve((query,), {})
                if ctx is None:
                    return fn(query)
                decision = self.lookup(query, namespace, context=ctx)
                if decision.outcome == "HIT":
                    return decision.response
                result = fn(query)
                self.set(query, result, namespace, context=ctx, ttl=ttl, tags=tags)
                return result

            wrapper.cache_context_for = context_for
            return wrapper

        return decorate(func) if func is not None else decorate

    def cached_stream(self, func=None, **kwargs):
        # Preserve the generator interface while unsupported replay remains pass-through.
        return func if func is not None else lambda fn: fn

    def inspect(self, query, namespace=None, *, context=None, **legacy):
        return self.lookup(query, namespace, context=context, _inspect=True, **legacy).inspection()

    def invalidate(self, namespace=None, *, context=None, **legacy):
        prepared, reason = self._prepare("maintenance", context, namespace, legacy)
        if reason:
            return 0
        with self._mutation_lock:
            count = self.backend.delete_namespace(prepared[1])
            self._rebuild_index_from_backend()
            return count

    def invalidate_by_tag(self, tag, namespace=None, *, context=None, **legacy):
        prepared, reason = self._prepare("maintenance", context, namespace, legacy)
        if reason:
            return 0
        with self._mutation_lock:
            count = self.backend.delete_by_tag(tag, prepared[1])
            self._rebuild_index_from_backend()
            return count

    def clear(self, *, operator=False):
        self._ensure_open()
        if not operator:
            raise ValueError("Clearing all partitions requires explicit operator=True")
        with self._mutation_lock:
            count = self.backend.clear()
            self.index.rebuild([])
            return count

    def entries(self, namespace=None, *, context=None, operator=False):
        self._ensure_open()
        with self._mutation_lock:
            if operator:
                return self.backend.list_all_entries()
            prepared, reason = self._prepare("maintenance", context, namespace, {})
            return [] if reason else self.backend.list_entries(prepared[1])

    def stats(self, *, context=None, namespace=None, operator=False):
        self._ensure_open()
        with self._mutation_lock:
            if operator:
                counts = self._stats.copy()
                total = len(self.backend.list_all_entries())
            else:
                prepared, reason = self._prepare("maintenance", context, namespace, {})
                counts = (
                    Counter()
                    if reason
                    else self._partition_stats.get(prepared[1], Counter()).copy()
                )
                total = 0 if reason else self.backend.count(prepared[1])
            hits, misses = counts["hits"], counts["misses"]
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
                "total_entries": total,
                "total_saved_ms": 0.0,
                "estimated_tokens_saved": 0,
                "estimated_cost_saved_usd": 0.0,
            }

    def _sweep(self):
        while not self._stop_event.wait(60):
            with self._mutation_lock:
                if not self._closed and self.backend.sweep_expired() and self._ready:
                    self._rebuild_index_from_backend()

    def close(self):
        with self._mutation_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
        if self._sweep_thread is not None:
            self._sweep_thread.join(timeout=2)
        if hasattr(self.backend, "close"):
            self.backend.close()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __repr__(self):
        return f"LongCache(backend={self.config.backend!r}, format=2)"
