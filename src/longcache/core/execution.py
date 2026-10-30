"""One exact-scoped execution protocol for manual, decorator and provider adapters."""

import asyncio
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar
from dataclasses import dataclass, field
import inspect
import json
import time
import threading

from .decision import LookupDecision
from .errors import (
    CacheClosedError,
    CacheOperationError,
    SingleflightLeaderError,
    SingleflightReentrancyError,
    SingleflightTimeoutError,
    SingleflightInvalidatedError,
)
from .identity import digest, canonical
from .policy import admit, decode_response, verify
from .singleflight import Completion, Flight, Singleflight


@dataclass
class Start:
    prepared: tuple | None
    decision: LookupDecision | None
    flight: Flight | None
    leader: bool
    epoch: int
    reason: str | None = None
    detached: bool = False
    _detach_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def role(self):
        return "none" if self.flight is None else "leader" if self.leader else "follower"


class Execution:
    def __init__(self, cache):
        self.cache = cache
        self.registry = Singleflight(cache.config)
        self._owned = ContextVar(f"longcache_flight_owner_{id(self)}", default=())

    def shutdown(self):
        self.registry.abort(shutdown=True)

    def detach(self, state):
        with state._detach_lock:
            if state.flight is not None and not state.leader and not state.detached:
                state.detached = True
                self.registry.detach(state.flight)

    @staticmethod
    def options(ttl, tags):
        if ttl is not None and (type(ttl) is not int or ttl < 0):
            raise ValueError("TTL must be a nonnegative integer")
        if tags is not None and (
            not isinstance(tags, (tuple, list)) or not all(type(t) is str for t in tags)
        ):
            raise ValueError("Tags must be strings")
        return tuple(sorted(set(tags or ())))

    def lookup(self, query, context, namespace, *, _cancel=None):
        return self.cache.lookup(
            query, namespace, context=context, _record=False, _internal=True, _cancel=_cancel
        )

    async def afast_start(self, query, context, namespace, ttl, tags):
        """Join before lookup when identity is initialized; only the leader searches.

        Pure identity work and the short registry lock are safe on the loop. The
        worker lookup still validates the embedding generation. Followers observe
        that post-join lookup through completion and recheck the stored entry.
        """
        cache = self.cache
        if not cache._ready or not cache._coherent or not cache.config.singleflight_enabled:
            return await self.astart(query, context, namespace, ttl, tags)
        prepared, reason = cache._prepare(query, context, namespace, {}, _initialize=False)
        if prepared is None:
            return Start(None, None, None, False, cache._epoch, reason)
        ctx = prepared[0]
        exact = ctx.exact_key(
            query,
            namespace if namespace is not None else cache.config.namespace,
            cache.config.scope_policy,
            cache._fingerprint,
            cache.config.policy_version,
        )
        policy = digest({"ttl": cache.config.ttl_seconds if ttl is None else ttl, "tags": tags})
        epoch = cache._epoch
        key = exact, policy, epoch
        if key in self._owned.get():
            raise SingleflightReentrancyError("Execution cannot wait on its own key")
        flight, leader = self.registry.claim(key)
        state = Start(prepared, LookupDecision("MISS", "no_candidate"), flight, leader, epoch)
        cache._observability.count("singleflight", role=state.role)
        if leader:
            owner = self._owned.set((*self._owned.get(), key))
            try:
                state.decision = await cache._lifecycle.call(self.lookup, query, ctx, namespace)
            except BaseException as exc:
                self.registry.complete(flight, Completion("error", error=exc))
                raise
            finally:
                self._owned.reset(owner)
        return state

    def start(self, query, context, namespace, ttl, tags, *, _cancel=None):
        cache = self.cache
        try:
            prepared, reason = cache._prepare(query, context, namespace, {})
        except CacheOperationError:
            if cache.config.cache_error_policy != "bypass":
                raise
            prepared, reason = None, "cache_unavailable"
        if prepared is None:
            return Start(None, None, None, False, cache._epoch, reason)
        ctx = prepared[0]
        decision = self.lookup(query, ctx, namespace)
        state = Start(prepared, decision, None, True, cache._epoch)
        if decision.outcome == "HIT":
            return state
        if cache.config.singleflight_enabled:
            exact = ctx.exact_key(
                query,
                namespace if namespace is not None else cache.config.namespace,
                cache.config.scope_policy,
                cache._fingerprint,
                cache.config.policy_version,
            )
            policy = digest({"ttl": cache.config.ttl_seconds if ttl is None else ttl, "tags": tags})
            with cache._mutation_lock:
                state.epoch = cache._epoch
                key = exact, policy, state.epoch
                if key in self._owned.get():
                    raise SingleflightReentrancyError("Execution cannot wait on its own key")
                state.flight, state.leader = self.registry.claim(key)
            cache._observability.count("singleflight", role=state.role)
        # One worker performs preparation, election and post-join recheck. It does
        # not run origin code, and avoids several thread/loop round trips per caller.
        owner = (
            self._owned.set((*self._owned.get(), state.flight.key))
            if state.flight and state.leader
            else None
        )
        try:
            state.decision = self.lookup(query, ctx, namespace)
            return state
        except BaseException as exc:
            if state.flight is not None:
                if state.leader:
                    self.registry.complete(state.flight, Completion("error", error=exc))
                else:
                    self.detach(state)
            raise
        finally:
            if owner is not None:
                self._owned.reset(owner)

    async def astart(self, *args):
        future, token = self.cache._lifecycle.submit(self.start, *args)
        bridge = asyncio.wrap_future(future)
        bridge.add_done_callback(lambda result: None if result.cancelled() else result.exception())
        try:
            return await asyncio.shield(bridge)
        except asyncio.CancelledError:
            token.set()

            # Cancellation can arrive after election but before the await receives
            # its result. Release that ownership even when the caller never saw it.
            def abandon(result):
                if not result.cancelled() and result.exception() is None:
                    state = result.result()
                    if state.flight is not None:
                        if state.leader:
                            self.registry.complete(state.flight, Completion("error"))
                        else:
                            self.detach(state)

            future.add_done_callback(abandon)
            raise

    def finish(
        self,
        prepared,
        decision,
        response,
        source,
        mode,
        role,
        entry_point,
        epoch,
        *,
        reason=None,
        _cancel=None,
    ):
        cache = self.cache
        with cache._mutation_lock:
            cache._ensure_open()
            if _cancel is not None and _cancel.is_set():
                raise CacheClosedError("Operation was cancelled before delivery")
            response_epoch = decision._epoch if source == "cache" else epoch
            if source in ("cache", "coalesced") and response_epoch != cache._epoch:
                raise SingleflightInvalidatedError(
                    "Shared response was invalidated before delivery"
                )
            if (
                source in ("cache", "coalesced")
                and decision is not None
                and decision._expires_at is not None
                and time.time() >= decision._expires_at
            ):
                raise SingleflightInvalidatedError("Shared response expired before delivery")
            if prepared is not None:
                if source == "cache":
                    try:
                        cache._dependency(
                            "accounting", cache.backend.update_hit_count, decision.entry_id
                        )
                    except CacheOperationError:
                        pass
                cache._account(prepared[1], "hits" if source == "cache" else "misses")
        outcome = "hit" if source == "cache" else "miss" if prepared else "bypass"
        reason = reason or (
            "accepted_shared_result"
            if source == "coalesced"
            else decision.reason
            if decision
            else "no_candidate"
            if prepared
            else "missing_identity"
        )
        cache._observability.count("requests", operation="execution", outcome=outcome, mode=mode)
        if role == "follower":
            cache._observability.count(
                "shared_deliveries", role=role, source=source, outcome=outcome
            )
        cache._observability.count(
            "decisions", operation="execution", outcome=outcome, source=source, reason=reason
        )
        cache._observability.event(
            operation="execution",
            outcome=outcome,
            source=source,
            mode=mode,
            role=role,
            entry_point=entry_point,
            reason=reason,
        )
        return response

    async def afinish(self, *args, **kwargs):
        return await self.cache._lifecycle.call(self.finish, *args, **kwargs)

    def envelope(self, result, context, expires_at, entry_id):
        payload, admission = self.cache._dependency(
            "admission", admit, result, context, self.cache._admit
        )
        if admission.action != "accept":
            return Completion("non_shareable")
        encoded = canonical(payload).encode("utf-8")
        if len(encoded) > self.cache.config.max_capture_bytes:
            return Completion("non_shareable")
        return Completion("value", encoded, expires_at=expires_at, entry_id=entry_id)

    def replay(self, completion, query, context, prepared, *, _cancel=None):
        if completion.kind == "invalidated":
            raise type(completion.error)(str(completion.error))
        if completion.kind == "error":
            raise SingleflightLeaderError(
                "Shared execution failed or its leader was cancelled"
            ) from completion.error
        if completion.kind != "value" or (
            completion.expires_at is not None and time.time() >= completion.expires_at
        ):
            return None, False, None
        cache = self.cache
        response = decode_response(json.loads(completion.payload), context.codec)
        payload, admission = cache._dependency("admission", admit, response, context, cache._admit)
        if admission.action != "accept":
            return None, False, None
        verification = cache._dependency(
            "verification", verify, query, query, payload, 1.0, context, cache._verifier
        )
        cache._observability.count(
            "verification", outcome=verification.action, reason=verification.reason
        )
        if verification.action != "accept" or (
            completion.expires_at is not None and time.time() >= completion.expires_at
        ):
            return None, False, None
        decision = LookupDecision(
            "MISS", "accepted_shared_result", response, _expires_at=completion.expires_at
        )
        if completion.entry_id is not None:
            with cache._mutation_lock:
                if not cache._coherent and cache.config.cache_error_policy == "raise":
                    raise CacheOperationError("coherence_uncertain")
                entry = None
                if cache._coherent:
                    try:
                        entry = cache._dependency(
                            "backend", cache.backend.retrieve, completion.entry_id
                        )
                    except CacheOperationError:
                        if cache.config.cache_error_policy != "bypass":
                            raise
                if (
                    entry is not None
                    and not entry.is_expired()
                    and entry.namespace == prepared[1]
                    and isinstance(entry.metadata, dict)
                    and entry.metadata.get("format") == 2
                    and entry.metadata.get("identity") == prepared[2]
                    and entry.metadata.get("embedding") == cache._fingerprint
                ):
                    try:
                        valid = (
                            digest(entry.response)
                            == entry.metadata.get("payload")
                            == digest(payload)
                        )
                    except (ValueError, TypeError):
                        valid = False
                    if valid:
                        decision = LookupDecision(
                            "HIT",
                            "accepted",
                            response,
                            entry_id=entry.id,
                            _epoch=cache._epoch,
                            _expires_at=entry.expires_at,
                        )
        return response, True, decision

    def publish(self, flight, result, context, epoch, expires_at=None, entry_id=None):
        completion = self.envelope(result, context, expires_at, entry_id)
        with self.cache._mutation_lock:
            if epoch != self.cache._epoch:
                completion = Completion(
                    "invalidated",
                    error=SingleflightInvalidatedError("Shared execution was invalidated"),
                )
            self.registry.complete(flight, completion)

    def cached(self, state, mode, entry_point, *, _cancel=None):
        try:
            if state.flight is not None and state.leader:
                self.publish(
                    state.flight,
                    state.decision.response,
                    state.prepared[0],
                    state.epoch,
                    state.decision._expires_at,
                    state.decision.entry_id,
                )
            return self.finish(
                state.prepared,
                state.decision,
                state.decision.response,
                "cache",
                mode,
                state.role,
                entry_point,
                state.epoch,
                _cancel=_cancel,
            )
        except BaseException as exc:
            if state.flight is not None and state.leader:
                self.registry.complete(state.flight, Completion("error", error=exc))
            raise
        finally:
            if state.flight is not None and not state.leader:
                self.detach(state)

    def store(self, state, query, result, namespace, ttl, tags, *, _cancel=None):
        cache = self.cache
        effective_ttl = cache.config.ttl_seconds if ttl is None else ttl
        expiry = time.time() + effective_ttl if effective_ttl else None
        stored = cache.set(
            query,
            result,
            namespace,
            context=state.prepared[0],
            ttl=ttl,
            tags=tags,
            _expected_epoch=state.epoch,
            _internal=True,
            _cancel=_cancel,
        )
        if state.flight is not None and state.leader:
            if stored.outcome == "REJECTED" and stored.reason not in (
                "cache_unavailable",
                "invalidated",
            ):
                self.registry.complete(state.flight, Completion("non_shareable"))
            else:
                self.publish(
                    state.flight, result, state.prepared[0], state.epoch, expiry, stored.entry_id
                )
        return result

    def origin(self, compute):
        self.cache._observability.count("origin_executions")
        started = time.monotonic()
        try:
            result = compute()
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("Use aget_or_compute with an asynchronous callback")
            return result
        except BaseException:
            self.cache._observability.count("errors", stage="origin", reason="provider_failed")
            raise
        finally:
            self.cache._observability.duration("origin", time.monotonic() - started)

    async def aorigin(self, compute):
        self.cache._observability.count("origin_executions")
        started = time.monotonic()
        try:
            pending = compute()
            if not inspect.isawaitable(pending):
                raise TypeError("aget_or_compute requires an awaitable-producing callback")
            return await pending
        except BaseException:
            self.cache._observability.count("errors", stage="origin", reason="provider_failed")
            raise
        finally:
            self.cache._observability.duration("origin", time.monotonic() - started, "async")

    def follower_result(self, state, completion, query, mode, entry_point, *, _cancel=None):
        response, accepted, decision = self.replay(
            completion, query, state.prepared[0], state.prepared
        )
        if not accepted:
            return None, False
        source = "cache" if decision.outcome == "HIT" else "coalesced"
        return self.finish(
            state.prepared,
            decision,
            response,
            source,
            mode,
            "follower",
            entry_point,
            state.epoch,
            _cancel=_cancel,
        ), True

    def run(
        self,
        query,
        compute,
        *,
        context=None,
        namespace=None,
        ttl=None,
        tags=None,
        _entry_point="manual",
    ):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use aget_or_compute from an event loop")
        if (
            not callable(compute)
            or inspect.iscoroutinefunction(compute)
            or inspect.iscoroutinefunction(getattr(compute, "__call__", None))
        ):
            raise TypeError("get_or_compute requires a synchronous callable")
        tags = self.options(ttl, tags)
        cache = self.cache
        with cache._lifecycle.operation():
            state = self.start(query, context, namespace, ttl, tags)
            if state.prepared is None:
                return self.finish(
                    None,
                    None,
                    self.origin(compute),
                    "origin",
                    "sync",
                    "none",
                    _entry_point,
                    state.epoch,
                    reason=state.reason,
                )
            if state.decision.outcome == "HIT":
                return self.cached(state, "sync", _entry_point)
            if state.flight is not None and not state.leader:
                started = time.monotonic()
                try:
                    try:
                        completion = state.flight.future.result(
                            timeout=cache.config.singleflight_wait_timeout_seconds
                        )
                    except FutureTimeout:
                        raise SingleflightTimeoutError(
                            "Timed out waiting for shared execution"
                        ) from None
                    result, accepted = self.follower_result(
                        state, completion, query, "sync", _entry_point
                    )
                    if accepted:
                        return result
                finally:
                    cache._observability.duration("wait", time.monotonic() - started)
                    self.detach(state)
            owner = (
                self._owned.set((*self._owned.get(), state.flight.key))
                if state.flight and state.leader
                else None
            )
            try:
                result = self.origin(compute)
                self.store(state, query, result, namespace, ttl, tags)
                return self.finish(
                    state.prepared,
                    None,
                    result,
                    "origin",
                    "sync",
                    state.role,
                    _entry_point,
                    state.epoch,
                )
            except BaseException as exc:
                if state.flight is not None and state.leader:
                    self.registry.complete(state.flight, Completion("error", error=exc))
                raise
            finally:
                if owner is not None:
                    self._owned.reset(owner)

    async def arun(
        self,
        query,
        compute,
        *,
        context=None,
        namespace=None,
        ttl=None,
        tags=None,
        _entry_point="manual",
    ):
        if not callable(compute):
            raise TypeError("aget_or_compute requires an awaitable-producing callable")
        tags = self.options(ttl, tags)
        cache = self.cache
        with cache._lifecycle.operation():
            state = await self.afast_start(query, context, namespace, ttl, tags)
            if state.prepared is None:
                result = await self.aorigin(compute)
                return await self.afinish(
                    None,
                    None,
                    result,
                    "origin",
                    "async",
                    "none",
                    _entry_point,
                    state.epoch,
                    reason=state.reason,
                )
            if state.decision.outcome == "HIT":
                # If cancellation loses this scheduled result, cached() still owns
                # and releases any elected flight in its worker finally block.
                try:
                    return await cache._lifecycle.call(self.cached, state, "async", _entry_point)
                except BaseException as exc:
                    if state.flight is not None and state.leader:
                        self.registry.complete(state.flight, Completion("error", error=exc))
                    self.detach(state)
                    raise
            if state.flight is not None and not state.leader:
                started = time.monotonic()
                try:
                    try:
                        completion = await self.registry.wait(
                            state.flight, cache.config.singleflight_wait_timeout_seconds
                        )
                    except asyncio.TimeoutError:
                        raise SingleflightTimeoutError(
                            "Timed out waiting for shared execution"
                        ) from None
                    result, accepted = await cache._lifecycle.call(
                        self.follower_result, state, completion, query, "async", _entry_point
                    )
                    if accepted:
                        return result
                finally:
                    cache._observability.duration("wait", time.monotonic() - started, "async")
                    self.detach(state)
            owner = (
                self._owned.set((*self._owned.get(), state.flight.key))
                if state.flight and state.leader
                else None
            )
            try:
                result = await self.aorigin(compute)
                await cache._lifecycle.call(self.store, state, query, result, namespace, ttl, tags)
                return await self.afinish(
                    state.prepared,
                    None,
                    result,
                    "origin",
                    "async",
                    state.role,
                    _entry_point,
                    state.epoch,
                )
            except BaseException as exc:
                if state.flight is not None and state.leader:
                    self.registry.complete(state.flight, Completion("error", error=exc))
                raise
            finally:
                if owner is not None:
                    self._owned.reset(owner)
