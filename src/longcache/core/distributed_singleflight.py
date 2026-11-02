"""Exact-request coordination across processes sharing one Redis authority."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

from .errors import (
    CacheOperationError,
    SingleflightInvalidatedError,
    SingleflightLeaderError,
    SingleflightTimeoutError,
)
from .identity import digest


class _LeaseRenewal:
    def __init__(self, backend, key: str, token: str, ttl_ms: int, interval: float):
        self.backend, self.key, self.token = backend, key, token
        self.ttl_ms, self.interval = ttl_ms, interval
        self.stop_event = threading.Event()
        self.lost = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="longcache-lease")

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                if not self.backend.renew_flight(self.key, self.token, self.ttl_ms):
                    self.lost.set()
                    return
            except Exception:
                self.lost.set()
                return

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)


class DistributedExecution:
    """Shares only stored, policy-verified results; leaders own their origin calls."""

    def __init__(self, cache):
        self.cache = cache

    @staticmethod
    def _invoke(function, *args, _cancel=None):
        return function(*args)

    async def _call(self, function, *args):
        return await self.cache._lifecycle.call(self._invoke, function, *args)

    def _prepare(self, query, context, namespace, ttl, tags):
        cache = self.cache
        prepared, reason = cache._prepare(query, context, namespace, {})
        if prepared is None:
            return None, reason, None, None
        ctx = prepared[0]
        shared_epoch = cache._dependency("backend", cache.backend.epoch)
        exact = ctx.exact_key(
            query,
            namespace if namespace is not None else cache.config.namespace,
            cache.config.scope_policy,
            cache._fingerprint,
            cache.config.policy_version,
        )
        policy = digest({"ttl": cache.config.ttl_seconds if ttl is None else ttl, "tags": tags})
        return prepared, None, shared_epoch, digest({"exact": exact, "policy": policy})

    def _check_epoch(self, expected):
        actual = self.cache._dependency("backend", self.cache.backend.epoch)
        if actual != expected:
            raise SingleflightInvalidatedError("Shared execution was invalidated")

    def _finish(self, prepared, decision, response, source, mode, role, entry_point, reason=None):
        return self.cache._execution.finish(
            prepared,
            decision,
            response,
            source,
            mode,
            role,
            entry_point,
            self.cache._epoch,
            reason=reason,
        )

    def _lookup(self, query, ctx, namespace):
        return self.cache.lookup(query, namespace, context=ctx, _record=False, _internal=True)

    async def _alookup(self, query, ctx, namespace):
        return await self.cache.alookup(
            query, namespace, context=ctx, _record=False, _internal=True
        )

    def run(self, query, compute, *, context, namespace, ttl, tags, entry_point):
        try:
            return self._run(
                query,
                compute,
                context=context,
                namespace=namespace,
                ttl=ttl,
                tags=tags,
                entry_point=entry_point,
            )
        except CacheOperationError:
            if self.cache.config.cache_error_policy != "bypass":
                raise
            with self.cache._lifecycle.operation():
                return self._finish(
                    None,
                    None,
                    self.cache._execution.origin(compute),
                    "origin",
                    "sync",
                    "none",
                    entry_point,
                    "cache_unavailable",
                )

    def _run(self, query, compute, *, context, namespace, ttl, tags, entry_point):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("Use aget_or_compute from an event loop")
        if not callable(compute) or inspect.iscoroutinefunction(compute):
            raise TypeError("get_or_compute requires a synchronous callable")
        cache = self.cache
        tags = cache._execution.options(ttl, tags)
        lease_ms = int(cache.config.distributed_lease_seconds * 1000)
        with cache._lifecycle.operation():
            prepared, reason, shared_epoch, flight_key = self._prepare(
                query, context, namespace, ttl, tags
            )
            if prepared is None:
                return self._finish(
                    None,
                    None,
                    cache._execution.origin(compute),
                    "origin",
                    "sync",
                    "none",
                    entry_point,
                    reason,
                )
            ctx = prepared[0]
            decision = self._lookup(query, ctx, namespace)
            if decision.outcome == "HIT":
                return self._finish(
                    prepared, decision, decision.response, "cache", "sync", "none", entry_point
                )
            if decision.outcome == "BYPASS":
                return self._finish(
                    None,
                    None,
                    cache._execution.origin(compute),
                    "origin",
                    "sync",
                    "none",
                    entry_point,
                    decision.reason,
                )
            deadline = time.monotonic() + cache.config.singleflight_wait_timeout_seconds
            while True:
                self._check_epoch(shared_epoch)
                key, token = cache._dependency(
                    "backend", cache.backend.acquire_flight, flight_key, lease_ms
                )
                if token is not None:
                    try:
                        self._check_epoch(shared_epoch)
                        decision = self._lookup(query, ctx, namespace)
                        if decision.outcome == "HIT":
                            return self._finish(
                                prepared,
                                decision,
                                decision.response,
                                "cache",
                                "sync",
                                "leader",
                                entry_point,
                            )
                        if decision.outcome == "BYPASS":
                            return self._finish(
                                None,
                                None,
                                cache._execution.origin(compute),
                                "origin",
                                "sync",
                                "leader",
                                entry_point,
                                decision.reason,
                            )
                        with _LeaseRenewal(
                            cache.backend,
                            key,
                            token,
                            lease_ms,
                            cache.config.distributed_renew_interval_seconds,
                        ) as renewal:
                            result = cache._execution.origin(compute)
                            if not renewal.lost.is_set():
                                stored = cache.set(
                                    query,
                                    result,
                                    namespace,
                                    ttl,
                                    tags,
                                    context=ctx,
                                    _expected_shared_epoch=shared_epoch,
                                    _lease_key=key,
                                    _lease_token=token,
                                    _internal=True,
                                )
                                if stored.outcome != "STORED":
                                    try:
                                        cache.backend.finish_flight(key, token, "non_shareable")
                                    except Exception:
                                        pass  # The lease expiry releases remote waiters.
                        return self._finish(
                            prepared, None, result, "origin", "sync", "leader", entry_point
                        )
                    except BaseException as exc:
                        try:
                            cache._dependency(
                                "backend",
                                cache.backend.finish_flight,
                                key,
                                token,
                                "cancelled" if isinstance(exc, KeyboardInterrupt) else "error",
                            )
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            cache._dependency("backend", cache.backend.release_flight, key, token)
                        except CacheOperationError:
                            pass  # The bounded lease expires; never repeat a completed origin.
                while time.monotonic() < deadline:
                    self._check_epoch(shared_epoch)
                    decision = self._lookup(query, ctx, namespace)
                    if decision.outcome == "HIT":
                        return self._finish(
                            prepared,
                            decision,
                            decision.response,
                            "cache",
                            "sync",
                            "follower",
                            entry_point,
                        )
                    if decision.outcome == "BYPASS":
                        return self._finish(
                            None,
                            None,
                            cache._execution.origin(compute),
                            "origin",
                            "sync",
                            "follower",
                            entry_point,
                            decision.reason,
                        )
                    if not cache._dependency("backend", cache.backend.flight_active, key):
                        outcome = cache._dependency("backend", cache.backend.flight_outcome, key)
                        if outcome in ("error", "cancelled"):
                            raise SingleflightLeaderError("Shared execution failed")
                        if outcome == "non_shareable":
                            return self._finish(
                                prepared,
                                None,
                                cache._execution.origin(compute),
                                "origin",
                                "sync",
                                "follower",
                                entry_point,
                            )
                        break
                    time.sleep(0.05)
                else:
                    raise SingleflightTimeoutError("Timed out waiting for shared execution")

    async def arun(self, query, compute, *, context, namespace, ttl, tags, entry_point):
        try:
            return await self._arun(
                query,
                compute,
                context=context,
                namespace=namespace,
                ttl=ttl,
                tags=tags,
                entry_point=entry_point,
            )
        except CacheOperationError:
            if self.cache.config.cache_error_policy != "bypass":
                raise
            with self.cache._lifecycle.operation():
                result = await self.cache._execution.aorigin(compute)
                return await self._call(
                    self._finish,
                    None,
                    None,
                    result,
                    "origin",
                    "async",
                    "none",
                    entry_point,
                    "cache_unavailable",
                )

    async def _arun(self, query, compute, *, context, namespace, ttl, tags, entry_point):
        if not callable(compute):
            raise TypeError("aget_or_compute requires an awaitable-producing callable")
        cache = self.cache
        tags = cache._execution.options(ttl, tags)
        lease_ms = int(cache.config.distributed_lease_seconds * 1000)
        with cache._lifecycle.operation():
            prepared, reason, shared_epoch, flight_key = await self._call(
                self._prepare, query, context, namespace, ttl, tags
            )
            if prepared is None:
                result = await cache._execution.aorigin(compute)
                return await self._call(
                    self._finish, None, None, result, "origin", "async", "none", entry_point, reason
                )
            ctx = prepared[0]
            decision = await self._alookup(query, ctx, namespace)
            if decision.outcome == "HIT":
                return await self._call(
                    self._finish,
                    prepared,
                    decision,
                    decision.response,
                    "cache",
                    "async",
                    "none",
                    entry_point,
                )
            if decision.outcome == "BYPASS":
                result = await cache._execution.aorigin(compute)
                return await self._call(
                    self._finish,
                    None,
                    None,
                    result,
                    "origin",
                    "async",
                    "none",
                    entry_point,
                    decision.reason,
                )
            deadline = time.monotonic() + cache.config.singleflight_wait_timeout_seconds
            while True:
                await self._call(self._check_epoch, shared_epoch)
                key, token = await self._call(cache.backend.acquire_flight, flight_key, lease_ms)
                if token is not None:
                    try:
                        await self._call(self._check_epoch, shared_epoch)
                        decision = await self._alookup(query, ctx, namespace)
                        if decision.outcome == "HIT":
                            return await self._call(
                                self._finish,
                                prepared,
                                decision,
                                decision.response,
                                "cache",
                                "async",
                                "leader",
                                entry_point,
                            )
                        if decision.outcome == "BYPASS":
                            result = await cache._execution.aorigin(compute)
                            return await self._call(
                                self._finish,
                                None,
                                None,
                                result,
                                "origin",
                                "async",
                                "leader",
                                entry_point,
                                decision.reason,
                            )
                        with _LeaseRenewal(
                            cache.backend,
                            key,
                            token,
                            lease_ms,
                            cache.config.distributed_renew_interval_seconds,
                        ) as renewal:
                            result = await cache._execution.aorigin(compute)
                            if not renewal.lost.is_set():
                                stored = await cache.aset(
                                    query,
                                    result,
                                    namespace,
                                    ttl=ttl,
                                    tags=tags,
                                    context=ctx,
                                    _expected_shared_epoch=shared_epoch,
                                    _lease_key=key,
                                    _lease_token=token,
                                    _internal=True,
                                )
                                if stored.outcome != "STORED":
                                    try:
                                        await self._call(
                                            cache.backend.finish_flight, key, token, "non_shareable"
                                        )
                                    except Exception:
                                        pass  # Do not repeat a completed origin.
                        return await self._call(
                            self._finish,
                            prepared,
                            None,
                            result,
                            "origin",
                            "async",
                            "leader",
                            entry_point,
                        )
                    except BaseException as exc:
                        try:
                            await self._call(
                                cache.backend.finish_flight,
                                key,
                                token,
                                "cancelled" if isinstance(exc, asyncio.CancelledError) else "error",
                            )
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            await self._call(cache.backend.release_flight, key, token)
                        except CacheOperationError:
                            pass  # A lost release is bounded by the Redis TTL.
                while time.monotonic() < deadline:
                    await self._call(self._check_epoch, shared_epoch)
                    decision = await self._alookup(query, ctx, namespace)
                    if decision.outcome == "HIT":
                        return await self._call(
                            self._finish,
                            prepared,
                            decision,
                            decision.response,
                            "cache",
                            "async",
                            "follower",
                            entry_point,
                        )
                    if decision.outcome == "BYPASS":
                        result = await cache._execution.aorigin(compute)
                        return await self._call(
                            self._finish,
                            None,
                            None,
                            result,
                            "origin",
                            "async",
                            "follower",
                            entry_point,
                            decision.reason,
                        )
                    active = await self._call(cache.backend.flight_active, key)
                    if not active:
                        outcome = await self._call(cache.backend.flight_outcome, key)
                        if outcome in ("error", "cancelled"):
                            raise SingleflightLeaderError("Shared execution failed")
                        if outcome == "non_shareable":
                            result = await cache._execution.aorigin(compute)
                            return await self._call(
                                self._finish,
                                prepared,
                                None,
                                result,
                                "origin",
                                "async",
                                "follower",
                                entry_point,
                            )
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise SingleflightTimeoutError("Timed out waiting for shared execution")
