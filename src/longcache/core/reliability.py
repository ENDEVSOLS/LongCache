"""Failure translation is restricted to dependency calls, never origin execution."""

from functools import wraps
import sqlite3
import time
import asyncio
import inspect

from .decision import LookupDecision, StoreDecision
from .errors import CacheError, CacheOperationError
from .errors import (
    SingleflightTimeoutError,
    SingleflightCapacityError,
    SingleflightInvalidatedError,
)


def operational(exception, stage):
    if isinstance(exception, CacheError):
        return False
    if isinstance(exception, (OSError, TimeoutError, sqlite3.OperationalError)):
        return True
    if stage in ("embedding", "index") and isinstance(exception, RuntimeError):
        return True
    try:
        from redis.exceptions import ConnectionError, TimeoutError as RedisTimeout, BusyLoadingError

        return isinstance(exception, (ConnectionError, RedisTimeout, BusyLoadingError))
    except ImportError:
        return False


def dependency(cache, stage, function, *args, **kwargs):
    started = time.monotonic()
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        if not operational(exc, stage):
            raise
        cache._observability.count("errors", stage=stage, reason="cache_unavailable")
        raise CacheOperationError(stage) from exc
    finally:
        cache._observability.duration(stage, time.monotonic() - started)


def decision_operation(operation):
    """Apply runtime cache policy and record one external decision."""

    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            started = time.monotonic()
            internal = kwargs.pop("_internal", False)
            try:
                try:
                    if not self._coherent:
                        raise CacheOperationError("coherence_uncertain")
                    result = function(self, *args, **kwargs)
                except CacheOperationError as exc:
                    self._observability.event(
                        warning=True, stage=exc.stage, reason="cache_unavailable"
                    )
                    if self.config.cache_error_policy != "bypass":
                        raise
                    result = (
                        LookupDecision("BYPASS", "cache_unavailable")
                        if operation == "lookup"
                        else StoreDecision("REJECTED", "cache_unavailable")
                    )
            except BaseException:
                if not internal:
                    self._observability.count("requests", operation=operation, outcome="error")
                raise
            else:
                if not internal:
                    self._observability.count(
                        "requests", operation=operation, outcome=result.outcome.lower()
                    )
                    self._observability.count(
                        "decisions",
                        operation=operation,
                        outcome=result.outcome.lower(),
                        reason=result.reason,
                    )
                    self._observability.event(
                        operation=operation, outcome=result.outcome.lower(), reason=result.reason
                    )
                return result
            finally:
                self._observability.duration(operation, time.monotonic() - started)

        return wrapped

    return decorate


def execution_operation(function):
    """Record terminal failures once; successful delivery is recorded centrally."""

    def failed(cache, error, mode):
        if isinstance(error, asyncio.CancelledError):
            outcome, reason = "cancelled", "cancelled"
        elif isinstance(error, SingleflightTimeoutError):
            outcome, reason = "timeout", "timeout"
        elif isinstance(error, SingleflightCapacityError):
            outcome, reason = "error", "capacity"
        elif isinstance(error, SingleflightInvalidatedError):
            outcome, reason = "error", "invalidated"
        else:
            outcome, reason = "error", "leader_failed"
        cache._observability.count("requests", operation="execution", outcome=outcome, mode=mode)
        cache._observability.count("singleflight", outcome=outcome, reason=reason)
        cache._observability.event(operation="execution", outcome=outcome, reason=reason, mode=mode)

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def async_wrapped(self, *args, **kwargs):
            started = time.monotonic()
            try:
                return await function(self, *args, **kwargs)
            except BaseException as exc:
                failed(self, exc, "async")
                raise
            finally:
                self._observability.duration("execution", time.monotonic() - started, "async")

        return async_wrapped

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return function(self, *args, **kwargs)
        except BaseException as exc:
            failed(self, exc, "sync")
            raise
        finally:
            self._observability.duration("execution", time.monotonic() - started)

    return wrapped
