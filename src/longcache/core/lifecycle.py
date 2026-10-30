"""Owned operations and bounded worker execution, shared by sync and async callers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import asyncio
import threading
import time

from .errors import CacheCapacityError, CacheClosedError, CacheShutdownTimeoutError


def managed(function):
    """Keep resources alive until this public synchronous operation exits."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with self._lifecycle.operation():
            return function(self, *args, **kwargs)

    return wrapped


class Operation:
    def __init__(self):
        self.cancelled = threading.Event()
        self.shutdown_cancelled = False
        try:
            self.task = asyncio.current_task()
        except RuntimeError:
            self.task = None

    def is_set(self):
        return self.cancelled.is_set()

    def set(self):
        self.cancelled.set()


class Lifecycle:
    def __init__(self, finalize, revoke, timeout):
        self._condition = threading.Condition()
        self._current = ContextVar(f"longcache_operation_{id(self)}", default=None)
        self._active = {}
        self.state = "OPEN"
        self._finalizing = False
        self._failure = None
        self._finalize = finalize
        self._revoke = revoke
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="longcache-worker")
        self._capacity = threading.BoundedSemaphore(64)
        self._jobs = 0
        self._close_thread = None
        self._close_error = None

    @property
    def current(self):
        binding = self._current.get()
        # create_task copies contextvars. A child request must not inherit its
        # parent's ownership/cancellation token merely because it inherited context.
        if binding is not None and binding[1] == self._actor():
            return binding[0]
        return None

    @staticmethod
    def _actor():
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        return threading.get_ident(), task

    def ensure_open(self):
        with self._condition:
            token = self.current
            if self.state == "OPEN":
                return
            if self.state == "CLOSING" and token in self._active and not token.is_set():
                return
            raise CacheClosedError("Cache is closed or closing")

    def _reserve(self):
        with self._condition:
            self.ensure_open()
            token = self.current
            if token not in self._active:
                token = Operation()
            self._active[token] = self._active.get(token, 0) + 1
            return token

    def _release(self, token):
        with self._condition:
            self._active[token] -= 1
            if not self._active[token]:
                del self._active[token]
            self._condition.notify_all()
        self._finish_if_idle()

    @contextmanager
    def operation(self):
        token = self._reserve()
        binding = self._current.set((token, self._actor()))
        try:
            yield token
        finally:
            self._current.reset(binding)
            self._release(token)

    def submit(self, function, *args, **kwargs):
        if not self._capacity.acquire(blocking=False):
            raise CacheCapacityError("Cache worker capacity reached")
        try:
            token = self._reserve()
        except BaseException:
            self._capacity.release()
            raise
        context = copy_context()

        def run():
            binding = self._current.set((token, self._actor()))
            try:
                self.ensure_open()
                if token.is_set():
                    raise CacheClosedError("Queued cache operation was cancelled")
                return function(*args, _cancel=token, **kwargs)
            finally:
                self._current.reset(binding)

        with self._condition:
            self._jobs += 1

        def done(_):
            with self._condition:
                self._jobs -= 1
            self._capacity.release()
            self._release(token)

        try:
            future = self._executor.submit(context.run, run)
        except BaseException:
            done(None)
            raise
        future.add_done_callback(done)
        return future, token

    async def call(self, function, *args, **kwargs):
        future, token = self.submit(function, *args, **kwargs)
        bridge = asyncio.wrap_future(future)
        bridge.add_done_callback(lambda result: None if result.cancelled() else result.exception())
        try:
            return await asyncio.shield(bridge)
        except asyncio.CancelledError:
            token.set()
            # Keep capacity reserved until the worker dequeues this item. Cancelling
            # a queued Future would free the slot while its WorkItem still retains
            # arguments, allowing repeated cancellations to grow the queue forever.
            raise

    def _finish_if_idle(self):
        with self._condition:
            if self.state != "CLOSING" or self._active or self._finalizing:
                return
            self._finalizing = True
        try:
            self._finalize()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)
            with self._condition:
                if self._failure is None:
                    self.state = "CLOSED"
                self._condition.notify_all()

    def close(self, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
            raise ValueError("Shutdown timeout must be positive and finite")
        if self.current in self._active:
            raise CacheClosedError("Cannot close the cache from its own active operation")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self.state == "OPEN":
                self.state = "CLOSING"
        self._finish_if_idle()
        with self._condition:
            while self.state != "CLOSED":
                if self._failure is not None:
                    raise CacheClosedError(
                        "Cache resource cleanup failed; resources may remain open"
                    ) from self._failure
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending = list(self._active)
                    for token in pending:
                        token.set()
                    cancel_tasks = [
                        token.task
                        for token in pending
                        if token.task is not None and not token.shutdown_cancelled
                    ]
                    for token in pending:
                        token.shutdown_cancelled = True
                    break
                self._condition.wait(remaining)
            else:
                if self._failure is not None:
                    raise CacheClosedError("Cache resource cleanup failed") from self._failure
                return
        self._revoke()
        for task in cancel_tasks:
            if not task.done():
                loop = task.get_loop()
                if not loop.is_closed():
                    loop.call_soon_threadsafe(task.cancel)
        raise CacheShutdownTimeoutError("Cache cleanup is pending active operations")

    async def aclose(self, timeout=None):
        timeout = self.timeout if timeout is None else timeout
        if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
            raise ValueError("Shutdown timeout must be positive and finite")
        if self.current in self._active:
            raise CacheClosedError("Cannot close the cache from its own active operation")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self.state == "CLOSED":
                return
            if self._failure is not None:
                raise CacheClosedError("Cache resource cleanup failed") from self._failure
            if self._close_thread is None or not self._close_thread.is_alive():
                self.state = "CLOSING"
                self._close_error = None

                def drain():
                    try:
                        self.close(timeout)
                    except BaseException as exc:
                        with self._condition:
                            self._close_error = exc

                self._close_thread = threading.Thread(
                    target=drain, name="longcache-close", daemon=True
                )
                self._close_thread.start()
            worker = self._close_thread
        # One owned cleanup worker is shared across loops and callers. Polling its
        # completion retains no callbacks or default-executor jobs after cancellation.
        while worker.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CacheShutdownTimeoutError("Shared cache cleanup is still pending")
            await asyncio.sleep(min(0.001, remaining))
        with self._condition:
            error = self._close_error
        if isinstance(error, CacheShutdownTimeoutError):
            raise CacheShutdownTimeoutError(str(error)) from error
        if error is not None:
            raise CacheClosedError("Cache resource cleanup failed") from error

    def snapshot(self):
        with self._condition:
            return {
                "state": self.state,
                "active_operations": sum(self._active.values()),
                "cache_jobs": self._jobs,
                "cleanup_failed": self._failure is not None,
                "cleanup_workers": int(
                    self._close_thread is not None and self._close_thread.is_alive()
                ),
            }
