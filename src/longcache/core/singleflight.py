"""Per-instance exact-request coordination; completion objects work across threads/loops."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, field
import threading
from typing import TYPE_CHECKING, Any

from .errors import CacheClosedError, SingleflightCapacityError, SingleflightInvalidatedError

if TYPE_CHECKING:
    from longcache.config import CacheConfig


@dataclass(frozen=True)
class Completion:
    kind: str
    payload: bytes | None = None
    error: BaseException | None = field(default=None, repr=False)
    expires_at: float | None = None
    entry_id: int | None = None


@dataclass(eq=False)
class Flight:
    key: tuple[Any, ...] = field(repr=False)
    future: Future[Completion] = field(default_factory=Future, repr=False)
    followers: int = 0
    retained_bytes: int = 0
    async_waiters: dict[asyncio.Future[Completion], asyncio.AbstractEventLoop] = field(
        default_factory=dict, repr=False
    )


class Singleflight:
    def __init__(self, config: CacheConfig) -> None:
        self._lock = threading.Lock()
        self._flights: dict[tuple[Any, ...], Flight] = {}
        self._followers = 0
        self._retained = 0
        self.max_inflight = config.singleflight_max_inflight
        self.max_followers = config.singleflight_max_waiters_per_key
        self.max_payload = config.max_capture_bytes
        self.max_retained = 32 * 1024 * 1024
        self.epoch = 0

    def claim(self, key: tuple[Any, ...]) -> tuple[Flight, bool]:
        with self._lock:
            if key[2] != self.epoch:
                raise SingleflightInvalidatedError(
                    "Flight identity was invalidated before registration"
                )
            flight = self._flights.get(key)
            if flight is not None:
                if flight.followers >= self.max_followers:
                    raise SingleflightCapacityError("Singleflight follower capacity reached")
                flight.followers += 1
                self._followers += 1
                return flight, False
            if len(self._flights) >= self.max_inflight:
                raise SingleflightCapacityError("Singleflight key capacity reached")
            flight = Flight(key)
            self._flights[key] = flight
            return flight, True

    def complete(self, flight: Flight, completion: Completion) -> None:
        with self._lock:
            if flight.future.done():
                return
            size = len(completion.payload or b"")
            if size > self.max_payload or size + self._retained > self.max_retained:
                completion, size = Completion("non_shareable"), 0
            if flight.followers:
                flight.retained_bytes = size
                self._retained += size
            if self._flights.get(flight.key) is flight:
                del self._flights[flight.key]
            flight.future.set_result(completion)
            waiters = list(flight.async_waiters.items())
            flight.async_waiters.clear()
        self._notify(waiters, completion)

    @staticmethod
    def _notify(
        waiters: list[tuple[asyncio.Future[Completion], asyncio.AbstractEventLoop]],
        completion: Completion,
    ) -> None:
        def deliver(waiter: asyncio.Future[Completion]) -> None:
            if not waiter.done():
                waiter.set_result(completion)

        for waiter, loop in waiters:
            try:
                loop.call_soon_threadsafe(deliver, waiter)
            except RuntimeError:
                pass  # The caller's loop has already closed.

    async def wait(self, flight: Flight, timeout: float | None) -> Completion:
        # wrap_future(shared_future) cannot remove its concurrent-future callback
        # on timeout. Explicit detachable subscriptions keep churn bounded even
        # when a leader runs indefinitely. Each local future belongs to one caller.
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[Completion] = loop.create_future()
        with self._lock:
            if flight.future.done():
                waiter.set_result(flight.future.result())
            else:
                flight.async_waiters[waiter] = loop
        try:
            return await asyncio.wait_for(waiter, timeout)
        finally:
            with self._lock:
                flight.async_waiters.pop(waiter, None)
            if not waiter.done():
                waiter.cancel()

    def detach(self, flight: Flight) -> None:
        with self._lock:
            flight.followers -= 1
            self._followers -= 1
            if not flight.followers:
                self._retained -= flight.retained_bytes
                flight.retained_bytes = 0

    def abort(self, shutdown: bool = False) -> None:
        error = (
            CacheClosedError("Cache shutdown interrupted shared execution")
            if shutdown
            else SingleflightInvalidatedError("Shared execution was invalidated")
        )
        with self._lock:
            if not shutdown:
                self.epoch += 1
            flights = list(self._flights.values())
            self._flights.clear()
            notifications: list[
                tuple[
                    list[tuple[asyncio.Future[Completion], asyncio.AbstractEventLoop]], Completion
                ]
            ] = []
            for flight in flights:
                if not flight.future.done():
                    completion = Completion("invalidated", error=error)
                    flight.future.set_result(completion)
                    notifications.append((list(flight.async_waiters.items()), completion))
                    flight.async_waiters.clear()
        for waiters, completion in notifications:
            self._notify(waiters, completion)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "active_flights": len(self._flights),
                "followers": self._followers,
                "retained_replay_bytes": self._retained,
            }
