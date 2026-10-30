"""Public reliability errors. Messages never contain request or dependency payloads."""


class CacheError(RuntimeError):
    """Base class for cache reliability failures."""


class CacheClosedError(CacheError):
    """The cache no longer accepts this operation."""


class CacheShutdownTimeoutError(CacheError, TimeoutError):
    """Cleanup is still pending; resources in use have not been closed."""


class CacheCapacityError(CacheError):
    """The bounded cache worker queue is full."""


class CacheOperationError(CacheError):
    """A recognized runtime cache dependency failed."""

    def __init__(self, stage):
        self.stage = stage
        super().__init__(f"Cache operation failed at stage: {stage}")


class SingleflightError(CacheError):
    """Base class for request coordination failures."""


class SingleflightTimeoutError(SingleflightError, TimeoutError):
    """A follower stopped waiting; shared execution continues."""


class SingleflightCapacityError(SingleflightError):
    """The active flight or follower limit was reached."""


class SingleflightLeaderError(SingleflightError):
    """The leader failed or was cancelled; this caller did not run the origin."""


class SingleflightReentrancyError(SingleflightError):
    """An execution tried to wait on its own key."""


class SingleflightInvalidatedError(SingleflightError):
    """Explicit invalidation fenced the shared result."""
