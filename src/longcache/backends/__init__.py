"""Cache storage backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from longcache.backends.base import AbstractCacheBackend
from longcache.backends.memory import InMemoryCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend

if TYPE_CHECKING:
    from longcache.backends.redis import RedisCacheBackend

__all__ = ["AbstractCacheBackend", "SQLiteCacheBackend", "InMemoryCacheBackend", "get_redis_backend"]


# Lazy import for Redis backend — only available when redis extra is installed
def get_redis_backend() -> type[RedisCacheBackend]:
    """Import and return RedisCacheBackend. Raises ImportError if redis is not installed."""
    try:
        from longcache.backends.redis import RedisCacheBackend

        return RedisCacheBackend
    except ImportError:
        raise ImportError(
            "Redis backend requires the 'redis' extra. "
            "Install with: uv pip install longcache[redis]"
        )
