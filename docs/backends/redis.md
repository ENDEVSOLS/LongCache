# Redis

Install `longcache[redis]` and set `CacheConfig(backend="redis", redis_url=..., cache_id=..., generation=...)`.

Every owned key uses `longcache:v2:<cache_id>:<generation>`, including entries, ID counters, namespace/tag sets, and manifest. A generation binds one embedding fingerprint. Cleanup operates only on that owned keyspace; it does not flush a Redis database.

A populated generation whose manifest is missing fails closed. Do not recreate its manifest manually or reuse its identity for a different model; select a fresh generation and preserve the old keys for diagnosis.

Redis persists responses and TTLs. The FAISS index stays local: use **one LongCache process per cache generation**. Shared response storage alone is not multi-instance consistency.

`backend="async_redis"` fails immediately through `LongCache`. `AsyncRedisCacheBackend` remains a standalone experimental storage adapter with separate tests.
