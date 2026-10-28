"""Explicit single-process Redis demo. Point REDIS_URL at an owned test service."""

import os
from longcache import CacheConfig, LongCache, RequestContext, RequestScope


def main():
    context = RequestContext(
        scope=RequestScope("demo-tenant", "demo-user"),
        operation="demo.answer",
        provider="fixture",
        provider_scope="demo-v1",
        model="fixture",
        prompt_version="v1",
    )
    config = CacheConfig(
        backend="redis",
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:16379/0"),
        cache_id="example",
        generation="v2",
        ttl_seconds=600,
    )
    with LongCache(config, context=context) as cache:
        cache.set("return policy", "An approved answer")
        decision = cache.lookup("return policy")
        print(decision.outcome, decision.reason)
        print(cache.stats())
    # Redis persists responses. It does not synchronize indexes between processes.


if __name__ == "__main__":
    main()
