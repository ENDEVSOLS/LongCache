# Upgrading to LongCache v0.5.0

Version `0.5.0` stabilizes and strictly types public interfaces before the `1.0.0` API freeze. Persisted Format-2 cache databases (`.longcache/v2/cache.db` and Redis format-2 keys) remain **100% backward-compatible** and require no migration or cache reset.

---

## 1. RequestContext is the Single Identity Contract

In `0.1.0`, legacy per-call arguments (`model=...`, `system_prompt=...`, `temperature=...`) were accepted by `set()`, `get()`, and `CacheConfig`. In `0.2.0`–`0.4.0`, immutable `RequestContext` and `RequestScope` became the canonical identity standard.

In `0.5.0`, vestigial legacy keyword arguments have been retired from `CacheConfig` and public `LongCache` methods:

### Before (v0.1–v0.4 legacy style):
```python
cache = LongCache(CacheConfig(backend="memory", model="gpt-4o"))
cache.set("What is a stock?", "Answer", system_prompt="Finance expert")
answer = cache.get("What is a stock?", system_prompt="Finance expert")
```

### After (v0.5.0 standard):
```python
from dataclasses import replace
from longcache import CacheConfig, Instruction, LongCache, RequestContext, RequestScope

context = RequestContext(
    scope=RequestScope(tenant_id="tenant-1", user_id="user-1"),
    operation="finance.qa",
    provider="openai",
    provider_scope="default",
    model="gpt-4o",
    prompt_version="v1",
    instructions=(Instruction("system", "Finance expert"),),
)

cache = LongCache(CacheConfig(backend="sqlite"), context=context)
cache.set("What is a stock?", "Answer")
answer = cache.get("What is a stock?")

# For per-call context variation:
trader_ctx = replace(context, instructions=(Instruction("system", "Wall street trader"),))
cache.set("What is a stock?", "Trader answer", context=trader_ctx)
answer = cache.get("What is a stock?", context=trader_ctx)
```

---

## 2. Removal of Dead `track_miss` Parameter

`LongCache.get()` and `aget()` no longer accept the unused `track_miss` parameter. Hit and miss accounting is automatically handled based on `track_stats=True` in `CacheConfig`. To inspect whether a query matches without recording stats, use `cache.inspect(query, context=context)`.

---

## 3. Explicit `backend="async_redis"` Validation

`CacheConfig(backend="async_redis")` is now rejected directly in configuration validation with a clear message. Standalone async Redis users should instantiate `AsyncRedisCacheBackend` directly, or configure `backend="redis"` with `LongCache` (which supports full sync and async APIs).

---

## 4. PEP 561 Typing and `py.typed`

LongCache ships with `py.typed`. All public entry points (`LongCache`, `CacheConfig`, `RequestScope`, `RequestContext`, decisions, decorators, backends) now export complete static type signatures compatible with strict `mypy` and IDE autocompletion.
