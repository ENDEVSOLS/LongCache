# Upgrading to LongCache v1.0.0 (Stable Foundation)

Version `1.0.0` is the **first frozen stable public API and documented support contract** for LongCache. It graduates the package from Beta (`Development Status :: 4 - Beta`) to Production/Stable (`Development Status :: 5 - Production/Stable`).

---

## 1. Zero Breaking Changes & Seamless Cache Persistence

Upgrading from `v0.6.0` (or `v0.5.0`) to `v1.0.0` requires **zero code rewrites** and **zero cache resets**:

* **Core APIs Unchanged:** All signatures for `LongCache`, `CacheConfig`, `RequestContext`, `RequestScope`, decorators, and backends remain identical to `v0.6.0`.
* **Format-2 Cache Persistence:** Existing SQLite databases (`.longcache/v2/cache.db`), single-process Redis stores, and shared Redis clusters (`deployment_mode="shared"`) upgrade cleanly with **zero data loss, zero re-indexing, and identical `v2:<sha256>` partition digests**.

---

## 2. API Stability Graduation (`stable-candidate` → `stable`)

In `v0.5.0` and `v0.6.0`, public symbols in `longcache.PUBLIC_API_STABILITY` were marked `"stable-candidate"`. In `v1.0.0`:

* All candidate interfaces have officially graduated to `"stable"`.
* In addition, `SharedRedisCacheBackend` and `SharedCoordinationBackend` (which passed full multi-process fault and memory bounds qualification in `0.6.0`) have graduated from `"experimental"` to `"stable"`.
* Standalone `AsyncRedisCacheBackend` remains `"experimental"`.

If your codebase or test suite inspects `longcache.PUBLIC_API_STABILITY`, update checks to expect `"stable"`:

```python
from longcache import PUBLIC_API_STABILITY

# Recommended check across 0.6 and 1.0:
is_stable = PUBLIC_API_STABILITY["LongCache"] in ("stable", "stable-candidate")
```

---

## 3. Explicit Pass-Through Warnings (`LongCachePassThroughWarning`)

LongCache includes compatibility adapters (`CachedLLM`, `CachedChain`, `CachedQueryEngine`, `CachedState`, `StreamingCacheAdapter`, and `cache.cached_stream`) for third-party frameworks whose opaque graph state cannot guarantee execution identity.

To prevent operators from mistakenly believing opaque framework calls are cached in production, **LongCache 1.0 emits a filterable `LongCachePassThroughWarning` on instantiation or use**.

### Example Warning
```text
LongCachePassThroughWarning: CachedLLM is a compatibility pass-through adapter in LongCache 1.0
(reason='unsupported_request_shape') and does not cache or reuse responses. Use
LongCache.get_or_compute() or @cache.cached with an explicit RequestContext for verified
semantic caching, or pass warn_passthrough=False to silence this warning.
```

### How to Address This Warning

#### Option A: Use Verified Semantic Caching (Recommended)
Migrate to `cache.get_or_compute()` or `@cache.cached` with an explicit `RequestContext`:

```python
from longcache import LongCache, RequestContext, RequestScope

context = RequestContext(
    scope=RequestScope(tenant_id="tenant-1", user_id="user-1"),
    operation="llm.chat",
    provider="openai",
    provider_scope="default",
    model="gpt-4o",
    prompt_version="v1",
)

# Active semantic cache lookup and singleflight origin execution
answer = cache.get_or_compute(
    query="Explain quantum computing simply",
    compute=lambda: llm.invoke("Explain quantum computing simply"),
    context=context,
)
```

#### Option B: Silence the Warning for Intentional Pass-Through Shims
If you are using these wrappers purely for structural framework compatibility:

```python
# Pass warn_passthrough=False directly
cached_llm = CachedLLM(llm, cache=cache, warn_passthrough=False)

# Or filter the warning globally in your application startup:
import warnings
from longcache import LongCachePassThroughWarning

warnings.filterwarnings("ignore", category=LongCachePassThroughWarning)
```

---

## 4. Verification & Support Matrix

* Review the [Support Matrix](support-matrix.md) for full platform, runtime, and stability tiers.
* Review the [1.x SemVer Contract](semver-contract.md) for long-term API stability and support guarantees.

