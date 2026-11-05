# LangChain compatibility

`CachedLLM` and `CachedChain` preserve invocation/proxy interfaces (`pass-through` tier in `PUBLIC_API_STABILITY`) and **bypass cache reads and writes in v0.5** (`cache_bypass_reason = "unsupported_request_shape"`). Generic wrappers cannot prove that hidden model settings, bound arguments, memory, and replay types are represented safely.

For a proven single-turn text workflow, use a query-only `@cache.cached(context=...)` function with explicit operation/provider/model/instruction/source versions and supported JSON responses. Extra arguments and all stream helpers bypass.
