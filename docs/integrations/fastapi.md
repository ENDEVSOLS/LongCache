# ASGI / FastAPI

`LongCacheMiddleware` supports bounded non-streaming chat-completion JSON requests. Supply `context` with provider/deployment/operation/version configuration and a synchronous `scope_resolver(scope)` reading an authenticated principal from trusted ASGI state.

Authentication middleware must run before scope resolution. Raw tenant/user headers, Authorization values, cookies, and API keys are not identity sources. Authenticated requests bypass without a trusted resolver. Resolver failure bypasses without shared-tenant fallback.

The capture limit defaults to 1 MiB. Oversized, malformed/duplicate-key, unsupported-shape, and streaming requests pass through without cache storage. Complete status-200 responses require a single `application/json` media type; an optional charset is allowed, while `application/jsonp` is rejected. Cookie-bearing, `Vary`, private/no-store, incomplete, or oversized responses are not stored. Every repeated Cache-Control value is checked, so a later header cannot hide an earlier no-store directive. An observed request or response disconnect prevents storage, including when the application subsequently completes its response.

`X-LongCache`, `X-LongCache-Reason`, and an applicable same-scope similarity header explain lookup. Headers sent at response start do not claim later admission results. Upstream diagnostic copies are replaced; unsupported responses keep their body/status/framing.

Generic HTTP app invocations are **not coalesced** in v0.3. Supported cache errors
before response start follow the strict/bypass policy. After upstream has sent its
response, an incidental cache-write failure does not retract it or invoke the app
again. Wire `await middleware.aclose()` into application lifespan if the middleware
created its cache; a supplied cache remains caller-owned. See [reliability](../reliability.md).
