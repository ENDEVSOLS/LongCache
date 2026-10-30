"""Bounded ASGI adapter. Authentication and authorization remain application concerns."""

from __future__ import annotations
from dataclasses import replace
import inspect
import json
from longcache import CacheConfig, LongCache, RequestScope
from longcache.integrations.openai import prepare_request
from longcache import CacheError


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate request keys are not cacheable")
        result[key] = value
    return result


class LongCacheMiddleware:
    def __init__(
        self,
        app,
        cache=None,
        config=None,
        namespace="http",
        cached_routes=None,
        exclude_namespaces=None,
        *,
        context=None,
        scope_resolver=None,
    ):
        self.app = app
        self._owns_cache = cache is None
        self.cache = cache if cache is not None else LongCache(config or CacheConfig())
        self.namespace = namespace
        self.routes = set(cached_routes or {"/v1/chat/completions", "/chat/completions"})
        self.excluded = set(exclude_namespaces or ())
        self.context = context
        if scope_resolver is not None and (
            not callable(scope_resolver)
            or inspect.iscoroutinefunction(scope_resolver)
            or inspect.iscoroutinefunction(getattr(scope_resolver, "__call__", None))
        ):
            raise ValueError("Scope resolver must be a synchronous callable")
        self.scope_resolver = scope_resolver

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in self.routes
            or self.namespace in self.excluded
        ):
            await self.app(scope, receive, send)
            return
        limit = self.cache.config.max_capture_bytes
        frames = []
        body = bytearray()
        oversized = False
        while True:
            message = await receive()
            frames.append(message)
            if message["type"] == "http.disconnect":
                await self.app(scope, self._replay(frames, receive), send)
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                oversized = True
                break
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        original_replay = self._replay(frames, receive)
        disconnected = False

        async def replay():
            nonlocal disconnected
            message = await original_replay()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        data = None
        reason = "unsupported_request_shape"
        if not oversized:
            try:
                data = json.loads(
                    body,
                    object_pairs_hook=_object,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite number")),
                )
            except (ValueError, UnicodeDecodeError):
                pass
        base = self.context
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        authenticated = bool(set(headers) & {b"authorization", b"cookie", b"x-api-key"})
        if self.scope_resolver is not None and base is not None:
            try:
                resolved = self.scope_resolver(scope)
                if inspect.iscoroutine(resolved):
                    resolved.close()
                base = replace(base, scope=resolved) if isinstance(resolved, RequestScope) else None
            except Exception:
                base = None
        elif authenticated:
            base = None
        prepared = None
        if isinstance(data, dict):
            prepared, reason = prepare_request(data, base, "openai-dict-v1")
        if prepared is None:
            self.cache._observability.count("requests", operation="lookup", outcome="bypass")
            self.cache._observability.count(
                "decisions", operation="lookup", outcome="bypass", reason=reason
            )
            self.cache._observability.event(entry_point="asgi", outcome="bypass", reason=reason)
            await self.app(scope, replay, self._decision_send(send, "BYPASS", reason))
            return
        query, ctx = prepared
        decision = await self.cache.alookup(query, self.namespace, context=ctx)
        if decision.outcome == "HIT":
            encoded = json.dumps(decision.response, allow_nan=False).encode()
            decorated = self._decision_send(send, "HIT", decision.reason, decision.similarity)
            await decorated(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await decorated({"type": "http.response.body", "body": encoded})
            return
        parts = []
        size = 0
        complete = False
        safe = False
        decorated = self._decision_send(send, decision.outcome, decision.reason)

        async def capture(message):
            nonlocal size, complete, safe
            if message["type"] == "http.response.start":
                response_headers = [(k.lower(), v.lower()) for k, v in message.get("headers", [])]
                names = {k for k, _ in response_headers}
                control = b",".join(v for k, v in response_headers if k == b"cache-control")
                media = [
                    v.split(b";", 1)[0].strip() for k, v in response_headers if k == b"content-type"
                ]
                safe = (
                    message.get("status") == 200
                    and b"set-cookie" not in names
                    and b"vary" not in names
                    and b"no-store" not in control
                    and b"private" not in control
                    and media == [b"application/json"]
                )
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                size += len(chunk)
                if safe and size <= limit:
                    parts.append(chunk)
                else:
                    safe = False
                    parts.clear()
                complete = not message.get("more_body", False)
            await decorated(message)

        await self.app(scope, replay, capture)
        if safe and complete and not disconnected and decision.outcome != "BYPASS":
            try:
                response = json.loads(b"".join(parts), object_pairs_hook=_object)
            except (ValueError, UnicodeDecodeError):
                return
            try:
                await self.cache.aset(query, response, self.namespace, context=ctx)
            except CacheError:
                # Upstream already sent its response. It cannot be retracted or
                # safely executed again because optional persistence failed.
                self.cache._observability.event(
                    warning=True, stage="store", reason="cache_unavailable"
                )

    async def aclose(self):
        """Wire into application lifespan when this middleware created its cache."""
        if self._owns_cache:
            await self.cache.aclose()

    @staticmethod
    def _replay(frames, receive):
        iterator = iter(frames)

        async def read():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        return read

    @staticmethod
    def _decision_send(send, outcome, reason, similarity=None):
        async def decorated(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                headers = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if not k.lower().startswith(b"x-longcache")
                ]
                headers.extend(
                    [(b"x-longcache", outcome.encode()), (b"x-longcache-reason", reason.encode())]
                )
                if similarity is not None:
                    headers.append((b"x-longcache-similarity", f"{similarity:.6f}".encode()))
                message["headers"] = headers
            await send(message)

        return decorated


class StreamingCacheAdapter:
    """Compatibility pass-through; streaming cache replay is unsupported in v0.2."""

    cache_bypass_reason = "unsupported_request_shape"

    def __init__(self, cache, stream_func, namespace="streaming", replay_delay=0, **kwargs):
        self.cache = cache
        self.stream_func = stream_func

    def stream(self, query):
        yield from self.stream_func(query)

    async def astream(self, query):
        async for chunk in self.stream_func(query):
            yield chunk
