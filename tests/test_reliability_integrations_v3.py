"""Supported adapters share the core execution and preserve borrowed ownership."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from longcache import CacheConfig, LongCache
from longcache.integrations.openai import CachedOpenAI
from longcache.middleware import LongCacheMiddleware
from tests.helpers import HashEmbedder, trusted_context
from tests.test_singleflight_v3 import joined_event
from tests.test_integrations_v2 import completion, payload
from tests.test_asgi_acceptance_v2 import drive, drain


def make_cache():
    return LongCache(
        CacheConfig(backend="memory", ttl_seconds=0),
        embedder=HashEmbedder(),
        context=trusted_context(),
    )


@pytest.mark.parametrize("adapter", ["decorator", "openai"])
def test_sync_adapter_coalesces_twenty_requests(monkeypatch, adapter):
    with make_cache() as cache:
        joined = joined_event(cache, monkeypatch, 19)
        calls = []

        def origin(*args, **kwargs):
            calls.append(1)
            assert joined.wait(10)
            return completion()

        if adapter == "decorator":
            wrapped = cache.cached(origin)

            def invoke():
                return wrapped("policy")
        else:
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=origin))
            )
            wrapped = CachedOpenAI(client, cache=cache, context=trusted_context())

            def invoke():
                return wrapped.chat.completions.create(**payload())

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(invoke) for _ in range(20)]
            results = [f.result(timeout=15) for f in futures]
        assert results == [completion()] * 20
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["decorator", "openai"])
async def test_async_adapter_coalesces_twenty_requests(monkeypatch, adapter):
    async with make_cache() as cache:
        joined = joined_event(cache, monkeypatch, 19)
        calls = []

        async def origin(*args, **kwargs):
            calls.append(1)
            assert await asyncio.to_thread(joined.wait, 10)
            return completion()

        if adapter == "decorator":
            wrapped = cache.cached(origin)

            def invoke():
                return wrapped("policy")
        else:
            client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=origin))
            )
            wrapped = CachedOpenAI(client, cache=cache, context=trusted_context())

            def invoke():
                return wrapped.chat.completions.create(**payload())

        results = await asyncio.wait_for(asyncio.gather(*[invoke() for _ in range(20)]), 15)
        assert results == [completion()] * 20
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_borrowed_cache_and_client_are_not_closed(monkeypatch):
    async with make_cache() as cache:

        def forbidden():
            pytest.fail("borrowed provider client was closed")

        client = SimpleNamespace(
            close=forbidden,
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: completion())),
        )
        wrapped = CachedOpenAI(client, cache=cache, context=trusted_context())
        wrapped.close()
        await wrapped.aclose()
        assert not cache._closed


@pytest.mark.asyncio
async def test_blocking_shared_execution_is_rejected_on_event_loop():
    async with make_cache() as cache:
        with pytest.raises(RuntimeError, match="aget_or_compute"):
            cache.get_or_compute("policy", lambda: "answer")


@pytest.mark.asyncio
async def test_asgi_preserves_sent_response_after_strict_store_failure(monkeypatch):
    async with make_cache() as cache:
        calls = []
        body = json.dumps(completion()).encode()

        async def upstream(scope, receive, send):
            calls.append(1)
            await drain(receive)
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})

        def broken(*_):
            raise OSError("store unavailable")

        monkeypatch.setattr(cache.backend, "store", broken)
        app = LongCacheMiddleware(upstream, cache=cache, context=trusted_context())
        sent = await drive(app)
        assert sent[0]["status"] == 200
        assert b"".join(frame.get("body", b"") for frame in sent) == body
        assert len(calls) == 1
        assert cache.metrics(operator=True)["health"] == "degraded"
