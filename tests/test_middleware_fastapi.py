"""
Integration tests for LongCacheMiddleware against a real FastAPI/Starlette app.

Uses httpx.ASGITransport (already a dev dependency) to drive real ASGI
requests through the middleware, rather than mocking the ASGI scope
directly. This exercises the actual request/response interception path.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from httpx import ASGITransport, AsyncClient

from longcache.config import CacheConfig
from longcache.middleware import LongCacheMiddleware


from tests.helpers import HashEmbedder, trusted_context


def _make_mock_embedder():
    return HashEmbedder()


def _build_app(cache):
    """A minimal FastAPI app with LongCacheMiddleware and one upstream route."""
    app = FastAPI()

    call_count = {"n": 0}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict):
        call_count["n"] += 1
        last_user = ""
        for msg in reversed(payload.get("messages", [])):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        return JSONResponse(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"answer to: {last_user}"},
                    }
                ]
            }
        )

    app.add_middleware(
        LongCacheMiddleware,
        cache=cache,
        context=trusted_context(),
        cached_routes={"/v1/chat/completions"},
    )
    return app, call_count


@pytest.fixture
def cache():
    with patch("longcache.core.cache.LocalEmbedder") as MockEmbedder:
        mock_embedder = _make_mock_embedder()
        MockEmbedder.return_value = mock_embedder

        from longcache import LongCache

        c = LongCache(CacheConfig(backend="memory", similarity_threshold=0.9))
        c.embedder = mock_embedder
        yield c


class TestLongCacheMiddleware:
    async def test_miss_then_hit_via_real_asgi_app(self, cache):
        app, call_count = _build_app(cache)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "What is the refund policy?"}],
            }

            # First request — MISS, should hit the upstream handler
            resp1 = await client.post("/v1/chat/completions", json=payload)
            assert resp1.status_code == 200
            assert resp1.headers["x-longcache"] == "MISS"
            assert call_count["n"] == 1
            body1 = resp1.json()
            assert (
                "answer to: What is the refund policy?" in body1["choices"][0]["message"]["content"]
            )

            # Second identical request — HIT, upstream should NOT be called again
            resp2 = await client.post("/v1/chat/completions", json=payload)
            assert resp2.status_code == 200
            assert resp2.headers.get("x-longcache") == "HIT"
            assert call_count["n"] == 1
            assert resp2.json() == body1

    async def test_different_model_or_history_bypasses_reuse(self, cache):
        app, call_count = _build_app(cache)
        transport = ASGITransport(app=app)
        messages = [{"role": "user", "content": "same question"}]
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/v1/chat/completions", json={"model": "a", "messages": messages})
            await client.post("/v1/chat/completions", json={"model": "b", "messages": messages})
            history = [{"role": "user", "content": "earlier"}, *messages]
            await client.post("/v1/chat/completions", json={"model": "a", "messages": history})
            await client.post("/v1/chat/completions", json={"model": "a", "messages": history})
        assert call_count["n"] == 4

    async def test_authenticated_requests_bypass_shared_cache(self, cache):
        app, call_count = _build_app(cache)
        transport = ASGITransport(app=app)
        payload = {"model": "a", "messages": [{"role": "user", "content": "question"}]}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for _ in range(2):
                await client.post(
                    "/v1/chat/completions",
                    json=payload,
                    headers={"authorization": "Bearer private-token"},
                )
        assert call_count["n"] == 2

    async def test_uncached_route_passes_through(self, cache):
        app, call_count = _build_app(cache)

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}
            # Uncached route should never touch the chat completion counter
            assert call_count["n"] == 0

    async def test_non_json_body_passes_through(self, cache):
        app, call_count = _build_app(cache)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions", content=b"not json", headers={"content-type": "text/plain"}
            )
            # FastAPI will reject invalid body for the `dict` param with a 422,
            # but the middleware itself must not crash on unparsable JSON —
            # it should pass through to the app rather than caching garbage.
            assert resp.status_code in (422, 400)
            assert resp.headers["x-longcache"] == "BYPASS"

    async def test_missing_user_message_passes_through_without_caching(self, cache):
        app, call_count = _build_app(cache)
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            payload = {"messages": [{"role": "system", "content": "no user turn here"}]}
            resp1 = await client.post("/v1/chat/completions", json=payload)
            resp2 = await client.post("/v1/chat/completions", json=payload)

            assert resp1.status_code == 200
            assert resp2.status_code == 200
            # No query extracted -> never cached -> upstream called both times
            assert call_count["n"] == 2

    async def test_repr(self, cache):
        app, _ = _build_app(cache)
        middleware = None
        for m in app.user_middleware:
            if m.cls is LongCacheMiddleware:
                middleware = m
                break
        assert middleware is not None
