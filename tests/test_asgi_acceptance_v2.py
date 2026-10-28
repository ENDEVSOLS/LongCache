"""ASGI boundary cases from the v0.2 acceptance matrix."""

import asyncio
import json
from unittest.mock import AsyncMock
import pytest
from longcache import CacheConfig, LongCache
from longcache.middleware import LongCacheMiddleware
from tests.test_embedding_contract_v2 import TinyEmbedder
from tests.test_identity_v2 import context
from tests.test_integrations_v2 import completion, payload


def request_frames(body=None):
    return [
        {
            "type": "http.request",
            "body": body if body is not None else json.dumps(payload()).encode(),
            "more_body": False,
        }
    ]


async def drive(app, frames=None, headers=()):
    messages = iter(frames or request_frames())
    sent = []

    async def receive():
        return next(messages, {"type": "http.disconnect"})

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": list(headers),
        },
        receive,
        send,
    )
    return sent


async def drain(receive):
    frames = []
    while True:
        message = await receive()
        frames.append(message)
        if message["type"] == "http.disconnect" or not message.get("more_body", False):
            return frames


@pytest.fixture
def cache():
    with LongCache(
        CacheConfig(backend="memory", max_capture_bytes=512), embedder=TinyEmbedder()
    ) as cache:
        yield cache


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_case",
    [
        "oversized",
        "incomplete",
        "bad_json",
        "http_400",
        "http_500",
        "cookie",
        "vary",
        "private",
        "no_store",
        "duplicate_no_store",
        "not_json",
    ],
)
async def test_ineligible_http_responses_preserve_body_status_and_never_store(cache, response_case):
    body = json.dumps(completion()).encode()
    headers = [(b"content-type", b"application/json")]
    status = 200
    final = True
    if response_case == "oversized":
        body = json.dumps(completion("x" * 1000)).encode()
    elif response_case == "incomplete":
        final = False
    elif response_case == "bad_json":
        body = b'{"broken":'
    elif response_case.startswith("http_"):
        status = int(response_case[5:])
    elif response_case == "cookie":
        headers.append((b"set-cookie", b"session=synthetic"))
    elif response_case == "vary":
        headers.append((b"vary", b"authorization"))
    elif response_case == "private":
        headers.append((b"cache-control", b"private"))
    elif response_case == "no_store":
        headers.append((b"cache-control", b"no-store"))
    elif response_case == "duplicate_no_store":
        headers.extend([(b"cache-control", b"no-store"), (b"cache-control", b"public")])
    elif response_case == "not_json":
        headers = [(b"content-type", b"application/jsonp")]

    async def upstream(scope, receive, send):
        await drain(receive)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": not final})

    app = LongCacheMiddleware(upstream, cache=cache, context=context())
    for _ in range(2):
        sent = await drive(app)
        assert sent[0]["status"] == status
        assert b"".join(m.get("body", b"") for m in sent) == body
        assert sent[-1].get("more_body", False) == (not final)
    assert cache.entries(operator=True) == []


@pytest.mark.asyncio
async def test_oversized_request_replays_every_original_frame_without_cache_access(cache):
    frames = [
        {"type": "http.request", "body": b"a" * 300, "more_body": True},
        {"type": "http.request", "body": b"b" * 300, "more_body": True},
        {"type": "http.request", "body": b"c" * 300, "more_body": False},
    ]
    received = []
    cache.alookup = AsyncMock(side_effect=AssertionError("Oversized input reached lookup"))
    cache.aset = AsyncMock(side_effect=AssertionError("Oversized input was stored"))

    async def upstream(scope, receive, send):
        received.extend(await drain(receive))
        await send({"type": "http.response.start", "status": 422, "headers": []})
        await send({"type": "http.response.body", "body": b"upstream validation"})

    sent = await drive(LongCacheMiddleware(upstream, cache=cache, context=context()), frames)
    assert received == frames
    assert dict(sent[0]["headers"])[b"x-longcache"] == b"BYPASS"


@pytest.mark.asyncio
@pytest.mark.parametrize("at", ["request", "response"])
async def test_disconnect_never_admits_partial_or_abandoned_response(cache, at):
    frames = request_frames()
    if at == "request":
        frames = [
            {"type": "http.request", "body": b'{"model":', "more_body": True},
            {"type": "http.disconnect"},
        ]

    async def upstream(scope, receive, send):
        received = await drain(receive)
        if received[-1]["type"] != "http.disconnect":
            assert (await receive())["type"] == "http.disconnect"
        # A completed send after disconnect must not revive cache eligibility.
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": json.dumps(completion()).encode()})

    await drive(LongCacheMiddleware(upstream, cache=cache, context=context()), frames)
    assert cache.entries(operator=True) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [RuntimeError, asyncio.CancelledError])
async def test_upstream_exception_and_cancellation_do_not_write(cache, exception):
    async def upstream(scope, receive, send):
        await drain(receive)
        raise exception("fixture failure")

    with pytest.raises(exception):
        await drive(LongCacheMiddleware(upstream, cache=cache, context=context()))
    assert cache.entries(operator=True) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolver", [lambda scope: None, lambda scope: "untrusted", lambda scope: 1 / 0]
)
async def test_failed_identity_resolver_never_falls_back_to_fixed_private_identity(cache, resolver):
    cache.alookup = AsyncMock(side_effect=AssertionError("Identity failure reached lookup"))

    async def upstream(scope, receive, send):
        await drain(receive)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"passthrough"})

    sent = await drive(
        LongCacheMiddleware(upstream, cache=cache, context=context(), scope_resolver=resolver),
        headers=[(b"x-user-id", b"spoofed")],
    )
    assert dict(sent[0]["headers"])[b"x-longcache-reason"] == b"missing_identity"
