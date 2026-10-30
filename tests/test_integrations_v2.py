import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import json
import pytest
from httpx import ASGITransport, AsyncClient
from longcache import CacheConfig, LongCache, RequestScope
from longcache.integrations.openai import CachedOpenAI
from longcache.middleware import LongCacheMiddleware
from tests.test_identity_v2 import context
from tests.test_embedding_contract_v2 import TinyEmbedder


def completion(text="answer"):
    return {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
    }


def payload():
    return {"model": "m", "messages": [{"role": "user", "content": "refund policy"}]}


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_openai_scoped_replay_and_context_changes(async_mode):
    call = (AsyncMock if async_mode else Mock)(return_value=completion())
    upstream = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)))
    with LongCache(CacheConfig(backend="memory", ttl_seconds=0), embedder=TinyEmbedder()) as cache:
        wrapped = CachedOpenAI(upstream, cache=cache, context=context())

        async def invoke(**kwargs):
            if async_mode:
                return await wrapped.chat.completions.create(**kwargs)
            return await asyncio.to_thread(wrapped.chat.completions.create, **kwargs)

        assert await invoke(**payload()) == completion()
        assert await invoke(**payload()) == completion()
        assert call.call_count == 1
        await invoke(**payload(), cache_context=context(scope=RequestScope("tenant", "bob")))
        assert call.call_count == 2
        await invoke(**(payload() | {"temperature": None}))
        await invoke(**(payload() | {"temperature": 0.0}))
        assert call.call_count == 4
        assert cache.stats(operator=True)["hits"] == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"model": " "},
        {"messages": [{"role": "user", "content": " "}]},
        {"tools": []},
        {"top_p": 0.5},
        {"response_format": {"type": "json_object"}},
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "a"},
            ]
        },
    ],
)
def test_unsupported_requests_bypass_both_lookup_and_storage(extra):
    call = Mock(return_value=completion())
    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:
        cache.lookup = Mock(side_effect=AssertionError("unsupported lookup"))
        cache.set = Mock(side_effect=AssertionError("unsupported store"))
        wrapped = CachedOpenAI(
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call))),
            cache=cache,
            context=context(),
        )
        wrapped.chat.create(**(payload() | extra))
        wrapped.chat.create(**(payload() | extra))
        assert call.call_count == 2


def test_sdk_response_survives_restart(tmp_path):
    from openai.types.chat import ChatCompletion

    call = Mock(return_value=ChatCompletion.model_validate(completion()))
    upstream = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)))
    config = CacheConfig(db_path=str(tmp_path / "v2.db"), ttl_seconds=0)
    for _ in range(2):
        with LongCache(config, embedder=TinyEmbedder()) as cache:
            wrapped = CachedOpenAI(
                upstream, cache=cache, context=context(), response_codec="openai-sdk-v1"
            )
            assert isinstance(wrapped.chat.create(**payload()), ChatCompletion)
    assert call.call_count == 1


@pytest.mark.parametrize("change", ["missing_finish", "refusal", "tool", "invalid_choice"])
def test_bad_completions_never_admitted(change):
    data = completion()
    if change == "missing_finish":
        data["choices"][0].pop("finish_reason")
    elif change == "refusal":
        data["choices"][0]["message"]["refusal"] = "cannot"
    elif change == "tool":
        data["choices"][0]["message"]["tool_calls"] = [{"name": "tool"}]
    else:
        data["choices"] = ["fake"]
    call = Mock(return_value=data)
    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:
        wrapped = CachedOpenAI(
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call))),
            cache=cache,
            context=context(),
        )
        wrapped.chat.create(**payload())
        wrapped.chat.create(**payload())
        assert call.call_count == 2
        assert not cache.entries(operator=True)


@pytest.mark.asyncio
async def test_asgi_headers_auth_scope_and_duplicate_keys():
    calls = []

    async def app(scope, receive, send):
        calls.append(scope)
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json"), (b"x-longcache", b"forged")],
            }
        )
        await send({"type": "http.response.body", "body": json.dumps(completion()).encode()})

    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:
        wrapped = LongCacheMiddleware(app, cache=cache, context=context())
        async with AsyncClient(
            transport=ASGITransport(app=wrapped), base_url="http://test"
        ) as client:
            first = await client.post("/v1/chat/completions", json=payload())
            second = await client.post("/v1/chat/completions", json=payload())
            assert first.headers["x-longcache"] == "MISS"
            assert second.headers["x-longcache"] == "HIT"
            assert "x-longcache-similarity" in second.headers
            assert len(calls) == 1
            auth = await client.post(
                "/v1/chat/completions", json=payload(), headers={"authorization": "secret"}
            )
            assert auth.headers["x-longcache"] == "BYPASS"
            duplicate = await client.post(
                "/v1/chat/completions", content=b'{"model":"a","model":"b"}'
            )
            assert duplicate.headers["x-longcache"] == "BYPASS"
        # Resolver input is trusted ASGI state in real applications, not raw headers.
        wrapped = LongCacheMiddleware(
            app,
            cache=cache,
            context=context(),
            scope_resolver=lambda _: RequestScope("tenant", "trusted-user"),
        )
        async with AsyncClient(
            transport=ASGITransport(app=wrapped), base_url="http://test"
        ) as client:
            await client.post(
                "/v1/chat/completions", json=payload(), headers={"authorization": "secret"}
            )
            hit = await client.post(
                "/v1/chat/completions", json=payload(), headers={"authorization": "secret"}
            )
            assert hit.headers["x-longcache"] == "HIT"


@pytest.mark.parametrize(
    "module,cls,method",
    [
        ("langchain", "CachedLLM", "invoke"),
        ("langchain", "CachedChain", "invoke"),
        ("llamaindex", "CachedQueryEngine", "query"),
    ],
)
def test_opaque_frameworks_do_not_access_cache(module, cls, method):
    import importlib

    upstream = Mock()
    getattr(upstream, method).return_value = "answer"
    cache = Mock()
    cache.get.side_effect = AssertionError("opaque lookup")
    cache.set.side_effect = AssertionError("opaque write")
    wrapper = getattr(importlib.import_module("longcache.integrations." + module), cls)(
        upstream, cache=cache
    )
    assert getattr(wrapper, method)("q") == "answer"
    assert getattr(wrapper, method)("q") == "answer"
    assert getattr(upstream, method).call_count == 2


def test_openai_invalid_unicode_passes_through_without_cache():
    call = Mock(return_value=completion())
    with LongCache(CacheConfig(backend="memory"), embedder=TinyEmbedder()) as cache:
        cache.lookup = Mock(side_effect=AssertionError("invalid Unicode was looked up"))
        wrapped = CachedOpenAI(
            SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call))),
            cache=cache,
            context=context(),
        )
        wrapped.chat.create(model="m", messages=[{"role": "user", "content": "bad\ud800"}])
        assert call.call_count == 1
