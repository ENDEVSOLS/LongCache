"""OpenAI adapter for the tested non-streaming single-turn text subset."""

from __future__ import annotations
from dataclasses import replace
import inspect
import math
from longcache import CacheConfig, Instruction, LongCache, RequestContext


def prepare_request(kwargs, base, codec):
    if not isinstance(base, RequestContext):
        return None, "missing_identity"
    if set(kwargs) - {"messages", "model", "temperature", "stream"}:
        return None, "unsupported_parameter"
    if kwargs.get("stream", False) is not False:
        return None, "unsupported_request_shape"
    model, messages = kwargs.get("model"), kwargs.get("messages")
    if (
        type(model) is not str
        or not model.strip()
        or type(messages) is not list
        or not 1 <= len(messages) <= 3
    ):
        return None, "unsupported_request_shape"
    instructions = []
    for message in messages[:-1]:
        if (
            type(message) is not dict
            or set(message) != {"role", "content"}
            or message["role"] not in ("system", "developer")
            or type(message["content"]) is not str
        ):
            return None, "unsupported_request_shape"
        instructions.append(Instruction(message["role"], message["content"]))
    last = messages[-1]
    if (
        type(last) is not dict
        or set(last) != {"role", "content"}
        or last["role"] != "user"
        or type(last["content"]) is not str
        or not last["content"].strip()
    ):
        return None, "unsupported_request_shape"
    try:
        model.encode("utf-8")
        last["content"].encode("utf-8")
        for instruction in instructions:
            instruction.content.encode("utf-8")
    except UnicodeError:
        return None, "unsupported_request_shape"
    generation = {}
    if "temperature" in kwargs:
        temp = kwargs["temperature"]
        if temp is not None and (
            type(temp) not in (int, float) or not math.isfinite(temp) or not 0 <= temp <= 2
        ):
            return None, "unsupported_parameter"
        generation["temperature"] = temp
    return (
        last["content"],
        replace(
            base, model=model, instructions=tuple(instructions), generation=generation, codec=codec
        ),
    ), None


class CachedOpenAI:
    def __init__(
        self,
        client,
        cache=None,
        config=None,
        namespace="openai",
        *,
        context=None,
        response_codec=None,
    ):
        self._client = client
        self.cache = cache if cache is not None else LongCache(config or CacheConfig())
        self._namespace = namespace
        self.context = context
        self.codec = response_codec or (
            "openai-sdk-v1" if type(client).__module__.startswith("openai") else "openai-dict-v1"
        )
        if self.codec not in ("openai-sdk-v1", "openai-dict-v1"):
            raise ValueError("Unsupported OpenAI response codec")
        self.chat = _ChatProxy(self)

    def _extract_query(self, messages):
        """Legacy inspection helper; never used to decide cache eligibility."""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return message.get("content", "")
        return str(messages)

    def __getattr__(self, name):
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, parent):
        cls = (
            AsyncCachedChatCompletions
            if inspect.iscoroutinefunction(parent._client.chat.completions.create)
            else CachedChatCompletions
        )
        self.completions = cls(parent)
        self._parent = parent

    def create(self, **kwargs):
        return self.completions.create(**kwargs)

    async def acreate(self, **kwargs):
        return await self.completions.acreate(**kwargs)

    def __getattr__(self, name):
        return getattr(self._parent._client.chat, name)


class CachedChatCompletions:
    def __init__(self, parent):
        self._parent = parent

    def create(self, *, cache_context=None, **kwargs):
        parent = self._parent
        prepared, _ = prepare_request(
            kwargs, cache_context if cache_context is not None else parent.context, parent.codec
        )
        if prepared is None:
            return parent._client.chat.completions.create(**kwargs)
        query, ctx = prepared
        decision = parent.cache.lookup(query, parent._namespace, context=ctx)
        if decision.outcome == "HIT":
            return decision.response
        result = parent._client.chat.completions.create(**kwargs)
        parent.cache.set(query, result, parent._namespace, context=ctx)
        return result

    async def acreate(self, *, cache_context=None, **kwargs):
        parent = self._parent
        client = parent._client.chat.completions
        call = (
            client.create
            if inspect.iscoroutinefunction(client.create)
            else getattr(client, "acreate", None)
        )
        if not inspect.iscoroutinefunction(call):
            raise TypeError("An async OpenAI client is required")
        prepared, _ = prepare_request(
            kwargs, cache_context if cache_context is not None else parent.context, parent.codec
        )
        if prepared is None:
            return await call(**kwargs)
        query, ctx = prepared
        decision = await parent.cache.alookup(query, parent._namespace, context=ctx)
        if decision.outcome == "HIT":
            return decision.response
        result = await call(**kwargs)
        await parent.cache.aset(query, result, parent._namespace, context=ctx)
        return result


class AsyncCachedChatCompletions(CachedChatCompletions):
    async def create(self, **kwargs):
        return await self.acreate(**kwargs)
