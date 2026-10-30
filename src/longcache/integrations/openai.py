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


def snapshot_request(kwargs, query, context):
    """Reconstruct the validated subset from the same snapshot used for identity."""
    request = {
        "model": context.model,
        "messages": [{"role": i.role, "content": i.content} for i in context.instructions]
        + [{"role": "user", "content": query}],
    }
    if "temperature" in context.generation:
        request["temperature"] = context.generation["temperature"]
    if "stream" in kwargs:
        request["stream"] = False
    return request


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
        self._owns_cache = cache is None
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

    def close(self):
        """Close only an internally constructed cache; the provider client is borrowed."""
        if self._owns_cache:
            self.cache.close()

    async def aclose(self):
        if self._owns_cache:
            await self.cache.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._owns_cache:
            return self.cache.__exit__(exc_type, exc, traceback)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        if self._owns_cache:
            return await self.cache.__aexit__(exc_type, exc, traceback)


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
        prepared, reason = prepare_request(
            kwargs, cache_context if cache_context is not None else parent.context, parent.codec
        )
        if prepared is None:
            return parent.cache._pass_through(
                lambda: parent._client.chat.completions.create(**kwargs), reason, "openai"
            )
        query, ctx = prepared
        kwargs = snapshot_request(kwargs, query, ctx)
        return parent.cache.get_or_compute(
            query,
            lambda: parent._client.chat.completions.create(**kwargs),
            namespace=parent._namespace,
            context=ctx,
            _entry_point="openai",
        )

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
        prepared, reason = prepare_request(
            kwargs, cache_context if cache_context is not None else parent.context, parent.codec
        )
        if prepared is None:
            return await parent.cache._apass_through(lambda: call(**kwargs), reason, "openai")
        query, ctx = prepared
        kwargs = snapshot_request(kwargs, query, ctx)
        return await parent.cache.aget_or_compute(
            query,
            lambda: call(**kwargs),
            namespace=parent._namespace,
            context=ctx,
            _entry_point="openai",
        )


class AsyncCachedChatCompletions(CachedChatCompletions):
    async def create(self, **kwargs):
        return await self.acreate(**kwargs)
