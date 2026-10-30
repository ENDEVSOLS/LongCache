# OpenAI

See the [complete OpenAI example](../index.md#openai). Supply trusted fixed `context` or a per-call `cache_context`; the latter is not sent upstream. Both `OpenAI` and `AsyncOpenAI` use `chat.completions.create` with the appropriate sync/await behavior.

The initial cached request subset accepts `model`, single user-turn `messages` with ordered system/developer instructions, `temperature`, and `stream=False`. The adapter records actual allowed values and distinguishes omission, null, and explicit values. Unknown options, histories, tools, multimodal, streaming, and structured-output settings bypass.

Supported completions contain exactly one complete assistant text choice with `finish_reason="stop"` and no refusal/tool output. SDK objects use a versioned codec and preserve their type through SQLite/Redis restart. Custom clients returning dictionary responses can declare `response_codec="openai-dict-v1"` explicitly.

v0.3 routes supported sync/async calls through shared execution. Set finite provider
timeouts yourself; SDK retry settings remain provider-client behavior. LongCache adds
no provider retry. Close borrowed clients and supplied caches in application context
managers; `CachedOpenAI.close/aclose` only closes an internally created cache. See
[reliability](../reliability.md) for waiting, cancellation, capacity, and failure rules.
