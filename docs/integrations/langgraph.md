# LangGraph compatibility

`CachedState.check` / `acheck` return a miss and can preserve a query for caller compatibility. `save` / `asave` do not cache opaque graph state in v0.2. Conversation history, tool results, and hidden memory require a complete identity/replay contract before caching can be enabled.

Use the [explicit manual interface](../api-reference.md) for a separately proven single-turn workload. A latest-message string alone is insufficient graph identity.
