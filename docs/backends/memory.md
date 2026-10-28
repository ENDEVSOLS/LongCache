# Memory

`CacheConfig(backend="memory")` keeps entries in process memory and discards them on shutdown. It follows the same identity, vector, manifest, TTL, policy, and scoped-maintenance rules as persistent backends.

Use it for deterministic tests, local experiments, and bounded ephemeral workloads. It has no cross-process sharing and needs no sweeper thread; expiry is checked during selection and final replay.
