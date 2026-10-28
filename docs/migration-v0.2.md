# Upgrade and rollback: v0.1.1 to v0.2

1. Preserve the v0.1.1 environment and a consistent backup of its store. Retain Python 3.10 if rollback needs it.
2. Install v0.2 on Python 3.11/3.12. Configure trusted scope and complete immutable execution context.
3. Use a fresh SQLite path (default `.longcache/v2/cache.db`) or v0.2 Redis keyspace/generation. Legacy entries are not relabeled or imported.
4. Run representative isolation, admission, verification, and replay tests. Expect initial misses while new responses are admitted.
5. Roll back by stopping v0.2 writers and restoring the preserved environment/store. Never open a v0.2 store with v0.1.1.

## Interface changes

- `RequestContext` is required for caching. Missing private identity bypasses lookup and storage.
- `set`/`aset` return `StoreDecision`. `lookup`/`alookup` expose richer decisions; `get` still returns response or `None`.
- Inspection is redacted and decision-based, including on a miss. Raw payload exports require explicit scope or operator access.
- Generic frameworks and all streaming helpers are pass-through until proven safe.
- Arbitrary objects and `None` are not stored. OpenAI text completion requires an explicit supported finish reason and response codec.
- Configuration dictionary typos raise rather than being ignored. Conflicting context/legacy keywords raise.
- Global clearing requires `operator=True`; namespace alone does not authorize cross-user maintenance.
- Manual `record_hit`/`record_miss` accounting is removed; accepted decoded cache results drive hit counters. Process counters reset on restart and unmeasured savings remain zero.

## Embedding changes

A generation has one explicit model/revision/dimension/preprocessing/normalization descriptor. A same-size different model remains incompatible. Use a new store generation when it changes; automatic re-embedding is not part of v0.2.

SQLite initialization rolls back incomplete schema creation, including process termination before commit. Incompatible partial schemas and missing manifests in populated SQLite/Redis stores are refused. Use a fresh path/generation; do not fabricate a manifest for existing entries. Recovery tests include an actual v0.1.1 backup-and-restore round-trip. See the [acceptance evidence](plans/v0.2.0-acceptance.md).

Legacy SQLite opening fails before modifying old data. Raw copies of a live SQLite WAL file are not a complete backup method. No upgrade step automatically deletes the old store.
