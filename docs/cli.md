# CLI

All commands default to `.longcache/v2/cache.db`. Add `--db-path` for another owned SQLite store. `--context-file` contains the JSON fields for `RequestContext`, nested `scope`, and instruction objects.

```bash
longcache stats --context-file context.json
longcache inspect --query 'refund policy' --context-file context.json
longcache export --context-file context.json -o scoped.json
longcache clear --context-file context.json --yes
longcache export --operator -o all-owned.json
longcache clear --all --operator --yes
```

Scoped commands select one exact execution partition; `--namespace` does not grant all-tenant access. Operator export/deletion requires explicit flags. Destructive commands retain confirmation unless `--yes` is supplied. Exports deliberately include raw owned payloads; inspection is redacted.

Metadata-only commands do not download a model. Vector inspection uses the configured default local model; use the Python interface for custom injected embedders. Process counters reset after restart while entry hit counts persist.
