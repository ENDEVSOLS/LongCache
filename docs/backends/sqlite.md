# SQLite

The default path is `.longcache/v2/cache.db`. SQLite stores vectors, strict JSON response payloads, versioned identity metadata, tags, and a store manifest. WAL mode supports the declared single-process threading model.

A legacy file without format-2 metadata is refused before migration. Use a fresh path and preserve the old store for rollback. Back up live WAL databases with a consistent SQLite backup procedure.

Schema creation is transactional: an exception or process termination before commit rolls back partial initialization. Incompatible partial schemas are refused. A populated store with a missing manifest is refused rather than assigned a new identity; preserve it for diagnosis and use a fresh path.

Metadata-only counts/export/maintenance do not need model weights. Normal vector operations validate the persisted embedding manifest before index rebuild or writes.

In v0.3, one backend-owned connection is shared through serialized access; it is no
longer a separate connection per worker thread. Cache shutdown drains registered
users before closing it. `backend_timeout_seconds` controls SQLite's busy timeout;
it is not an end-to-end request deadline. Existing compatible v0.2 format-2 stores
remain usable without a reset. See [v0.3 migration](../migration-v0.3.md).
