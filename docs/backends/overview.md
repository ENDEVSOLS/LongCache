# Storage support

Memory, SQLite, and synchronous Redis backends support format-2 entries within one cache process. They preserve identity metadata, codec/policy versions, expiry, tags, payload integrity metadata, and an immutable embedding manifest.

A storage generation has one embedding fingerprint. A changed model/revision/preprocessing setting needs a new generation or path. Legacy SQLite stores fail without modification; Redis uses a new fully versioned keyspace.

Async cache methods over these backends use executor-based blocking operations. The standalone async Redis backend remains experimental and is not supported through `LongCache`. Redis response persistence does not synchronize FAISS indexes across processes.
