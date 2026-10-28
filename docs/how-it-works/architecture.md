# Request lifecycle

1. Resolve trusted scope and reject unsupported shapes before embedding.
2. Canonicalize exact execution context. The current query stays outside the semantic partition.
3. Validate the store manifest and embedding generation.
4. Retrieve scored, live candidates from the exact partition without recording hits.
5. Apply current admission policy, verifier checks, and strict response decoding.
6. Recheck identity, payload, expiry, and deletion under the mutation lock.
7. Commit one hit, or return a reasoned miss/bypass.
8. On a provider result, admit and strictly encode before storing.

Identity includes scope policy, tenant/user, logical namespace, operation, provider/deployment, model, instructions, generation parameter states, versions, codec, policy, and embedding fingerprint. Full SHA-256 digests are identifiers, not encryption or authentication.

Hooks execute outside storage/index locks. Async calls offload blocking work. Cancellation observed before the commit prevents accounting/storage; an already-started commit may complete. There is no singleflight or distributed consistency promise in v0.2.

Format-2 manifests prevent incompatible vectors from entering the index. The local FAISS index is rebuilt from compatible live entries on restart and after maintenance, retaining flat/HNSW configuration.
