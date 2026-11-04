# LlamaIndex compatibility

`CachedQueryEngine` delegates `query` / `aquery` to the engine without cache reads or writes in v0.5 (`pass-through` tier in `PUBLIC_API_STABILITY`, `cache_bypass_reason = "unsupported_request_shape"`). Opaque retrieval/model state and unproven response codecs cannot safely become cached hits.

A separately implemented query-only workflow can use the manual interface with explicit corpus/document versions and a tested strict response type. Retrieval metadata is not automatically discovered or versioned.
