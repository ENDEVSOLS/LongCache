# LlamaIndex compatibility

`CachedQueryEngine` delegates `query` / `aquery` to the engine without cache reads or writes in v0.3. Opaque retrieval/model state and unproven response codecs cannot safely become cached hits.

A separately implemented query-only workflow can use the manual interface with explicit corpus/document versions and a tested strict response type. Retrieval metadata is not automatically discovered or versioned.
