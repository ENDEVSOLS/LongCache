# Reuse strategies

Use private user-level scope for personalized results. Enable tenant sharing only when the application explicitly permits every user in the tenant to see the same answers. Public mode requires non-personalized application/operation context.

Keep source versions explicit for RAG workloads. Changing `corpus_version` or `document_versions` separates future lookups from old answers. `invalidate_by_tag(..., context=...)` removes only the current exact partition; broader operator maintenance is explicit.

Do not cache hidden conversation memory, unrepresented tool input, or mutable framework state. Generic framework and streaming wrappers pass through in v0.2. Wrap a proven query-only function with explicit complete context instead.

For response admission and local verification, use [policy hooks](../api-reference.md#policy-and-embedding-interfaces). A higher hit rate is useful only alongside acceptable false-hit rate and safe-paraphrase coverage.
