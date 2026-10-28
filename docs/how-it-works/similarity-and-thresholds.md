# Similarity and thresholds

The default threshold is 0.92 with normalized vectors. Flat search uses inner product, equivalent to cosine similarity for the validated vector contract. HNSW provides approximate retrieval and may miss a useful candidate.

Similarity selects a candidate inside an exact execution partition. Admission, verification, strict decoding, and final expiry/deletion checks determine whether it can be served. A closer rejected candidate can fall through to a farther verified candidate within the verification budget.

Choose thresholds on a development split and freeze them before evaluating a held-out set. Report accepted-hit count, incorrect hits, useful safe-paraphrase coverage, bypass reasons, and latency together. The synthetic evaluation's threshold is not a recommendation for every domain.

Built-in checks are conservative risk reduction for supported English ASCII text. They are not a complete semantic-equivalence test. Entity extraction and relation handling have documented limits; application verifiers should enforce domain-specific rules.
