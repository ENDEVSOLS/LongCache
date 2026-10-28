# v0.2 synthetic release evaluation

Protocol: `safe-reuse-v1`. Passed configured gates: **True**.

```json
{
  "requests": 4800,
  "accepted_hits": 2000,
  "incorrect_hits": 0,
  "hit_rate": 0.4166666666666667,
  "false_hit_rate": 0.0,
  "false_hit_upper_95": 0.0014967448951882837,
  "safe_paraphrase_coverage": 1.0,
  "bypass_rate": 0.041666666666666664,
  "unsupported_bypass_rate": 1.0,
  "isolation_errors": 0,
  "reasons": {
    "accepted": 2000,
    "negation_mismatch": 500,
    "number_mismatch": 500,
    "no_candidate": 1080,
    "relation_mismatch": 500,
    "missing_identity": 40,
    "unsupported_request_shape": 160,
    "legacy_format": 20
  },
  "latency_ms": {
    "50": 0.1893345015560044,
    "95": 0.26198005252808804,
    "99": 0.3424815019752714
  },
  "verification_p95_ms": 0.04755099780595629
}
```

## Limits

- Synthetic template-expanded cases; expanded variants are correlated, not independent semantic evidence.
- Scenario latency uses precomputed pinned-model vectors and excludes model inference; end-to-end performance is reported separately.
- One-sided 95% Clopper-Pearson bounds apply to the sample under an independence assumption; family counts disclose correlation.
- Human review of domain representativeness and hosted CI remain release requirements.
