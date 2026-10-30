# LongCache roadmap

The current development package is **v0.3.0, unreleased**. Local implementation
and validation are recorded in [the execution record](docs/plans/v0.3.0-execution.md).
Release status is determined by published artifacts and release evidence, not this
planning summary. Python 3.11 and 3.12 remain the supported tested interpreters.

## 0.3 — Reliability

Implemented in this development line:

- Exact scoped sync/async singleflight for manual execution, decorators and OpenAI.
- Defined timeout, cancellation, failure, capacity, invalidation and shutdown behavior.
- Strict/default and opt-in bypass cache-error policies without unsafe replay.
- Owned SQLite/Redis/worker lifecycle and bounded private decision measurements.
- Reproducible reliability benchmarks and compatibility/artifact checks.

The supported topology remains one cache instance/process per local FAISS index.
Full controlled-host performance qualification, hosted CI and candidate feedback
remain release gates. Prometheus, OpenTelemetry, and new schema/Pydantic caching
are deferred. See [reliability](docs/reliability.md) and
[acceptance evidence](docs/plans/v0.3.0-acceptance.md).

## Later milestones

| Version | Direction |
| --- | --- |
| 0.4 | Defined multi-instance consistency, distributed coordination, and deployment/freshness evidence |
| 1.0 | Stable Python interfaces, support matrix, migrations and release provenance |
| 1.1 | Bounded conversational caching; streaming replay only after validation |
| 1.2 | Additional providers and multilingual coverage |
| 1.3 | Measured large-cache and index maintenance improvements |
| 1.4 | Optional proxy and one demand-driven external storage adapter |
| 2.0 | Opt-in adaptation validated against fixed-threshold baselines; justified breaking changes only |

Later milestones are provisional. Correctness and isolation fixes should ship in
the supported patch line without waiting for a future milestone.
