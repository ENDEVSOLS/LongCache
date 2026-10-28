# Evaluation and benchmarks

## Recorded local results

The validation receipt is `validation/v0.2.0.json`; detailed JSON and pytest reports are in repository `evaluation/reports/`.

| Check | Recorded result |
| --- | --- |
| Full suite on Python 3.11 and 3.12 | 372 passed per interpreter; zero failures, errors, or skips |
| Frozen synthetic correctness protocol | 4,800 cases; 2,000 accepted hits; zero incorrect hits or isolation errors |
| Safe-paraphrase coverage / unsupported bypass | 100% / 100% on the declared fixtures |
| Changing-document RAG with real embeddings | All six backend/index combinations passed |
| Paired warm-hit latency | All four configured comparisons passed |

These are local results. Hosted CI after manual push, customer-domain label/representativeness review, and release-candidate feedback remain external release gates. Test results do not establish live-provider answer quality.

## Reproduce the evaluations

Run from the repository root after installing the locked environment:

```bash
uv sync --locked --extra all --extra dev --extra docs
export LONGCACHE_TEST_REDIS_URL=redis://localhost:16379/0
LONGCACHE_TEST_MODEL=1 uv run --no-sync pytest -q -rs
uv run --no-sync python -m evaluation.run --config evaluation/configs/v0_2_release.json
uv run --no-sync python -m evaluation.rag
```

A disposable Redis service is required for the full suite and RAG workflow. Model weights must be downloaded once or already cached. The evaluation model revision is `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` of `BAAI/bge-small-en-v1.5`. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` when that snapshot is cached and an offline run is required. A skipped Redis/model test is not evidence that it passed.

## Synthetic quality protocol

The frozen 4,800-case suite checks safe paraphrases, adversarial near misses, scope/context isolation, unsupported requests, and freshness. Dataset definitions and predeclared gates are versioned under `evaluation/`. The command exits nonzero if a configured quality gate fails.

The recorded run accepted 2,000 safe hits with no incorrect hits or isolation errors and bypassed all declared unsupported cases. Expanded variants are correlated across 35 held-out safe-operation families. These results are not a representative production error estimate. The report records family counts, model/dependency versions, dataset/config/source hashes, and limitations. Scenario timing excludes model inference through precomputed real-model vectors. Outputs are `evaluation/reports/v0_2_release.json` and `.md`.

## Warm-hit end-to-end comparison

Prepare the actual v0.1.1 baseline from repository history, then run the comparison:

```bash
mkdir -p /tmp/longcache-v011
git archive 8a47ebd | tar -x -C /tmp/longcache-v011
uv run --no-sync python -m evaluation.performance --baseline-root /tmp/longcache-v011
```

Use a fresh baseline directory and a checkout containing that commit. To include the v0.1.1 semantic comparison in the quality report, append `--baseline-root /tmp/longcache-v011` to the `evaluation.run` command. Run timed benchmarks without concurrent model tests or evaluations.

The `paired-warm-hit-v2` protocol includes actual model inference, uses one Torch/FAISS thread per worker, and records 50 alternating paired samples per scenario. Separate old/new workers read the same immutable model snapshot; IPC time is excluded. The unchanged budget is `v0.2 p95 <= 1.2 × v0.1.1 p95 + 5 ms`.

| Index | Entries | Recorded v0.2 p95 | Allowed p95 | Result |
| --- | --- | --- | --- | --- |
| Flat | 1,000 | 32.61 ms | 44.09 ms | Passed |
| Flat | 10,000 | 32.29 ms | 42.33 ms | Passed |
| HNSW | 1,000 | 29.86 ms | 40.62 ms | Passed |
| HNSW | 10,000 | 30.49 ms | 39.59 ms | Passed |

These values come from `evaluation/reports/warm_lookup.json` on the recorded hardware; they are not a general latency guarantee. Original failed diagnostic runs remain archived and are explained in the [execution record](plans/v0.2.0-execution.md).

Synthetic random distractors isolate index-size overhead. Miss latency, concurrent maintenance, distributed operation, and cost savings are not established by this benchmark.

## Changing-document RAG

`evaluation.rag` uses real pinned-model embeddings, authored document revisions, and an extractive fixture generator. All memory/SQLite/Redis and flat/HNSW combinations passed repeated-hit, changed-document, tenant-isolation, and scoped-invalidation assertions. The report is `evaluation/reports/versioned_rag.json`. Controlled in-flight document changes are covered separately by the application integration tests.

See the [RAG workflow guide](integrations/versioned-rag.md) and [acceptance evidence map](plans/v0.2.0-acceptance.md). This is executable application wiring and freshness evidence, not customer dataset review or a live LLM accuracy evaluation.
