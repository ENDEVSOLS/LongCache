"""One-command release evaluation. Fails nonzero on any configured quality gate."""

from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import numpy as np
from scipy.stats import beta
from longcache import (
    BaseEmbedder,
    CacheConfig,
    Instruction,
    LongCache,
    RequestContext,
    RequestScope,
)
from longcache.core.embedder import LocalEmbedder
from longcache.core.policy import verify
from .dataset import cases


def upper_bound(errors, count):
    if not count:
        return None
    return 1.0 if errors == count else float(beta.ppf(0.95, errors + 1, count - errors))


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def metrics(rows):
    hits = [r for r in rows if r["outcome"] == "HIT"]
    errors = sum(not r["reuse"] for r in hits)
    safe = [r for r in rows if r["group"] == "safe"]
    unsupported = [r for r in rows if r["group"] == "unsupported"]
    return {
        "requests": len(rows),
        "accepted_hits": len(hits),
        "incorrect_hits": errors,
        "hit_rate": len(hits) / len(rows) if rows else 0,
        "false_hit_rate": errors / len(hits) if hits else None,
        "false_hit_upper_95": upper_bound(errors, len(hits)),
        "safe_paraphrase_coverage": sum(r["outcome"] == "HIT" for r in safe) / len(safe)
        if safe
        else 0,
        "bypass_rate": sum(r["outcome"] == "BYPASS" for r in rows) / len(rows) if rows else 0,
        "unsupported_bypass_rate": sum(r["outcome"] == "BYPASS" for r in unsupported)
        / len(unsupported)
        if unsupported
        else 0,
        "isolation_errors": sum(
            r["outcome"] == "HIT" for r in rows if r["group"] in ("isolation", "freshness")
        ),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "latency_ms": {
            str(p): percentile([r["latency_ms"] for r in rows], p) for p in (50, 95, 99)
        },
    }


def check_gates(result, gates):
    failures = []
    for name, target in gates.items():
        key, comparison = name.rsplit("_", 1)
        value = result.get(key)
        if (
            value is None
            or (comparison == "max" and value > target)
            or (comparison == "min" and value < target)
        ):
            failures.append({"gate": name, "target": target, "actual": value})
    return failures


def base_context():
    return RequestContext(
        scope=RequestScope("synthetic-tenant", "synthetic-user"),
        operation="evaluation.answer",
        provider="fixture",
        provider_scope="fixture-v1",
        model="synthetic-model",
        prompt_version="v1",
        corpus_version="v1",
    )


def mutated(ctx, mutation):
    changes = {
        "tenant": {"scope": RequestScope("other-tenant", "synthetic-user")},
        "user": {"scope": RequestScope("synthetic-tenant", "other-user")},
        "model": {"model": "other-model"},
        "provider": {"provider": "other-provider"},
        "deployment": {"provider_scope": "other-deployment"},
        "instructions": {"instructions": (Instruction("system", "other policy"),)},
        "generation": {"generation": {"temperature": None}},
        "prompt": {"prompt_version": "v2"},
        "operation": {"operation": "other.answer"},
        "corpus": {"corpus_version": "v2"},
        "document": {"document_versions": {"document": "v2"}},
        "missing": {"scope": None},
    }
    return replace(ctx, **changes.get(mutation, {"request_shape": mutation}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    source, dataset, dataset_hash = cases(config["dataset"])
    import torch

    torch.set_num_threads(config["torch_threads"])
    model = LocalEmbedder(config["model"], config["revision"])
    texts = sorted({text for case in dataset for text in (case.seed, case.query)})
    print(f"Embedding {len(texts)} unique texts with pinned model revision", flush=True)
    start = time.perf_counter()
    vectors = model.embed_batch(texts)
    embedding_seconds = time.perf_counter() - start
    table = dict(zip(texts, vectors))

    class PreparedEmbedder(BaseEmbedder):
        dimension = model.dimension
        descriptor = model.descriptor

        def embed(self, text):
            return table[text].copy()

    cfg = CacheConfig(
        backend="memory",
        ttl_seconds=0,
        similarity_threshold=config["threshold"],
        policy_version=config["policy_version"],
    )
    ctx = base_context()
    rows = []
    verification = []
    with LongCache(cfg, embedder=PreparedEmbedder()) as cache:
        for i, case in enumerate(dataset):
            cache.clear(operator=True)
            stored = cache.set(case.seed, "synthetic answer", context=ctx)
            if stored.outcome != "STORED":
                raise RuntimeError("Evaluation seed was not admitted")
            query_context = (
                mutated(ctx, case.mutation)
                if case.group in ("isolation", "unsupported") or case.mutation == "corpus"
                else ctx
            )
            if case.mutation == "expired":
                cache.backend.retrieve(stored.entry_id).expires_at = 1
            elif case.mutation == "deleted":
                cache.backend.delete(stored.entry_id)
            elif case.mutation == "invalidated":
                cache.invalidate(context=ctx)
            elif case.mutation == "legacy":
                cache.backend.retrieve(stored.entry_id).metadata = {"format": 1}
            start = time.perf_counter()
            decision = cache.lookup(case.query, context=query_context)
            elapsed = (time.perf_counter() - start) * 1000
            rows.append(
                {
                    "id": case.id,
                    "family": case.family,
                    "group": case.group,
                    "reuse": case.reuse,
                    "outcome": decision.outcome,
                    "reason": decision.reason,
                    "latency_ms": elapsed,
                }
            )
            start = time.perf_counter()
            verify(case.query, case.seed, "synthetic answer", 1.0, ctx)
            verification.append((time.perf_counter() - start) * 1000)
            if i and i % 1000 == 0:
                print(f"Evaluated {i}/{len(dataset)} cases", flush=True)
    summary = metrics(rows)
    summary["verification_p95_ms"] = percentile(verification, 95)
    failures = check_gates(summary, config["gates"])
    report = {
        "protocol": config["protocol"],
        "source_tree_sha256": sha256(
            b"".join(
                str(p).encode() + p.read_bytes()
                for p in sorted(Path("src/longcache").rglob("*.py"))
            )
        ).hexdigest(),
        "dependencies": {
            name: version(name)
            for name in ("numpy", "faiss-cpu", "sentence-transformers", "torch", "scipy")
        },
        "dataset_version": source["version"],
        "dataset_sha256": dataset_hash,
        "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
        "model": asdict(model.descriptor),
        "threshold": config["threshold"],
        "policy_version": config["policy_version"],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_threads": config["torch_threads"],
        },
        "embedding_precompute_seconds": embedding_seconds,
        "unique_texts": len(texts),
        "independent_safe_families": len({r["family"] for r in rows if r["group"] == "safe"}),
        "metrics": summary,
        "gates": config["gates"],
        "failed_gates": failures,
        "passed": not failures,
        "limitations": [
            "Synthetic template-expanded cases; expanded variants are correlated, not independent semantic evidence.",
            "Scenario latency uses precomputed pinned-model vectors and excludes model inference; end-to-end performance is reported separately.",
            "One-sided 95% Clopper-Pearson bounds apply to the sample under an independence assumption; family counts disclose correlation.",
            "Human review of domain representativeness and hosted CI remain release requirements.",
        ],
        "family_metrics": {
            family: metrics([r for r in rows if r["family"] == family])
            for family in sorted({r["family"] for r in rows})
        },
        "cases": rows,
    }
    if args.baseline_root:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "cases.json").write_text(json.dumps([asdict(c) for c in dataset]))
            (base / "texts.json").write_text(json.dumps(texts))
            np.savez(base / "vectors.npz", vectors=vectors)
            env = os.environ | {"PYTHONPATH": str(args.baseline_root.resolve() / "src")}
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("legacy_baseline.py").resolve()),
                    str(base / "cases.json"),
                    str(base / "vectors.npz"),
                    str(base / "texts.json"),
                    str(config["threshold"]),
                ],
                cwd=directory,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            baseline = json.loads(result.stdout)
            by_id = {r["id"]: r for r in rows}
            converted = [
                dict(
                    by_id[r["id"]],
                    outcome="HIT" if r["hit"] else "MISS",
                    reason="v011_similarity",
                    latency_ms=r["latency_ms"],
                )
                for r in baseline
            ]
            report["baseline_semantic_metrics"] = metrics(converted)
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# v0.2 synthetic release evaluation",
        "",
        f"Protocol: `{config['protocol']}`. Passed configured gates: **{not failures}**.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Limits",
        "",
    ] + [f"- {item}" for item in report["limitations"]]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"metrics": summary, "failed_gates": failures, "output": str(output)}, indent=2)
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
