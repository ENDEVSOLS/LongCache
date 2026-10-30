"""Paired 1K/10K warm-hit comparison, including real pinned-model inference."""

from __future__ import annotations
import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import longcache
from longcache.core.embedder import LocalEmbedder

QUERY = "How do I reset my password?"
MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


def worker(legacy):
    import torch
    import faiss
    from huggingface_hub import snapshot_download

    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    # Both implementations read exactly the same immutable local snapshot.
    snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
    model = LocalEmbedder(snapshot)
    model.embed(QUERY)

    class Hybrid(getattr(longcache, "BaseEmbedder", object)):
        dimension = 384
        inference_ms = 0.0
        if not legacy:
            descriptor = longcache.EmbeddingDescriptor(MODEL, REVISION, 384)

        def embed(self, text):
            if text == QUERY:
                start = time.perf_counter()
                value = model.embed(text)
                self.inference_ms = (time.perf_counter() - start) * 1000
                return value
            rng = np.random.default_rng(int.from_bytes(sha256(text.encode()).digest()[:8], "big"))
            value = rng.normal(size=384).astype(np.float32)
            return value / np.linalg.norm(value)

    embedder = Hybrid()
    cache = None
    options = {}
    print(
        json.dumps({"ready": True, "version": longcache.__version__, "source": longcache.__file__}),
        flush=True,
    )
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["operation"] == "prepare":
                if cache is not None:
                    cache.close()
                cfg = longcache.CacheConfig(
                    backend="memory",
                    ttl_seconds=0,
                    max_entries=20000,
                    faiss_index_type=request["index"],
                )
                if legacy:
                    cache = longcache.LongCache(cfg)
                    cache.embedder = embedder
                else:
                    ctx = longcache.RequestContext(
                        scope=longcache.RequestScope("benchmark", "user"),
                        operation="benchmark.hit",
                        provider="fixture",
                        provider_scope="benchmark-v1",
                        model="fixture",
                        prompt_version="v1",
                    )
                    cache = longcache.LongCache(cfg, embedder=embedder)
                    options = {"context": ctx}
                cache.set(QUERY, "answer", **options)
                for i in range(request["entries"] - 1):
                    cache.set(f"noise {i}", "distractor", **options)
                gc.collect()  # Setup garbage is excluded; GC stays enabled during samples.
                print(json.dumps({"prepared": True}), flush=True)
            elif request["operation"] == "lookup":
                start = time.perf_counter()
                response = cache.get(QUERY, **options)
                elapsed = (time.perf_counter() - start) * 1000
                assert response == "answer"
                print(
                    json.dumps({"latency_ms": elapsed, "embedding_ms": embedder.inference_ms}),
                    flush=True,
                )
            else:
                raise ValueError("Unknown benchmark operation")
    finally:
        if cache is not None:
            cache.close()


def summarize(rows):
    return {
        "samples": len(rows),
        **{
            f"p{p}_ms": float(np.percentile([r["latency_ms"] for r in rows], p))
            for p in (50, 95, 99)
        },
        "embedding_p95_ms": float(np.percentile([r["embedding_ms"] for r in rows], 95)),
        "non_embedding_p95_ms": float(
            np.percentile([r["latency_ms"] - r["embedding_ms"] for r in rows], 95)
        ),
    }


def comparisons(current, previous):
    if not current or len(current) != len(previous):
        raise ValueError("A matched nonempty benchmark is required")
    result = []
    for a, b in zip(current, previous):
        assert (a["index"], a["entries"]) == (b["index"], b["entries"])
        budget = 1.2 * b["p95_ms"] + 5
        result.append(
            {
                "entries": a["entries"],
                "index": a["index"],
                "budget_p95_ms": budget,
                "actual_p95_ms": a["p95_ms"],
                "passed": a["p95_ms"] <= budget,
            }
        )
    return result


def paired_samples(processes, reply, count):
    """Alternate worker order so elapsed host drift does not favor one version."""
    rows = [[], []]
    for i in range(count):
        for j in (0, 1) if i % 2 == 0 else (1, 0):
            rows[j].append(reply(processes[j], {"operation": "lookup"}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/reports/warm_lookup_current.json")
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.legacy)
        return 0
    if args.baseline_root is None:
        parser.error("--baseline-root is required for the comparison gate")
    script = str(Path(__file__).resolve())
    environments = [
        os.environ.copy(),
        os.environ | {"PYTHONPATH": str(args.baseline_root.resolve() / "src")},
    ]
    processes = []

    def reply(process, command=None):
        if command is not None:
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Benchmark worker exited before completing its sample")
        return json.loads(line)

    try:
        identities = []
        for index, environment in enumerate(environments):
            command = [sys.executable, "-u", script, "--worker"] + (["--legacy"] if index else [])
            process = subprocess.Popen(
                command,
                cwd=args.baseline_root if index else Path.cwd(),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            identities.append(reply(process))
        assert (
            identities[0]["version"] == longcache.__version__
            and identities[1]["version"] == "0.1.1"
        )
        current, previous = [], []
        samples = []
        for kind in ("flat", "hnsw"):
            for size in (1000, 10000):
                for process in processes:
                    reply(process, {"operation": "prepare", "index": kind, "entries": size})
                # Alternate execution order, keeping both models warm before timing.
                for _ in range(5):
                    for process in processes:
                        reply(process, {"operation": "lookup"})
                rows = paired_samples(processes, reply, 50)
                current.append({"index": kind, "entries": size, **summarize(rows[0])})
                previous.append({"index": kind, "entries": size, **summarize(rows[1])})
                samples.append(
                    {"index": kind, "entries": size, "candidate": rows[0], "v011": rows[1]}
                )
        gates = comparisons(current, previous)
        report = {
            "protocol": "paired-warm-hit-v2",
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "torch_threads_per_worker": 1,
                "faiss_threads_per_worker": 1,
            },
            "model_revision": REVISION,
            "source_tree_sha256": sha256(
                b"".join(
                    str(p).encode() + p.read_bytes()
                    for p in sorted(Path("src/longcache").rglob("*.py"))
                )
            ).hexdigest(),
            "worker_identities": identities,
            "candidate": current,
            "v011": previous,
            "comparisons": gates,
            "comparison_passed": all(r["passed"] for r in gates),
            "samples": samples,
            "limits": "Synthetic distractors; real pinned-snapshot inference. Samples alternate worker/version order to reduce time drift. IPC time is excluded. Setup collection precedes warm-up; GC remains enabled during measurement. No distributed or concurrent-maintenance throughput claim.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
        return int(not report["comparison_passed"])
    finally:
        for process in processes:
            process.stdin.close()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
