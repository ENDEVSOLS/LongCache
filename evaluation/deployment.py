"""Paired v0.3/v0.4 Redis deployment measurements with real worker processes.

The worker imports LongCache from the selected source tree after spawn, so the
baseline is actual archived v0.3 code rather than a label on current code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import platform
import resource
import sys
import time
from uuid import uuid4

import numpy as np


def _source_hash(root: Path) -> str:
    digest = sha256()
    for path in sorted((root / "src" / "longcache").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _summary(values: list[float], minimum: int) -> dict:
    return {
        "samples": len(values),
        "p50_ms": float(np.percentile(values, 50)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
        "p99_ms": float(np.percentile(values, 99)) if len(values) >= minimum else None,
        "p99_evidence_sufficient": len(values) >= minimum,
    }


def _worker(source_root, mode, cache_id, index_type, redis_url, commands, replies):
    sys.path.insert(0, str(Path(source_root) / "src"))
    from longcache import BaseEmbedder, CacheConfig, EmbeddingDescriptor, LongCache
    from longcache import RequestContext, RequestScope
    import longcache
    import redis

    class FixedEmbedder(BaseEmbedder):
        dimension = 384
        descriptor = EmbeddingDescriptor("deployment-fixture", "sha256-v1", 384)

        def embed(self, text):
            seed = int.from_bytes(sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            return vector / np.linalg.norm(vector)

    options = dict(
        backend="redis",
        redis_url=redis_url,
        cache_id=cache_id,
        generation="benchmark",
        faiss_index_type=index_type,
        ttl_seconds=0,
        max_entries=20_000,
    )
    if mode == "shared":
        options["deployment_mode"] = "shared"
    cache = LongCache(CacheConfig(**options), embedder=FixedEmbedder())
    context = RequestContext(
        scope=RequestScope("benchmark-tenant", "benchmark-user"),
        operation="deployment.answer",
        provider="fixture",
        provider_scope="fixture-v1",
        model="fixture-v1",
        prompt_version="v1",
    )
    client = redis.Redis.from_url(redis_url)
    try:
        replies.put({"ready": True, "source": longcache.__file__, "version": longcache.__version__})
        while True:
            action, payload = commands.get()
            try:
                if action == "seed":
                    started = time.perf_counter()
                    for number in range(payload):
                        decision = cache.set(f"question-{number:05d}", "answer", context=context)
                        if decision.outcome != "STORED":
                            raise AssertionError(
                                f"Seed rejected at row {number}: {decision.reason}"
                            )
                    replies.put({"seeded": payload, "elapsed_s": time.perf_counter() - started})
                elif action == "warm":
                    values = []
                    batch_started = time.perf_counter()
                    for _ in range(payload):
                        started = time.perf_counter()
                        decision = cache.lookup("question-00000", context=context)
                        values.append((time.perf_counter() - started) * 1000)
                        if decision.outcome != "HIT" or decision.response != "answer":
                            raise AssertionError(f"Unexpected warm decision: {decision.outcome}")
                    replies.put(
                        {"samples_ms": values, "elapsed_s": time.perf_counter() - batch_started}
                    )
                elif action == "burst":
                    count, delay_ms, signal = payload

                    def origin():
                        client.incr(signal)
                        time.sleep(delay_ms / 1000)
                        return "burst-answer"

                    with ThreadPoolExecutor(max_workers=count) as executor:
                        futures = [
                            executor.submit(
                                cache.get_or_compute, "burst-question", origin, context=context
                            )
                            for _ in range(count)
                        ]
                        answers = [future.result(timeout=30) for future in futures]
                    replies.put({"burst_answers": answers})
                elif action == "probe_set":
                    replies.put(
                        {
                            "probe_store": cache.set(
                                "probe", "fresh", context=context, tags=["probe"]
                            ).outcome
                        }
                    )
                elif action == "probe_get":
                    started = time.perf_counter()
                    value = cache.get("probe", context=context)
                    replies.put(
                        {"probe": value, "elapsed_ms": (time.perf_counter() - started) * 1000}
                    )
                elif action == "probe_invalidate":
                    replies.put({"invalidated": cache.invalidate_by_tag("probe", context=context)})
                elif action == "resources":
                    replies.put(
                        {
                            "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            "index_size": cache.index.size,
                        }
                    )
                elif action == "stop":
                    if payload:
                        cache.clear(operator=True)
                    replies.put({"stopped": True})
                    break
            except BaseException as exc:
                replies.put({"error": type(exc).__name__, "message": str(exc)[:200]})
                break
    finally:
        cache.close()


def _start_worker(source_root, mode, cache_id, index_type, redis_url):
    context = get_context("spawn")
    commands, replies = context.Queue(), context.Queue()
    process = context.Process(
        target=_worker,
        args=(source_root, mode, cache_id, index_type, redis_url, commands, replies),
    )
    process.start()
    ready = replies.get(timeout=30)
    if not ready.get("ready"):
        raise RuntimeError(f"Worker failed to start: {ready}")
    return process, commands, replies, ready


def _ask(worker, action, payload=None, timeout=90):
    _, commands, replies, _ = worker
    commands.put((action, payload))
    result = replies.get(timeout=timeout)
    if "error" in result:
        raise RuntimeError(f"Worker {action} failed: {result}")
    return result


def _measure(source_root, mode, index_type, entries, samples, config, redis_url):
    import redis

    cache_id = "deployment-" + uuid4().hex
    count = 2 if mode == "shared" else 1
    workers = []
    client = redis.Redis.from_url(redis_url)
    signal = f"benchmark:{cache_id}:origin"
    try:
        workers = [
            _start_worker(source_root, mode, cache_id, index_type, redis_url) for _ in range(count)
        ]
        if workers[0][3]["source"] != str(source_root / "src" / "longcache" / "__init__.py"):
            raise ValueError("Worker loaded the wrong source tree")
        if mode == "baseline" and workers[0][3]["version"] != "0.3.0":
            raise ValueError("Baseline worker did not import archived v0.3.0")
        seed = _ask(workers[0], "seed", entries, timeout=300)
        reader = workers[-1]
        warm = _ask(reader, "warm", samples, timeout=300)
        samples_ms = warm["samples_ms"]
        probe = {}
        if mode == "shared":
            probe["store"] = _ask(workers[0], "probe_set")["probe_store"]
            remote = _ask(reader, "probe_get")
            probe["remote_visible"] = remote["probe"] == "fresh"
            probe["visibility_ms"] = remote["elapsed_ms"]
            probe["invalidated_count"] = _ask(workers[0], "probe_invalidate")["invalidated"]
            invalidated = _ask(reader, "probe_get")
            probe["remote_missing_after_invalidation"] = invalidated["probe"] is None
            probe["invalidation_visibility_ms"] = invalidated["elapsed_ms"]
        per_worker = config["burst_concurrency"] // count
        for worker in workers:
            worker[1].put(("burst", (per_worker, config["origin_delay_ms"], signal)))
        bursts = []
        for worker in workers:
            result = worker[2].get(timeout=60)
            if "error" in result:
                raise RuntimeError(f"Burst worker failed: {result}")
            bursts.extend(result["burst_answers"])
        origin_calls = int(client.get(signal) or 0)
        resources = [_ask(worker, "resources") for worker in workers]
        return {
            "mode": mode,
            "source": workers[0][3],
            "summary": _summary(samples_ms, config["minimum_percentile_samples"]),
            "seed_entries_per_second": entries / seed["elapsed_s"],
            "warm_queries_per_second": samples / warm["elapsed_s"],
            "warm_samples_ms": samples_ms,
            "burst_origin_calls": origin_calls,
            "burst_answers_valid": bursts == ["burst-answer"] * config["burst_concurrency"],
            "probe": probe,
            "resources": resources,
        }
    finally:
        for index, worker in enumerate(workers):
            if worker[0].is_alive():
                try:
                    _ask(worker, "stop", index == 0, timeout=30)
                except Exception:
                    pass
                worker[0].join(timeout=5)
                if worker[0].is_alive():
                    worker[0].terminate()
                    worker[0].join(timeout=5)
        client.delete(signal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("evaluation/configs/v0_4_deployment.json")
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--reference-host-id")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--release", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    baseline = args.baseline_root.resolve()
    if not (baseline / "src" / "longcache" / "__init__.py").exists():
        parser.error("--baseline-root must contain actual archived v0.3 source")
    if args.release and not args.reference_host_id:
        parser.error("--release requires --reference-host-id")
    redis_url = os.environ.get("LONGCACHE_TEST_REDIS_URL")
    if not redis_url:
        parser.error("Set LONGCACHE_TEST_REDIS_URL to an isolated Redis authority")
    import redis

    client = redis.Redis.from_url(redis_url, socket_connect_timeout=2, socket_timeout=2)
    client.ping()
    config = json.loads(args.config.read_text())
    if config["protocol"] != "deployment-v1":
        parser.error("Unsupported deployment protocol")
    if args.smoke:
        config.update(cache_sizes=[1000], indexes=["flat"], repeats=1, warm_samples_per_repeat=100)
    rows = []
    for index_type in config["indexes"]:
        for entries in config["cache_sizes"]:
            for repeat in range(config["repeats"]):
                variants = [
                    ("baseline", baseline),
                    ("candidate_single", root),
                    ("shared", root),
                ]
                if repeat % 2:
                    variants.reverse()
                for variant, source in variants:
                    print(
                        f"{index_type} {entries} entries repeat {repeat + 1}: {variant}", flush=True
                    )
                    result = _measure(
                        source,
                        variant,
                        index_type,
                        entries,
                        config["warm_samples_per_repeat"],
                        config,
                        redis_url,
                    )
                    rows.append(
                        {"index": index_type, "entries": entries, "repeat": repeat, **result}
                    )
    gates = []
    for index_type in config["indexes"]:
        for entries in config["cache_sizes"]:
            grouped = {
                variant: [
                    row
                    for row in rows
                    if row["index"] == index_type
                    and row["entries"] == entries
                    and row["mode"] == variant
                ]
                for variant in ("baseline", "candidate_single", "shared")
            }
            summaries = {
                variant: _summary(
                    [sample for row in group for sample in row["warm_samples_ms"]],
                    config["minimum_percentile_samples"],
                )
                for variant, group in grouped.items()
            }
            budget = config["budgets"]
            gates.append(
                {
                    "index": index_type,
                    "entries": entries,
                    "baseline": summaries["baseline"],
                    "candidate_single": summaries["candidate_single"],
                    "shared": summaries["shared"],
                    "passed": (
                        all(summary["p99_evidence_sufficient"] for summary in summaries.values())
                        and summaries["candidate_single"]["p95_ms"]
                        <= budget["legacy_warm_p95_multiplier"] * summaries["baseline"]["p95_ms"]
                        + budget["legacy_warm_p95_add_ms"]
                        and summaries["candidate_single"]["p99_ms"]
                        <= budget["legacy_warm_p99_multiplier"] * summaries["baseline"]["p99_ms"]
                        + budget["legacy_warm_p99_add_ms"]
                        and summaries["shared"]["p95_ms"]
                        <= summaries["candidate_single"]["p95_ms"]
                        + budget["shared_warm_p95_add_ms"]
                        and summaries["shared"]["p99_ms"]
                        <= summaries["candidate_single"]["p99_ms"]
                        + budget["shared_warm_p99_add_ms"]
                        and all(
                            row["burst_origin_calls"] == budget["shared_origin_calls"]
                            and row["burst_answers_valid"]
                            and row["probe"]["remote_visible"]
                            and row["probe"]["remote_missing_after_invalidation"]
                            and row["probe"]["visibility_ms"] <= budget["shared_visibility_max_ms"]
                            and row["probe"]["invalidation_visibility_ms"]
                            <= budget["shared_visibility_max_ms"]
                            for row in grouped["shared"]
                        )
                    ),
                }
            )
    output = Path(config["output"] + ("-smoke.json" if args.smoke else ".json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": config["protocol"],
        "benchmark_gates_passed": bool(args.release and all(gate["passed"] for gate in gates)),
        "mode": "smoke" if args.smoke else "release",
        "reference_host_id": args.reference_host_id,
        "candidate_hash": _source_hash(root),
        "baseline_hash": _source_hash(baseline),
        "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "redis_version": client.info("server")["redis_version"],
            "redis_memory_bytes": client.info("memory")["used_memory"],
        },
        "config": config,
        "gates": gates,
        "rows": rows,
        "limits": "Synthetic deterministic embedder; same-host Redis. Smoke samples do not satisfy p99 gates. No live-provider or customer-data claim.",
    }
    output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "output": str(output),
                "benchmark_gates_passed": report["benchmark_gates_passed"],
                "gates": gates,
            },
            indent=2,
        )
    )
    if args.release and not report["benchmark_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
