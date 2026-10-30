"""Paired reliability benchmarks. Smoke output is explicitly not release evidence.

Run with --baseline-root containing an actual v0.2 checkout. Worker processes use
that source via PYTHONPATH, and all measurements exclude process IPC and setup.
--real-model adds actual pinned-snapshot inference to warm-hit measurements.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time

import numpy as np


def summarize(samples, elapsed=None, errors=0, minimum=10000):
    return {
        "samples": len(samples),
        "errors": errors,
        "p50_ms": float(np.percentile(samples, 50)) if samples else None,
        "p95_ms": float(np.percentile(samples, 95)) if samples else None,
        "p99_ms": float(np.percentile(samples, 99)) if len(samples) >= minimum else None,
        "percentile_evidence_sufficient": len(samples) >= minimum,
        "throughput_per_second": len(samples) / elapsed if elapsed else None,
    }


def compare(current, baseline, budgets):
    if not current["samples"] or not baseline["samples"]:
        raise ValueError("A matched nonempty benchmark is required")
    p95 = budgets["warm_p95_multiplier"] * baseline["p95_ms"] + budgets["warm_p95_add_ms"]
    p99 = (
        None
        if baseline["p99_ms"] is None
        else budgets["warm_p99_multiplier"] * baseline["p99_ms"] + budgets["warm_p99_add_ms"]
    )
    return {
        "p95_budget_ms": p95,
        "p95_passed": current["p95_ms"] <= p95,
        "p99_budget_ms": p99,
        "p99_passed": current["p99_ms"] <= p99
        if p99 is not None and current["p99_ms"] is not None
        else None,
    }


def source_hash(root):
    return sha256(
        b"".join(
            str(p.relative_to(root)).encode() + p.read_bytes() for p in sorted(root.rglob("*.py"))
        )
    ).hexdigest()


def worker(
    real_model=False,
    *,
    model="BAAI/bge-small-en-v1.5",
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
):
    import faiss
    import longcache
    from longcache.core.entry import CacheEntry
    from longcache.core.identity import digest

    faiss.omp_set_num_threads(1)
    query = "How do I reset my password?"
    underlying = None
    if real_model:
        import torch
        from huggingface_hub import snapshot_download
        from longcache.core.embedder import LocalEmbedder

        torch.set_num_threads(1)
        underlying = LocalEmbedder(
            snapshot_download(
                model,
                revision=revision,
                local_files_only=True,
            )
        )

    class Embedder(longcache.BaseEmbedder):
        dimension = 384
        descriptor = longcache.EmbeddingDescriptor(
            model if real_model else "benchmark-hash",
            revision if real_model else "sha256-v1",
            384,
        )

        def embed(self, text):
            if underlying is not None and text == query:
                return underlying.embed(text)
            rng = np.random.default_rng(int.from_bytes(sha256(text.encode()).digest()[:8], "big"))
            vector = rng.normal(size=384).astype(np.float32)
            return vector / np.linalg.norm(vector)

    context = longcache.RequestContext(
        scope=longcache.RequestScope("benchmark", "user"),
        operation="benchmark.answer",
        provider="fixture",
        provider_scope="fixture-v1",
        model="fixture",
        prompt_version="v1",
    )
    cache = None
    embedder = Embedder()
    candidate = hasattr(longcache.LongCache, "get_or_compute")
    identity = {
        "version": longcache.__version__,
        "source": longcache.__file__,
        "source_hash": source_hash(Path(longcache.__file__).parent),
        "has_singleflight": candidate,
    }
    from importlib.metadata import version as package_version
    from dataclasses import asdict

    identity["dependencies"] = {
        name: package_version(name)
        for name in ("numpy", "faiss-cpu", "torch", "sentence-transformers")
    }
    identity["embedding"] = asdict(embedder.descriptor)
    print(json.dumps(identity), flush=True)

    def invoke(text, compute):
        if candidate:
            return cache.get_or_compute(text, compute, context=context)
        decision = cache.lookup(text, context=context)
        if decision.outcome == "HIT":
            return decision.response
        result = compute()
        cache.set(text, result, context=context)
        return result

    async def ainvoke(text, compute):
        if candidate:
            return await cache.aget_or_compute(text, compute, context=context)
        decision = await cache.alookup(text, context=context)
        if decision.outcome == "HIT":
            return decision.response
        result = await compute()
        await cache.aset(text, result, context=context)
        return result

    try:
        for line in sys.stdin:
            command = json.loads(line)
            action = command["action"]
            if action == "prepare":
                if cache is not None:
                    cache.close()
                config = longcache.CacheConfig(
                    backend="memory",
                    ttl_seconds=0,
                    faiss_index_type=command["index"],
                    max_entries=max(20000, command["entries"]),
                    track_stats=command.get("tracking", True),
                    similarity_threshold=command.get("threshold", 0.92),
                )
                cache = longcache.LongCache(config, embedder=embedder)
                prepared, reason = cache._prepare(query, context, None, {})
                assert reason is None
                entries = []
                # Setup uses the documented v0.2 format, bypassing O(N^2) count/evict
                # setup work. All timed operations cross the public cache interface.
                for n in range(command["entries"]):
                    text = query if n == 0 else f"noise {n}"
                    entry = CacheEntry(
                        namespace=prepared[1],
                        query=text,
                        response="answer",
                        vector=embedder.embed(text),
                        metadata={
                            "format": 2,
                            "identity": prepared[2],
                            "embedding": cache._fingerprint,
                            "payload": digest("answer"),
                        },
                    )
                    entry.id = cache.backend.store(entry)
                    entries.append(entry)
                cache.index.rebuild(entries)
                assert invoke(query, lambda: "unexpected") == "answer"
                result = {"prepared": True}
            elif action == "warm":
                samples = []
                for _ in range(command["count"]):
                    started = time.perf_counter()
                    assert invoke(query, lambda: "unexpected") == "answer"
                    samples.append((time.perf_counter() - started) * 1000)
                result = {"samples_ms": samples}
            elif action == "tracking":
                cache.config.track_stats = command["enabled"]
                if candidate:
                    cache._observability.enabled = command["enabled"]
                result = {"tracking": command["enabled"]}
            elif action in ("miss", "burst", "scheduled_miss"):
                cache.clear(operator=True)
                samples, errors, calls = [], [], []
                counter_lock = threading.Lock()
                delay = command["delay_ms"] / 1000
                concurrency, count = command["concurrency"], command["count"]
                mode, nonce = command["mode"], command["nonce"]
                interval = delay / concurrency if action == "scheduled_miss" else 0
                wave_started = time.perf_counter()
                queue_waits = []

                def text(n):
                    return "policy" if action == "burst" else f"request {nonce} number {n}"

                def request(n):
                    def compute():
                        with counter_lock:
                            calls.append(1)
                        time.sleep(delay)
                        return "answer"

                    scheduled = wave_started + n * interval
                    if interval:
                        time.sleep(max(0, scheduled - time.perf_counter()))
                    actual = time.perf_counter()
                    queue_waits.append(max(0, actual - scheduled) * 1000 if interval else 0)
                    started = scheduled if interval else actual
                    try:
                        assert invoke(text(n), compute) == "answer"
                        samples.append((time.perf_counter() - started) * 1000)
                    except Exception as exc:
                        errors.append(type(exc).__name__)

                async def arequests():
                    semaphore = asyncio.Semaphore(concurrency)

                    async def request(n):
                        scheduled = wave_started + n * interval
                        if interval:
                            await asyncio.sleep(max(0, scheduled - time.perf_counter()))
                        async with semaphore:

                            async def compute():
                                calls.append(1)
                                await asyncio.sleep(delay)
                                return "answer"

                            actual = time.perf_counter()
                            queue_waits.append(max(0, actual - scheduled) * 1000 if interval else 0)
                            started = scheduled if interval else actual
                            try:
                                assert await ainvoke(text(n), compute) == "answer"
                                samples.append((time.perf_counter() - started) * 1000)
                            except Exception as exc:
                                errors.append(type(exc).__name__)

                    await asyncio.gather(*(request(n) for n in range(count)))

                started = time.perf_counter()
                if mode == "async":
                    asyncio.run(arequests())
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        list(pool.map(request, range(count)))
                result = {
                    "samples_ms": samples,
                    "elapsed_seconds": time.perf_counter() - started,
                    "errors": errors,
                    "origin_calls": len(calls),
                    "queue_wait_ms": queue_waits,
                    "scheduled_arrival_interval_ms": interval * 1000,
                }
            elif action == "resources":
                result = {
                    "rss_peak_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "cpu_user_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                    "cpu_system_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                    "threads": threading.active_count(),
                    "metrics": cache.metrics(operator=True) if candidate else None,
                }
            else:
                raise ValueError("Unknown benchmark action")
            print(json.dumps(result), flush=True)
    finally:
        if cache is not None:
            cache.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("evaluation/configs/v0_3_reliability.json")
    )
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--quick-concurrency",
        action="store_true",
        help="Qualify warm percentiles but keep concurrency runs preliminary",
    )
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument(
        "--reference-host-id", help="Identifier of the predeclared controlled measurement host"
    )
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        settings = json.loads(args.config.read_text())
        worker(args.real_model, model=settings["model"], revision=settings["revision"])
        return 0
    if args.baseline_root is None:
        parser.error("--baseline-root must point to the v0.2 source checkout")
    if (args.smoke or args.quick_concurrency) and args.release:
        parser.error("Preliminary modes cannot supply --release evidence")
    if args.release and not args.reference_host_id:
        parser.error("--release requires a predeclared --reference-host-id")
    config = json.loads(args.config.read_text())
    if args.smoke:
        config = config | {
            "cache_sizes": [100],
            "repeats": 1,
            "warm_samples_per_repeat": 50,
            "miss_requests_per_repeat": 20,
            "concurrencies": [20],
            "provider_delay_ms": 100,
            "burst_repeats": 1,
        }
    elif args.quick_concurrency:
        config = config | {
            "miss_requests_per_repeat": 20,
            "concurrencies": [20],
            "burst_repeats": 5,
        }
    root = Path(__file__).resolve().parents[1]
    processes, identities, cells, failures = [], [], [], []
    raw, overhead = [], []

    def reply(process, command=None):
        if command is not None:
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Benchmark worker exited before reporting")
        return json.loads(line)

    try:
        for source in (root, args.baseline_root.resolve()):
            environment = os.environ | {
                "PYTHONPATH": os.pathsep.join((str(source / "src"), str(root))),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            command = [
                sys.executable,
                "-u",
                "-m",
                "evaluation.reliability",
                "--worker",
                "--config",
                str(args.config.resolve()),
            ]
            if args.real_model:
                command.append("--real-model")
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            identities.append(reply(process))
        if not identities[0]["has_singleflight"] or identities[1]["version"] != "0.2.0":
            raise ValueError("Expected candidate shared execution and actual v0.2 baseline")
        for index in config["indexes"]:
            for size in config["cache_sizes"]:
                for process in processes:
                    reply(
                        process,
                        {
                            "action": "prepare",
                            "index": index,
                            "entries": size,
                            "threshold": config["threshold"],
                        },
                    )
                    reply(process, {"action": "warm", "count": config["warmup"]})
                rows = [[], []]
                repeats = [[], []]
                for repetition in range(config["repeats"]):
                    for which in (0, 1) if repetition % 2 == 0 else (1, 0):
                        sample = reply(
                            processes[which],
                            {"action": "warm", "count": config["warm_samples_per_repeat"]},
                        )["samples_ms"]
                        rows[which].extend(sample)
                        repeats[which].append(
                            summarize(sample, minimum=config["minimum_percentile_samples"])
                        )
                summaries = [
                    summarize(r, minimum=config["minimum_percentile_samples"]) for r in rows
                ]
                comparison = compare(*summaries, config["budgets"])
                if not comparison["p95_passed"] or comparison["p99_passed"] is False:
                    failures.append(f"warm-{index}-{size}")
                cells.append(
                    {
                        "workload": "warm",
                        "index": index,
                        "entries": size,
                        "candidate": summaries[0],
                        "baseline": summaries[1],
                        "per_repeat": repeats,
                        "comparison": comparison,
                    }
                )
                raw.append(
                    {
                        "index": index,
                        "entries": size,
                        "candidate_ms": rows[0],
                        "baseline_ms": rows[1],
                    }
                )
                # Balanced enabled/disabled pairs on the same candidate worker.
                tracking_samples = {True: [], False: []}
                for repetition in range(config["repeats"]):
                    for enabled in (True, False) if repetition % 2 == 0 else (False, True):
                        reply(processes[0], {"action": "tracking", "enabled": enabled})
                        sample = reply(
                            processes[0],
                            {"action": "warm", "count": config["warm_samples_per_repeat"]},
                        )["samples_ms"]
                        tracking_samples[enabled].extend(sample)
                reply(processes[0], {"action": "tracking", "enabled": True})
                enabled = summarize(
                    tracking_samples[True], minimum=config["minimum_percentile_samples"]
                )
                disabled = summarize(
                    tracking_samples[False], minimum=config["minimum_percentile_samples"]
                )
                allowed = (
                    config["budgets"]["measurement_p95_multiplier"] * disabled["p95_ms"]
                    + config["budgets"]["measurement_p95_add_ms"]
                )
                overhead.append(
                    {
                        "index": index,
                        "entries": size,
                        "enabled": enabled,
                        "disabled": disabled,
                        "allowed_p95_ms": allowed,
                        "passed": enabled["p95_ms"] <= allowed,
                    }
                )
                if enabled["p95_ms"] > allowed:
                    failures.append(f"measurement-{index}-{size}")
        # Concurrency and origin counts use isolated cold runs after warm comparisons.
        for mode in ("sync", "async"):
            for action in ("miss", "burst", "scheduled_miss"):
                for concurrency in [20] if action != "miss" else config["concurrencies"]:
                    runs = [[], []]
                    for repetition in range(
                        config["burst_repeats"] if action == "burst" else config["repeats"]
                    ):
                        for which in (0, 1) if repetition % 2 == 0 else (1, 0):
                            result = reply(
                                processes[which],
                                {
                                    "action": action,
                                    "mode": mode,
                                    "concurrency": concurrency,
                                    "count": concurrency
                                    if action == "burst"
                                    else config["miss_requests_per_repeat"],
                                    "delay_ms": config["provider_delay_ms"],
                                    "nonce": f"{config['seed']}-{repetition}",
                                },
                            )
                            runs[which].append(result)
                    summary = []
                    for run in runs:
                        summary.append(
                            summarize(
                                [x for r in run for x in r["samples_ms"]],
                                elapsed=sum(r["elapsed_seconds"] for r in run),
                                errors=sum(len(r["errors"]) for r in run),
                                minimum=config["minimum_percentile_samples"],
                            )
                            | {
                                "origin_calls": [r["origin_calls"] for r in run],
                                "error_types": sorted({e for r in run for e in r["errors"]}),
                                "queue_wait_p95_ms": float(
                                    np.percentile([q for r in run for q in r["queue_wait_ms"]], 95)
                                ),
                            }
                        )
                    if action == "burst" and (
                        any(n != 1 for n in summary[0]["origin_calls"]) or summary[0]["errors"]
                    ):
                        failures.append(f"burst-{mode}")
                    overhead_ms = (
                        summary[0]["p95_ms"] - config["provider_delay_ms"]
                        if action == "burst"
                        else None
                    )
                    if (
                        action == "burst"
                        and summary[0]["percentile_evidence_sufficient"]
                        and overhead_ms > config["budgets"]["singleflight_added_p95_ms_max"]
                    ):
                        failures.append(f"singleflight-overhead-{mode}")
                    if (
                        action == "miss"
                        and concurrency <= 20
                        and (
                            summary[0]["errors"] != summary[1]["errors"]
                            or summary[0]["throughput_per_second"]
                            < config["budgets"]["distinct_miss_throughput_ratio_min"]
                            * summary[1]["throughput_per_second"]
                        )
                    ):
                        failures.append(f"throughput-{mode}-{concurrency}")
                    cells.append(
                        {
                            "workload": action,
                            "mode": mode,
                            "concurrency": concurrency,
                            "candidate": summary[0],
                            "baseline": summary[1],
                            "singleflight_added_p95_ms": overhead_ms,
                            "orchestration_budget_qualified": action == "burst"
                            and summary[0]["percentile_evidence_sufficient"],
                        }
                    )
        sufficient = all(
            c["candidate"]["percentile_evidence_sufficient"]
            and c["baseline"]["percentile_evidence_sufficient"]
            for c in cells
            if c["workload"] in ("warm", "burst")
            or (c["workload"] in ("miss", "scheduled_miss") and c["concurrency"] <= 20)
        ) and all(
            h["enabled"]["percentile_evidence_sufficient"]
            and h["disabled"]["percentile_evidence_sufficient"]
            for h in overhead
        )
        report = {
            "protocol": config["protocol"],
            "smoke": args.smoke,
            "quick_concurrency": args.quick_concurrency,
            "real_model": args.real_model,
            "config": config,
            "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
            "lock_sha256": sha256((root / "uv.lock").read_bytes()).hexdigest(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "cpu": platform.processor(),
                "faiss_threads": 1,
            },
            "workers": identities,
            "cells": cells,
            "raw_warm_samples": raw,
            "measurement_overhead": overhead,
            "resources": [reply(p, {"action": "resources"}) for p in processes],
            "failures": failures,
            "percentile_evidence_sufficient": sufficient,
            "reference_host_id": args.reference_host_id,
            "release_evidence_complete": sufficient
            and not args.smoke
            and not failures
            and bool(args.reference_host_id),
            "limits": "Memory backend timing; deterministic embedder unless real_model=true. Closed-loop and scheduled-arrival workloads; queue delay is included in scheduled latency. Balanced enabled/disabled measurement comparisons. Low-sample p99 is omitted; burst/overload cells are count evidence. Redis and fault semantics are covered by the acceptance suite. No production latency or cost claim.",
        }
        destination = Path(
            config["output"]
            + ("-smoke" if args.smoke else "-local" if args.quick_concurrency else "")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [
            "# v0.3 reliability measurements",
            "",
            f"Smoke: {args.smoke}; real model: {args.real_model}.",
            f"Release evidence complete: {report['release_evidence_complete']}.",
            "",
            "| Workload | Candidate p95 ms | Baseline p95 ms | Candidate errors |",
            "| --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| {c['workload']} {c.get('index', c.get('mode'))} {c.get('entries', c.get('concurrency'))} | {c['candidate']['p95_ms']} | {c['baseline']['p95_ms']} | {c['candidate']['errors']} |"
            for c in cells
        )
        lines += ["", report["limits"], "", f"Failed budgets: {failures}"]
        destination.with_suffix(".md").write_text("\n".join(lines) + "\n")
        print(
            json.dumps(
                {
                    "report": str(destination),
                    "failures": failures,
                    "release_evidence_complete": report["release_evidence_complete"],
                }
            )
        )
        return int(bool(failures) or (args.release and not report["release_evidence_complete"]))
    finally:
        for process in processes:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
