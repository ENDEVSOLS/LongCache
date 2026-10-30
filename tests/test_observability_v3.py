"""Measurements conserve requests, omit secrets, and remain bounded."""

import asyncio
from dataclasses import replace
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from longcache import CacheConfig, LongCache, PolicyDecision
from tests.helpers import HashEmbedder, trusted_context
from tests.test_singleflight_v3 import joined_event


def make_cache(**kwargs):
    return LongCache(
        CacheConfig(backend="memory", ttl_seconds=0, **kwargs),
        embedder=HashEmbedder(),
        context=trusted_context(),
    )


def count(snapshot, name, **labels):
    return sum(
        row["value"]
        for row in snapshot["counters"]
        if row["name"] == name and all(row["labels"].get(k) == v for k, v in labels.items())
    )


def test_internal_rechecks_do_not_inflate_external_requests(monkeypatch):
    with make_cache() as cache:
        ready = joined_event(cache, monkeypatch, 19)

        def origin():
            assert ready.wait(5)
            return "answer"

        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(cache.get_or_compute, "policy", origin) for _ in range(20)]
            assert all(f.result(timeout=10) == "answer" for f in futures)
        snapshot = cache.metrics(operator=True)
        assert count(snapshot, "requests", operation="execution") == 20
        assert count(snapshot, "origin_executions") == 1
        assert count(snapshot, "singleflight", role="leader") == 1
        assert count(snapshot, "singleflight", role="follower") == 19
        assert count(snapshot, "requests", operation="lookup") == 0
        assert count(snapshot, "requests", operation="store") == 0
        cache.inspect("policy")
        assert count(cache.metrics(operator=True), "requests", operation="lookup") == 0


@pytest.mark.asyncio
async def test_origin_error_and_cancellation_have_one_terminal_outcome():
    async with make_cache() as cache:

        async def broken():
            raise ValueError("secret provider failure")

        with pytest.raises(ValueError):
            await cache.aget_or_compute("policy", broken)
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked():
            entered.set()
            await release.wait()

        task = asyncio.create_task(cache.aget_or_compute("another policy", blocked))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        snapshot = cache.metrics(operator=True)
        assert count(snapshot, "requests", operation="execution") == 2
        assert count(snapshot, "requests", outcome="error") == 1
        assert count(snapshot, "requests", outcome="cancelled") == 1
        assert count(snapshot, "origin_executions") == 2


def test_scoped_stat_retention_is_bounded_and_process_totals_remain_exact():
    with make_cache() as cache:
        first = trusted_context(operation="operation-0")
        for number in range(1050):
            cache.lookup("policy", context=replace(first, operation=f"operation-{number}"))
        assert len(cache._partition_stats) == 1024
        assert cache.stats(operator=True)["misses"] == 1050
        assert cache.stats(operator=True)["stats_retention_evictions"] == 26
        assert cache.stats(context=first)["stats_window_started_at"] is None
        cache.lookup("policy", context=first)
        assert cache.stats(context=first)["misses"] == 1
        assert cache.stats(context=first)["stats_window_started_at"] is not None


def test_default_events_and_metrics_do_not_contain_payload_or_identity(caplog):
    with make_cache() as cache, caplog.at_level(logging.DEBUG, logger="longcache"):
        cache.get_or_compute(
            "private-query-text",
            lambda: {"answer": "private-response-text"},
            namespace="private-namespace",
        )
        serialized = json.dumps(cache.metrics(operator=True)) + json.dumps(
            [getattr(record, "longcache", {}) for record in caplog.records]
        )
        for forbidden in (
            "private-query-text",
            "private-response-text",
            "private-namespace",
            "test-tenant",
            "test-user",
            "test.answer",
            "fixture-model",
        ):
            assert forbidden not in serialized


def test_dynamic_hook_reasons_do_not_create_dynamic_metric_series():
    with make_cache() as cache:
        for number in range(2500):
            cache._admit = lambda *_, n=number: PolicyDecision("reject", f"tenant_reason_{n}")
            cache.set("policy", "answer")
        snapshot = cache.metrics(operator=True)
        assert len(snapshot["counters"]) + len(snapshot["latency"]) < 50
        assert "tenant_reason_" not in json.dumps(snapshot)


def test_measurements_can_be_disabled_and_snapshot_survives_close():
    cache = make_cache(track_stats=False)
    assert cache.get_or_compute("policy", lambda: "answer") == "answer"
    cache.close()
    snapshot = cache.metrics(operator=True)
    assert snapshot["enabled"] is False
    assert snapshot["counters"] == snapshot["latency"] == []
    assert snapshot["resources"]["state"] == "CLOSED"
    with pytest.raises(ValueError, match="operator"):
        cache.metrics()


def test_unsupported_decorator_reports_bypass_without_cache_access():
    with make_cache() as cache:

        @cache.cached
        def operation(query, extra):
            return extra

        assert operation("policy", "answer") == "answer"
        snapshot = cache.metrics(operator=True)
        assert count(snapshot, "requests", operation="execution", outcome="bypass") == 1
        assert count(snapshot, "origin_executions") == 1
        assert cache.entries() == []
