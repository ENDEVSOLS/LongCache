from evaluation.dataset import cases
from evaluation.run import check_gates, metrics, upper_bound


def test_versioned_dataset_counts_and_disjoint_families():
    source, rows, fingerprint = cases("evaluation/data/seeds.json")
    assert len(rows) == 4800
    assert len({r.id for r in rows}) == 4800
    assert len(fingerprint) == 64
    assert not set(source["development_actions"]) & {r.family for r in rows}
    assert sum(r.group == "safe" for r in rows) == 2000
    assert sum(r.group == "near" for r in rows) == 2000


def test_zero_hits_cannot_pass_false_hit_gate():
    result = metrics([])
    assert result["false_hit_rate"] is None
    assert check_gates(result, {"false_hit_rate_max": 0.005, "accepted_hits_min": 1000})
    assert upper_bound(0, 1000) < 0.01
    assert upper_bound(1, 1) == 1


def test_latency_comparison_detects_real_regressions_without_relaxing_budget():
    from evaluation.performance import comparisons

    baseline = [{"index": "flat", "entries": 10000, "p95_ms": 30.0}]
    assert comparisons([{"index": "flat", "entries": 10000, "p95_ms": 40.0}], baseline)[0]["passed"]
    assert not comparisons([{"index": "flat", "entries": 10000, "p95_ms": 50.0}], baseline)[0][
        "passed"
    ]
    assert (
        comparisons([{"index": "flat", "entries": 10000, "p95_ms": 40.0}], baseline)[0][
            "budget_p95_ms"
        ]
        == 41.0
    )


def test_paired_sampling_resists_sequential_clock_drift():
    from evaluation.performance import paired_samples, comparisons, summarize

    tick = iter(range(8))
    calls = []

    def lookup(worker, command):
        calls.append(worker)
        # Both versions do identical work; the host becomes faster over time.
        elapsed = 1000.0 - 100.0 * next(tick)
        return {"latency_ms": elapsed, "embedding_ms": elapsed - 0.2}

    rows = paired_samples(["new", "old"], lookup, 4)
    assert calls == ["new", "old", "old", "new", "new", "old", "old", "new"]
    current = {"index": "flat", "entries": 10000, **summarize(rows[0])}
    previous = {"index": "flat", "entries": 10000, **summarize(rows[1])}
    assert comparisons([current], [previous])[0]["passed"]
