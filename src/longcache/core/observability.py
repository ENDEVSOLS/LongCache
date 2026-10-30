"""Bounded in-process measurements and payload-free standard logging."""

from collections import Counter
from functools import wraps
import logging
import threading
import time

LOGGER = logging.getLogger("longcache")
LOGGER.addHandler(logging.NullHandler())
BUCKETS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 30, 120, float("inf"))
LABELS = {
    "stage": {
        "prepare",
        "embedding",
        "index",
        "backend",
        "store",
        "accounting",
        "sweeper",
        "lifecycle",
        "verification",
        "admission",
        "wait",
        "origin",
        "execution",
        "lookup",
    },
    "operation": {"lookup", "store", "execution"},
    "mode": {"sync", "async"},
    "outcome": {
        "hit",
        "miss",
        "bypass",
        "stored",
        "rejected",
        "error",
        "cancelled",
        "timeout",
        "accept",
        "reject",
        "abstain",
        "failure",
    },
    "source": {"cache", "origin", "coalesced", "none"},
    "role": {"leader", "follower", "none"},
    "entry_point": {"manual", "decorator", "openai", "asgi"},
    "reason": {
        "accepted",
        "accepted_shared_result",
        "missing_identity",
        "unsupported_request_shape",
        "unsupported_parameter",
        "no_candidate",
        "cache_unavailable",
        "coherence_uncertain",
        "hook_failed",
        "application_policy",
        "cancelled",
        "timeout",
        "invalidated",
        "capacity",
        "leader_failed",
        "non_shareable",
        "response_inadmissible",
        "negation_mismatch",
        "number_mismatch",
        "entity_mismatch",
        "relation_mismatch",
        "verification_abstained",
        "verification_budget_exhausted",
        "legacy_format",
        "codec_invalid",
        "expired_or_deleted",
        "shutdown",
        "background_failed",
        "response_too_large",
        "provider_failed",
    },
}
FAMILIES = {
    "requests",
    "decisions",
    "errors",
    "verification",
    "admission",
    "singleflight",
    "shared_deliveries",
    "origin_executions",
    "observability_drops",
}


def best_effort(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:
            return None  # Never recursively report a broken diagnostic path.

    return wrapped


class Observability:
    def __init__(self, enabled, backend):
        self.enabled, self.backend = enabled, backend
        self._lock = threading.Lock()
        self._counts = Counter()
        self._histograms = {}
        self._warnings = {}

    @staticmethod
    def _labels(labels):
        return tuple(
            sorted(
                (
                    key,
                    value
                    if type(value) is str and value in LABELS[key]
                    else "application_policy"
                    if key == "reason"
                    else "other",
                )
                for key, value in labels.items()
                if key in LABELS
            )
        )

    @best_effort
    def count(self, family, **labels):
        if not self.enabled or family not in FAMILIES:
            return
        key = family, self._labels(labels)
        with self._lock:
            if key in self._counts or len(self._counts) + len(self._histograms) < 2000:
                self._counts[key] += 1

    @best_effort
    def duration(self, stage, seconds, mode="sync"):
        if not self.enabled:
            return
        key = self._labels({"stage": stage, "mode": mode})
        with self._lock:
            if key not in self._histograms:
                if len(self._counts) + len(self._histograms) >= 2000:
                    return
                self._histograms[key] = [0, 0.0, [0] * len(BUCKETS)]
            histogram = self._histograms[key]
            histogram[0] += 1
            histogram[1] += seconds
            for index, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    histogram[2][index] += 1

    @best_effort
    def event(self, *, warning=False, **labels):
        safe = dict(self._labels(labels))
        now = time.monotonic()
        if warning:
            key = tuple(sorted(safe.items()))
            with self._lock:
                if now - self._warnings.get(key, float("-inf")) < 60:
                    return
                if key not in self._warnings and len(self._warnings) >= 64:
                    return
                self._warnings[key] = now
        try:
            LOGGER.log(
                logging.WARNING if warning else logging.DEBUG,
                "longcache.dependency" if warning else "longcache.decision",
                extra={"longcache": {"event_version": 1, "backend": self.backend, **safe}},
            )
        except Exception:
            self.count("observability_drops")

    def snapshot(self):
        with self._lock:
            return {
                "enabled": self.enabled,
                "counters": [
                    {"name": name, "labels": dict(labels), "value": value}
                    for (name, labels), value in sorted(self._counts.items())
                ],
                "latency": [
                    {
                        "labels": dict(labels),
                        "count": h[0],
                        "sum_seconds": h[1],
                        "buckets": [
                            {"le": bound if bound != float("inf") else "+Inf", "count": count}
                            for bound, count in zip(BUCKETS, h[2])
                        ],
                    }
                    for labels, h in sorted(self._histograms.items())
                ],
            }
