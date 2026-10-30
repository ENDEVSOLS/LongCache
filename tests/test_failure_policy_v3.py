"""Only cache dependency failures participate in strict/bypass policy."""

import logging
import pytest

from longcache import CacheConfig, LongCache
from longcache.core.errors import CacheOperationError
from tests.helpers import HashEmbedder, trusted_context


def cache_for(policy="raise"):
    return LongCache(
        CacheConfig(backend="memory", ttl_seconds=0, cache_error_policy=policy),
        embedder=HashEmbedder(),
        context=trusted_context(),
    )


def unavailable(*args, **kwargs):
    raise OSError("secret prompt and credential must not reach logs")


@pytest.mark.parametrize("policy", ["raise", "bypass"])
@pytest.mark.parametrize("stage", ["embedding", "index", "backend"])
def test_lookup_dependency_policy_and_redaction(monkeypatch, caplog, policy, stage):
    with cache_for(policy) as cache:
        cache.set("policy", "answer")
        obj, method = {
            "embedding": (cache._embedding, "embed"),
            "index": (cache.index, "candidates"),
            "backend": (cache.backend, "retrieve"),
        }[stage]
        monkeypatch.setattr(obj, method, unavailable)
        with caplog.at_level(logging.DEBUG, logger="longcache"):
            if policy == "raise":
                with pytest.raises(CacheOperationError) as raised:
                    cache.lookup("policy")
                assert raised.value.stage == stage
            else:
                decision = cache.lookup("policy")
                assert decision.outcome == "BYPASS"
                assert decision.response is None
        assert "secret prompt" not in caplog.text
        assert "credential" not in caplog.text


@pytest.mark.parametrize("policy", ["raise", "bypass"])
def test_partial_write_quarantines_state(monkeypatch, policy):
    with cache_for(policy) as cache:
        monkeypatch.setattr(cache.index, "add", unavailable)
        if policy == "raise":
            with pytest.raises(CacheOperationError):
                cache.set("policy", "answer")
        else:
            assert cache.set("policy", "answer").outcome == "REJECTED"
        assert cache.metrics(operator=True)["health"] == "degraded"
        assert not cache._coherent
        if policy == "raise":
            with pytest.raises(CacheOperationError):
                cache.lookup("policy")
        else:
            assert cache.lookup("policy").outcome == "BYPASS"


def test_bypass_never_hides_invalid_embedding_contract(monkeypatch):
    with cache_for("bypass") as cache:
        cache.set("policy", "answer")

        def invalid(*_):
            raise ValueError("Invalid embedding shape")

        monkeypatch.setattr(cache._embedding, "embed", invalid)
        with pytest.raises(ValueError, match="shape"):
            cache.lookup("policy")


def test_optional_hit_accounting_failure_does_not_discard_valid_hit(monkeypatch):
    with cache_for() as cache:
        cache.set("policy", "answer")
        monkeypatch.setattr(cache.backend, "update_hit_count", unavailable)
        assert cache.get("policy") == "answer"
        assert cache.stats()["hits"] == 1


def test_failed_invalidation_is_never_reported_as_success(monkeypatch):
    with cache_for("bypass") as cache:
        cache.set("policy", "answer")
        monkeypatch.setattr(cache.backend, "delete_namespace", unavailable)
        with pytest.raises(CacheOperationError):
            cache.invalidate()
        assert cache.get("policy") is None
        assert cache.metrics(operator=True)["health"] == "degraded"


def test_broken_logging_handler_does_not_change_cache_decisions():
    class Broken(logging.Handler):
        def emit(self, record):
            raise RuntimeError("broken diagnostics")

    logger = logging.getLogger("longcache")
    handler, previous = Broken(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with cache_for() as cache:
            cache.set("policy", "answer")
            assert cache.get("policy") == "answer"
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
