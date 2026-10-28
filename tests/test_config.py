"""Tests for CacheConfig: defaults, validation, and factory methods."""

import pytest

from longcache.config import CacheConfig


class TestCacheConfigDefaults:
    def test_default_values(self):
        config = CacheConfig()
        assert config.similarity_threshold == 0.92
        assert config.ttl_seconds == 3600
        assert config.max_entries == 10_000
        assert config.namespace == "default"
        assert config.backend == "sqlite"
        assert config.track_stats is True

    def test_to_dict_round_trip(self):
        config = CacheConfig(similarity_threshold=0.8, backend="memory")
        d = config.to_dict()
        assert d["similarity_threshold"] == 0.8
        assert d["backend"] == "memory"
        assert CacheConfig.from_dict(d) == config


class TestCacheConfigValidation:
    def test_similarity_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            CacheConfig(similarity_threshold=-0.1)

    def test_similarity_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            CacheConfig(similarity_threshold=1.1)

    def test_similarity_threshold_boundary_values_are_valid(self):
        CacheConfig(similarity_threshold=0.0)
        CacheConfig(similarity_threshold=1.0)

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds"):
            CacheConfig(ttl_seconds=-1)

    def test_zero_ttl_is_valid(self):
        config = CacheConfig(ttl_seconds=0)
        assert config.ttl_seconds == 0

    def test_max_entries_below_one_raises(self):
        with pytest.raises(ValueError, match="max_entries"):
            CacheConfig(max_entries=0)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            CacheConfig(backend="dynamodb")

    def test_all_supported_backends_are_valid(self):
        for backend in ("sqlite", "redis", "async_redis", "memory"):
            config = CacheConfig(backend=backend)
            assert config.backend == backend


class TestCacheConfigFromDict:
    def test_from_dict_basic(self):
        config = CacheConfig.from_dict(
            {
                "similarity_threshold": 0.90,
                "backend": "redis",
                "ttl_seconds": 7200,
            }
        )
        assert config.similarity_threshold == 0.90
        assert config.backend == "redis"
        assert config.ttl_seconds == 7200

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown configuration"):
            CacheConfig.from_dict({"scope_polcy": "public"})

    def test_from_dict_empty_uses_defaults(self):
        config = CacheConfig.from_dict({})
        assert config == CacheConfig()


class TestCacheConfigFromEnv:
    def test_from_env_reads_prefixed_vars(self, monkeypatch):
        monkeypatch.setenv("LONGCACHE_SIMILARITY_THRESHOLD", "0.90")
        monkeypatch.setenv("LONGCACHE_BACKEND", "redis")
        monkeypatch.setenv("LONGCACHE_TTL_SECONDS", "7200")

        config = CacheConfig.from_env()
        assert config.similarity_threshold == 0.90
        assert config.backend == "redis"
        assert config.ttl_seconds == 7200

    def test_from_env_casts_bool(self, monkeypatch):
        monkeypatch.setenv("LONGCACHE_TRACK_STATS", "false")
        config = CacheConfig.from_env()
        assert config.track_stats is False

        monkeypatch.setenv("LONGCACHE_TRACK_STATS", "true")
        config = CacheConfig.from_env()
        assert config.track_stats is True

    def test_from_env_casts_int(self, monkeypatch):
        monkeypatch.setenv("LONGCACHE_MAX_ENTRIES", "500")
        config = CacheConfig.from_env()
        assert config.max_entries == 500
        assert isinstance(config.max_entries, int)

    def test_from_env_no_vars_set_uses_defaults(self, monkeypatch):
        # Ensure a clean slate for LONGCACHE_* variables
        for key in list(__import__("os").environ):
            if key.startswith("LONGCACHE_"):
                monkeypatch.delenv(key, raising=False)

        config = CacheConfig.from_env()
        assert config == CacheConfig()

    def test_from_env_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MYAPP_BACKEND", "memory")
        config = CacheConfig.from_env(prefix="MYAPP_")
        assert config.backend == "memory"
