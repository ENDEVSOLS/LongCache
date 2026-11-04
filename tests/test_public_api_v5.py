"""Verify public API surface, parameter cleanliness, and stability classification for v0.5.0."""

import inspect
import pytest
import longcache
from longcache import (
    CacheConfig,
    LongCache,
    PUBLIC_API_STABILITY,
)
from longcache.backends.base import AbstractCacheBackend
from longcache.backends.redis_shared import SharedRedisCacheBackend


def test_public_api_stability_classification_completeness():
    """Every symbol exported in __all__ must be classified into a stability tier."""
    assert isinstance(PUBLIC_API_STABILITY, dict)
    valid_tiers = {"stable-candidate", "experimental", "pass-through"}

    for name in longcache.__all__:
        assert name in PUBLIC_API_STABILITY, (
            f"Exported symbol {name!r} lacks an entry in PUBLIC_API_STABILITY"
        )
        assert PUBLIC_API_STABILITY[name] in valid_tiers, (
            f"Symbol {name!r} has invalid tier: {PUBLIC_API_STABILITY[name]}"
        )


def test_longcache_public_methods_have_no_private_prefixed_parameters():
    """No public method on LongCache should define `_`-prefixed parameters."""
    public_methods = [
        "lookup",
        "alookup",
        "set",
        "aset",
        "get",
        "aget",
        "get_or_compute",
        "aget_or_compute",
        "cached",
        "stats",
        "metrics",
        "inspect",
        "invalidate",
        "clear",
        "entries",
        "close",
        "aclose",
    ]

    for method_name in public_methods:
        method = getattr(LongCache, method_name)
        sig = inspect.signature(method)
        for param_name, param in sig.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            assert not param_name.startswith("_"), (
                f"Public method LongCache.{method_name} defines private parameter: {param_name}"
            )


def test_longcache_public_methods_are_fully_annotated():
    """All public methods on LongCache must have parameter and return type annotations."""
    public_methods = [
        "lookup",
        "alookup",
        "set",
        "aset",
        "get",
        "aget",
        "get_or_compute",
        "aget_or_compute",
        "cached",
        "stats",
        "metrics",
        "inspect",
        "invalidate",
        "clear",
        "entries",
        "close",
        "aclose",
    ]

    for method_name in public_methods:
        method = getattr(LongCache, method_name)
        sig = inspect.signature(method)
        # Check return annotation
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"LongCache.{method_name} lacks return annotation"
        )

        # Check parameter annotations
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.annotation is not inspect.Parameter.empty, (
                f"LongCache.{method_name} parameter {param_name} lacks type annotation"
            )


def test_dead_parameters_are_not_accepted_by_cache_config():
    """Retired parameters (model, temperature, system_prompt, track_miss) cannot be passed to CacheConfig."""
    for dead_arg in ["model", "temperature", "system_prompt", "track_miss"]:
        with pytest.raises(TypeError):
            CacheConfig(**{dead_arg: "forbidden"})  # type: ignore[arg-type]


def test_shared_redis_backend_lsp_conformance():
    """SharedRedisCacheBackend.store must match AbstractCacheBackend.store signature."""
    base_sig = inspect.signature(AbstractCacheBackend.store)
    shared_sig = inspect.signature(SharedRedisCacheBackend.store)

    base_params = list(base_sig.parameters.keys())
    shared_params = list(shared_sig.parameters.keys())

    assert base_params == shared_params == ["self", "entry"], (
        f"Parameter mismatch: base {base_params} vs shared {shared_params}"
    )

    # Also verify that store_shared exists on SharedRedisCacheBackend for multi-tenant / lease usage
    assert hasattr(SharedRedisCacheBackend, "store_shared")
    shared_store_sig = inspect.signature(SharedRedisCacheBackend.store_shared)
    assert "entry" in shared_store_sig.parameters
    assert "expected_epoch" in shared_store_sig.parameters
    assert "lease_key" in shared_store_sig.parameters
