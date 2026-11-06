"""Verify public API surface, parameter cleanliness, and stability classification for v0.5.0."""

import inspect
from typing import get_type_hints
import pytest
import longcache
import longcache.backends
import longcache.integrations
from longcache.integrations.langchain import CachedChain, CachedLLM
from longcache.integrations.langgraph import CachedState
from longcache.integrations.llamaindex import CachedQueryEngine
import longcache.middleware
from longcache.middleware import StreamingCacheAdapter
from longcache import (
    CacheConfig,
    LongCache,
    PUBLIC_API_STABILITY,
)
from longcache.backends.base import AbstractCacheBackend
from longcache.backends.redis_shared import SharedRedisCacheBackend


def test_public_api_stability_classification_completeness():
    """Every symbol exported across public module __all__ lists must be classified."""
    assert isinstance(PUBLIC_API_STABILITY, dict)
    valid_tiers = {"stable", "experimental", "pass-through"}

    for mod in (
        longcache,
        longcache.backends,
        longcache.integrations,
        longcache.middleware,
    ):
        assert hasattr(mod, "__all__"), f"Module {mod.__name__} missing __all__"
        for name in mod.__all__:
            assert name in PUBLIC_API_STABILITY, (
                f"Exported symbol {mod.__name__}.{name} lacks an entry in PUBLIC_API_STABILITY"
            )
            assert PUBLIC_API_STABILITY[name] in valid_tiers, (
                f"Symbol {name!r} has invalid tier: {PUBLIC_API_STABILITY[name]}"
            )
    assert "stable-candidate" not in set(PUBLIC_API_STABILITY.values())


def test_longcache_public_methods_have_no_private_or_legacy_kwargs():
    """No public method on LongCache should define `_`-prefixed parameters or opaque **kwargs (except cached_stream)."""
    public_methods = [
        name
        for name in dir(LongCache)
        if not name.startswith("_") and callable(getattr(LongCache, name))
    ]

    for method_name in public_methods:
        method = getattr(LongCache, method_name)
        sig = inspect.signature(method)
        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD and method_name != "cached_stream":
                pytest.fail(
                    f"Public method LongCache.{method_name} exposes opaque **{param_name} catch-all"
                )
            assert not param_name.startswith("_"), (
                f"Public method LongCache.{method_name} defines private parameter: {param_name}"
            )


def test_longcache_public_methods_are_fully_annotated():
    """All public methods on LongCache must have full parameter and return type annotations."""
    public_methods = [
        name
        for name in dir(LongCache)
        if not name.startswith("_") and callable(getattr(LongCache, name))
    ]

    for method_name in public_methods:
        method = getattr(LongCache, method_name)
        sig = inspect.signature(method)
        assert sig.return_annotation is not inspect.Signature.empty, (
            f"LongCache.{method_name} lacks return annotation"
        )
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            assert param.annotation is not inspect.Parameter.empty, (
                f"LongCache.{method_name} parameter {param_name} lacks type annotation"
            )
        # Ensure runtime type hint resolution succeeds
        hints = get_type_hints(method)
        assert "return" in hints


def test_pass_through_classes_expose_bypass_reason():
    """All pass-through wrapper classes must declare cache_bypass_reason == 'unsupported_request_shape'."""
    for cls in (
        StreamingCacheAdapter,
        CachedLLM,
        CachedChain,
        CachedState,
        CachedQueryEngine,
    ):
        assert getattr(cls, "cache_bypass_reason", None) == "unsupported_request_shape", (
            f"{cls.__name__} missing cache_bypass_reason == 'unsupported_request_shape'"
        )
        assert PUBLIC_API_STABILITY.get(cls.__name__) == "pass-through"


def test_dead_parameters_are_not_accepted_by_cache_config_or_methods():
    """Retired parameters (model, temperature, system_prompt, track_miss) cannot be passed."""
    for dead_arg in ["model", "temperature", "system_prompt", "track_miss"]:
        with pytest.raises(TypeError):
            CacheConfig(**{dead_arg: "forbidden"})  # type: ignore[arg-type]

    # Verify CacheEntry no longer carries legacy model/system_prompt/temperature fields
    from longcache import CacheEntry

    entry_fields = {f.name for f in CacheEntry.__dataclass_fields__.values()}
    assert entry_fields.isdisjoint({"model", "system_prompt", "temperature"})


def test_shared_redis_backend_lsp_conformance():
    """SharedRedisCacheBackend.store must match AbstractCacheBackend.store signature."""
    base_sig = inspect.signature(AbstractCacheBackend.store)
    shared_sig = inspect.signature(SharedRedisCacheBackend.store)

    base_params = list(base_sig.parameters.keys())
    shared_params = list(shared_sig.parameters.keys())

    assert base_params == shared_params == ["self", "entry"], (
        f"Parameter mismatch: base {base_params} vs shared {shared_params}"
    )

    assert hasattr(SharedRedisCacheBackend, "store_shared")
    shared_store_sig = inspect.signature(SharedRedisCacheBackend.store_shared)
    assert "entry" in shared_store_sig.parameters
    assert "expected_epoch" in shared_store_sig.parameters
    assert "lease_key" in shared_store_sig.parameters
