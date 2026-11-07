"""Verify public API surface, parameter cleanliness, and stability classification for v0.5.0."""

import inspect
from pathlib import Path
from typing import get_type_hints
import pytest
import longcache
import longcache.backends
from longcache.backends.async_redis import AsyncRedisCacheBackend
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
from longcache.core.errors import (
    LongCacheExperimentalWarning,
    LongCachePassThroughWarning,
)


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
        assert getattr(cls, "is_passthrough", False) is True
        assert getattr(cls, "api_stability", None) == "pass-through"
        assert PUBLIC_API_STABILITY.get(cls.__name__) == "pass-through"


def test_pass_through_classes_emit_warnings_by_default():
    """Instantiating or invoking pass-through wrappers emits LongCachePassThroughWarning."""

    class DummyLLM:
        def invoke(self, *a, **kw):
            return "ans"

        def stream(self, *a, **kw):
            yield "ans"

    cache = LongCache(CacheConfig(backend="memory"))

    with pytest.warns(
        LongCachePassThroughWarning, match="CachedLLM is a compatibility pass-through"
    ):
        CachedLLM(DummyLLM(), cache=cache)

    with pytest.warns(
        LongCachePassThroughWarning, match="CachedChain is a compatibility pass-through"
    ):
        CachedChain(DummyLLM(), cache=cache)

    with pytest.warns(
        LongCachePassThroughWarning, match="CachedQueryEngine is a compatibility pass-through"
    ):
        CachedQueryEngine(DummyLLM(), cache=cache)

    with pytest.warns(
        LongCachePassThroughWarning, match="StreamingCacheAdapter is a compatibility pass-through"
    ):
        StreamingCacheAdapter(cache=cache, stream_func=lambda q: ["chunk"])

    with pytest.warns(
        LongCachePassThroughWarning, match="cached_stream is a compatibility pass-through"
    ):
        cache.cached_stream(lambda q: ["chunk"])

    with pytest.warns(
        LongCachePassThroughWarning, match="CachedState is a compatibility pass-through"
    ):
        CachedState.check({"messages": "hi"}, cache=cache)

    with pytest.warns(
        LongCachePassThroughWarning, match="CachedState is a compatibility pass-through"
    ):
        CachedState.save({"messages": "hi"}, "response", cache=cache)


def test_pass_through_classes_can_silence_warnings():
    """Passing warn_passthrough=False suppresses LongCachePassThroughWarning."""
    import warnings

    class DummyLLM:
        def invoke(self, *a, **kw):
            return "ans"

    cache = LongCache(CacheConfig(backend="memory"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", LongCachePassThroughWarning)
        CachedLLM(DummyLLM(), cache=cache, warn_passthrough=False)
        CachedChain(DummyLLM(), cache=cache, warn_passthrough=False)
        CachedQueryEngine(DummyLLM(), cache=cache, warn_passthrough=False)
        StreamingCacheAdapter(cache=cache, stream_func=lambda q: ["chunk"], warn_passthrough=False)
        cache.cached_stream(lambda q: ["chunk"], warn_passthrough=False)
        CachedState.check({"messages": "hi"}, cache=cache, warn_passthrough=False)
        CachedState.save({"messages": "hi"}, "response", cache=cache, warn_passthrough=False)


def test_async_redis_backend_emits_experimental_warning():
    """AsyncRedisCacheBackend emits LongCacheExperimentalWarning by default."""
    with pytest.warns(LongCacheExperimentalWarning, match="experimental standalone backend"):
        AsyncRedisCacheBackend(redis_url="redis://localhost:6379/0")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", LongCacheExperimentalWarning)
        AsyncRedisCacheBackend(redis_url="redis://localhost:6379/0", warn_experimental=False)


def test_support_matrix_matches_public_api_stability():
    """Verify docs/support-matrix.md mentions all symbols in PUBLIC_API_STABILITY."""
    doc_path = Path(__file__).resolve().parent.parent / "docs" / "support-matrix.md"
    assert doc_path.is_file(), f"Support matrix doc missing at {doc_path}"
    content = doc_path.read_text(encoding="utf-8")

    # Every stable and pass-through symbol must be listed in support-matrix.md
    for symbol, tier in PUBLIC_API_STABILITY.items():
        if symbol in (
            "__version__",
            "PUBLIC_API_STABILITY",
            "StabilityTier",
            "API_CONTRACT_VERSION",
        ):
            continue
        assert symbol in content, f"Symbol {symbol} (tier: {tier}) missing from {doc_path.name}"


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
