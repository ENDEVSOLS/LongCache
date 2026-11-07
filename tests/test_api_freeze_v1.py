"""Test API freeze and contract lock for LongCache v1.0.0.

Guarantees that public signatures, parameter kinds/defaults, exports,
stability tiers, dataclass fields, and exception hierarchies cannot break
in v1.0.0 or future 1.x releases.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path
import pytest

import longcache
import longcache.backends
from longcache.backends.base import AbstractCacheBackend
from longcache.backends.memory import InMemoryCacheBackend
from longcache.backends.redis import RedisCacheBackend
from longcache.backends.redis_shared import SharedRedisCacheBackend
from longcache.backends.sqlite import SQLiteCacheBackend
from longcache import cli
import longcache.integrations
from longcache.integrations.openai import CachedOpenAI
import longcache.middleware
from longcache.middleware import LongCacheMiddleware
from longcache.core.errors import (
    CacheError,
    CacheClosedError,
    CacheShutdownTimeoutError,
    CacheCapacityError,
    CacheOperationError,
    SingleflightError,
    SingleflightTimeoutError,
    SingleflightCapacityError,
    SingleflightLeaderError,
    SingleflightReentrancyError,
    SingleflightInvalidatedError,
    LongCacheWarning,
    LongCachePassThroughWarning,
    LongCacheExperimentalWarning,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "api_contract_v1_0.json"


@pytest.fixture(scope="module")
def api_contract() -> dict:
    assert FIXTURE_PATH.is_file(), f"API contract fixture missing at {FIXTURE_PATH}"
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_api_contract_version_and_release_metadata():
    """Verify API_CONTRACT_VERSION is '1.0' and version starts with 1.0.0."""
    assert hasattr(longcache, "API_CONTRACT_VERSION")
    assert longcache.API_CONTRACT_VERSION == "1.0"
    assert longcache.__version__.startswith("1.0.0")


def test_public_api_stability_promoted_to_stable(api_contract):
    """Verify all qualified symbols have graduated from stable-candidate to stable."""
    assert longcache.PUBLIC_API_STABILITY == api_contract["stability_tiers"]
    assert "stable-candidate" not in set(longcache.PUBLIC_API_STABILITY.values())

    # Shared Redis backend must be stable in 1.0.0
    assert longcache.PUBLIC_API_STABILITY["SharedRedisCacheBackend"] == "stable"
    assert longcache.PUBLIC_API_STABILITY["SharedCoordinationBackend"] == "stable"

    # Standalone async Redis backend remains experimental
    assert longcache.PUBLIC_API_STABILITY["AsyncRedisCacheBackend"] == "experimental"


def test_public_module_exports_match_contract(api_contract):
    """Verify that module __all__ exports match the frozen contract."""
    modules = {
        "longcache": longcache,
        "longcache.backends": longcache.backends,
        "longcache.integrations": longcache.integrations,
        "longcache.middleware": longcache.middleware,
    }
    for mod_name, mod in modules.items():
        assert mod.__all__ == api_contract["modules_exports"][mod_name]


def test_stable_classes_signatures_match_contract(api_contract):
    """Verify method parameters, kinds, and defaults for all stable classes."""
    classes = {
        "LongCache": longcache.LongCache,
        "CacheConfig": longcache.CacheConfig,
        "RequestScope": longcache.RequestScope,
        "RequestContext": longcache.RequestContext,
        "Instruction": longcache.Instruction,
        "LookupDecision": longcache.LookupDecision,
        "StoreDecision": longcache.StoreDecision,
        "PolicyDecision": longcache.PolicyDecision,
        "BaseEmbedder": longcache.BaseEmbedder,
        "EmbeddingDescriptor": longcache.EmbeddingDescriptor,
        "CacheEntry": longcache.CacheEntry,
        "AbstractCacheBackend": AbstractCacheBackend,
        "SQLiteCacheBackend": SQLiteCacheBackend,
        "RedisCacheBackend": RedisCacheBackend,
        "SharedRedisCacheBackend": SharedRedisCacheBackend,
        "InMemoryCacheBackend": InMemoryCacheBackend,
        "CachedOpenAI": CachedOpenAI,
        "LongCacheMiddleware": LongCacheMiddleware,
    }

    for cls_name, cls in classes.items():
        expected_methods = api_contract["classes"][cls_name]["methods"]
        for method_name, expected in expected_methods.items():
            assert hasattr(cls, method_name), (
                f"Class {cls_name} missing frozen method {method_name}"
            )
            method = getattr(cls, method_name)
            sig = inspect.signature(method)
            actual_params = []
            for p_name, param in sig.parameters.items():
                default_repr = (
                    repr(param.default) if param.default is not inspect.Parameter.empty else None
                )
                actual_params.append(
                    {
                        "name": p_name,
                        "kind": param.kind.name,
                        "default": default_repr,
                        "annotation": str(param.annotation)
                        if param.annotation is not inspect.Parameter.empty
                        else "Any",
                    }
                )
            # Verify parameter names and kinds
            expected_params = expected["parameters"]
            assert len(actual_params) == len(expected_params), (
                f"{cls_name}.{method_name} parameter count mismatch"
            )
            for act, exp in zip(actual_params, expected_params):
                assert act["name"] == exp["name"], (
                    f"{cls_name}.{method_name} parameter name mismatch: {act['name']} != {exp['name']}"
                )
                assert act["kind"] == exp["kind"], (
                    f"{cls_name}.{method_name} parameter {act['name']} kind mismatch: {act['kind']} != {exp['kind']}"
                )
                assert act["default"] == exp["default"], (
                    f"{cls_name}.{method_name} parameter {act['name']} default mismatch: {act['default']} != {exp['default']}"
                )


def test_dataclass_fields_match_contract(api_contract):
    """Verify dataclass field names, order, and defaults."""
    dataclass_map = {
        "CacheConfig": longcache.CacheConfig,
        "RequestScope": longcache.RequestScope,
        "RequestContext": longcache.RequestContext,
        "Instruction": longcache.Instruction,
        "LookupDecision": longcache.LookupDecision,
        "StoreDecision": longcache.StoreDecision,
        "PolicyDecision": longcache.PolicyDecision,
        "EmbeddingDescriptor": longcache.EmbeddingDescriptor,
        "CacheEntry": longcache.CacheEntry,
    }
    for name, cls in dataclass_map.items():
        expected_fields = api_contract["dataclasses"][name]
        actual_fields = [
            {
                "name": f.name,
                "type": str(f.type),
                "default": repr(f.default) if f.default is not dataclasses.MISSING else None,
            }
            for f in dataclasses.fields(cls)
        ]
        assert [f["name"] for f in actual_fields] == [f["name"] for f in expected_fields]
        assert [f["default"] for f in actual_fields] == [f["default"] for f in expected_fields]


def test_exception_and_warning_hierarchy_matches_contract(api_contract):
    """Verify exception and warning inheritance hierarchies."""
    exc_list = [
        CacheError,
        CacheClosedError,
        CacheShutdownTimeoutError,
        CacheCapacityError,
        CacheOperationError,
        SingleflightError,
        SingleflightTimeoutError,
        SingleflightCapacityError,
        SingleflightLeaderError,
        SingleflightReentrancyError,
        SingleflightInvalidatedError,
        LongCacheWarning,
        LongCachePassThroughWarning,
        LongCacheExperimentalWarning,
    ]
    for exc in exc_list:
        actual_bases = [base.__name__ for base in exc.__mro__[1:]]
        expected_bases = api_contract["exceptions"][exc.__name__]
        assert actual_bases == expected_bases, f"Exception hierarchy mismatch for {exc.__name__}"


def test_cli_commands_and_options_match_contract(api_contract):
    """Verify Click CLI commands and parameter options."""
    for cmd_name, expected_options in api_contract["cli_commands"].items():
        assert cmd_name in cli.main.commands, f"CLI command {cmd_name} missing"
        cmd = cli.main.commands[cmd_name]
        actual_options = {}
        for p in cmd.params:
            d = p.default
            if d is None or "Sentinel" in str(d):
                norm = None
            else:
                norm = repr(d)
            actual_options[p.name] = {
                "opts": sorted(list(p.opts)),
                "required": p.required,
                "is_flag": getattr(p, "is_flag", False),
                "default": norm,
            }
        assert actual_options == expected_options, f"CLI command options mismatch for {cmd_name}"
