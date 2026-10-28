from dataclasses import replace
import pytest
from longcache.core.identity import Instruction, RequestContext, RequestScope
from longcache.config import CacheConfig


def context(**kwargs):
    defaults = dict(
        scope=RequestScope("tenant", "alice"),
        operation="answer",
        provider="test",
        provider_scope="deployment",
        model="model",
        prompt_version="v1",
    )
    return RequestContext(**(defaults | kwargs))


def key(ctx):
    return ctx.partition("faq", "private", "embedding-v1", "policy-v1")


def test_context_snapshots_nested_parameters():
    params = {"stop": ["end"], "temperature": 0.0}
    ctx = context(generation=params)
    before = key(ctx)
    params["stop"].append("other")
    assert key(ctx) == before
    with pytest.raises(TypeError):
        ctx.generation["temperature"] = 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", RequestScope("tenant", "bob")),
        ("model", "other"),
        ("provider_scope", "other"),
        ("prompt_version", "v2"),
        ("operation", "other"),
        ("corpus_version", "v2"),
        ("generation", {"temperature": None}),
        ("instructions", (Instruction("system", "policy"),)),
        ("document_versions", {"doc": "2"}),
    ],
)
def test_response_affecting_fields_separate_partitions(field, value):
    assert key(context()) != key(replace(context(), **{field: value}))


def test_mapping_order_and_exact_query():
    a = context(generation={"a": 1, "b": 2})
    b = context(generation={"b": 2, "a": 1})
    assert key(a) == key(b)
    assert a.exact_key("hello", "faq", "private", "e", "p") != a.exact_key(
        "Hello", "faq", "private", "e", "p"
    )


def test_invalid_config_and_nonfinite_identity():
    with pytest.raises(ValueError):
        CacheConfig.from_dict({"scope_polcy": "public"})
    with pytest.raises(ValueError):
        context(generation={"temperature": float("nan")})
    assert "alice" not in repr(context())


@pytest.mark.parametrize(
    "left,right",
    [
        ({}, {"temperature": None}),
        ({"temperature": 0}, {"temperature": "0"}),
        ({"temperature": False}, {"temperature": 0}),
    ],
)
def test_canonical_identity_preserves_omission_null_and_scalar_types(left, right):
    assert key(context(generation=left)) != key(context(generation=right))


def test_instruction_order_is_response_affecting():
    first = Instruction("system", "Use current policy")
    second = Instruction("developer", "Only answer in English")
    assert key(context(instructions=(first, second))) != key(context(instructions=(second, first)))


def test_unregistered_custom_backend_is_explicitly_unsupported():
    with pytest.raises(ValueError):
        CacheConfig(backend="drops-metadata")
