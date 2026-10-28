"""Immutable request identity. Similarity never substitutes for exact execution scope."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(type(k) is str for k in value):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    raise ValueError("Identity contains unsupported or non-finite values")


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(
        plain(freeze(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, repr=False)
class RequestScope:
    tenant_id: str
    user_id: str | None = None

    def __post_init__(self):
        if type(self.tenant_id) is not str or not self.tenant_id.strip():
            raise ValueError("A trusted nonempty tenant identifier is required")
        if self.user_id is not None and (type(self.user_id) is not str or not self.user_id.strip()):
            raise ValueError("User identifier must be nonempty when supplied")

    def __repr__(self):
        return "RequestScope(<redacted>)"


@dataclass(frozen=True)
class Instruction:
    role: str
    content: str = field(repr=False)

    def __post_init__(self):
        if self.role not in ("system", "developer") or type(self.content) is not str:
            raise ValueError("Instructions require a supported role and text content")


@dataclass(frozen=True)
class RequestContext:
    scope: RequestScope | None = field(repr=False)
    operation: str
    provider: str
    provider_scope: str
    model: str
    prompt_version: str
    instructions: tuple[Instruction, ...] = field(default=(), repr=False)
    generation: Mapping[str, Any] = field(default_factory=dict, repr=False)
    corpus_version: str | None = None
    document_versions: Mapping[str, str] = field(default_factory=dict, repr=False)
    protected_entities: tuple[str, ...] = field(default=(), repr=False)
    language: str = "en"
    request_shape: str = "single-turn-text"
    codec: str = "json-v1"

    def __post_init__(self):
        for name in (
            "operation",
            "provider",
            "provider_scope",
            "model",
            "prompt_version",
            "language",
            "request_shape",
            "codec",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.scope is not None and not isinstance(self.scope, RequestScope):
            raise ValueError("scope must be a RequestScope")
        if not all(isinstance(x, Instruction) for x in self.instructions):
            raise ValueError("instructions must contain Instruction values")
        if not isinstance(self.generation, Mapping) or not isinstance(
            self.document_versions, Mapping
        ):
            raise ValueError("Context parameters must be mappings")
        if self.corpus_version is not None and (
            type(self.corpus_version) is not str or not self.corpus_version
        ):
            raise ValueError("corpus_version must be a nonempty string")
        if not all(type(v) is str and v for v in self.document_versions.values()):
            raise ValueError("Document versions must be nonempty strings")
        if not isinstance(self.protected_entities, (tuple, list)) or not all(
            type(x) is str and x for x in self.protected_entities
        ):
            raise ValueError("Protected entities must be nonempty strings")
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "generation", freeze(self.generation))
        object.__setattr__(self, "document_versions", freeze(self.document_versions))
        object.__setattr__(self, "protected_entities", tuple(self.protected_entities))

    def record(self, namespace: str, sharing: str, embedding: str, policy: str) -> dict:
        scope = self.scope
        scope_value = None
        if sharing != "public" and scope is not None:
            scope_value = digest(
                {"tenant": scope.tenant_id, "user": scope.user_id if sharing == "private" else None}
            )
        return {
            "format": 2,
            "namespace": namespace,
            "sharing": sharing,
            "scope": scope_value,
            "operation": self.operation,
            "provider": self.provider,
            "provider_scope": self.provider_scope,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "instructions": digest(
                [{"role": x.role, "content": x.content} for x in self.instructions]
            ),
            "generation": plain(self.generation),
            "corpus_version": self.corpus_version,
            "documents": plain(self.document_versions),
            "entities": list(self.protected_entities),
            "language": self.language,
            "shape": self.request_shape,
            "codec": self.codec,
            "embedding": embedding,
            "policy": policy,
        }

    def partition(self, namespace: str, sharing: str, embedding: str, policy: str) -> str:
        return "v2:" + digest(self.record(namespace, sharing, embedding, policy))

    def exact_key(
        self, query: str, namespace: str, sharing: str, embedding: str, policy: str
    ) -> str:
        return digest(
            {"partition": self.partition(namespace, sharing, embedding, policy), "query": query}
        )
