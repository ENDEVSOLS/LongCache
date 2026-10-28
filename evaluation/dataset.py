"""Synthetic labeled cases expanded from reviewed operation templates."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    family: str
    split: str
    group: str
    seed: str
    query: str
    reuse: bool
    mutation: str = ""


def cases(path):
    raw = Path(path).read_bytes()
    source = json.loads(raw)
    result = []
    held = [a for a in source["actions"] if a not in source["development_actions"]]
    for i in range(2000):
        action = held[i % len(held)]
        account = f"Account{i:04d}"
        seed = f"How do I {action} for {account}?"
        query = f"How can I {action} for {account}?"
        result.append(Case(f"safe-{i}", action, "held-out", "safe", seed, query, True))
    for i in range(2000):
        kind = i % 4
        account = f"Account{i:04d}"
        if kind == 0:
            seed = f"Can I cancel the subscription for {account}?"
            query = f"Can I not cancel the subscription for {account}?"
        elif kind == 1:
            seed = f"Can I return an item within 30 days for {account}?"
            query = f"Can I return an item within 40 days for {account}?"
        elif kind == 2:
            seed = f"How do I close {account}?"
            query = f"How do I close Other{i:04d}?"
        else:
            seed = f"Can I cancel before shipment for {account}?"
            query = f"Can I cancel after shipment for {account}?"
        result.append(Case(f"near-{i}", f"near-{kind}", "held-out", "near", seed, query, False))
    mutations = [
        "tenant",
        "user",
        "model",
        "provider",
        "deployment",
        "instructions",
        "generation",
        "prompt",
        "operation",
        "corpus",
        "document",
    ]
    for i in range(500):
        text = f"How do I update Account{i:04d}?"
        result.append(
            Case(
                f"scope-{i}",
                "scope-" + mutations[i % len(mutations)],
                "held-out",
                "isolation",
                text,
                text,
                False,
                mutations[i % len(mutations)],
            )
        )
    for i in range(200):
        text = f"How do I update Account{i:04d}?"
        kind = ["missing", "multi-turn", "tool-calling", "multimodal", "streaming"][i % 5]
        result.append(
            Case(f"unsupported-{i}", kind, "held-out", "unsupported", text, text, False, kind)
        )
    for i in range(100):
        text = f"How do I update Account{i:04d}?"
        kind = ["corpus", "expired", "deleted", "invalidated", "legacy"][i % 5]
        result.append(Case(f"fresh-{i}", kind, "held-out", "freshness", text, text, False, kind))
    return source, result, hashlib.sha256(raw).hexdigest()
