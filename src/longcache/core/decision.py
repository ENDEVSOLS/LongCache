"""Public, redacted cache decisions and policy hook results."""

from dataclasses import dataclass, field
from typing import Any, Literal
import re


@dataclass(frozen=True)
class PolicyDecision:
    action: Literal["accept", "reject", "abstain"]
    reason: str = "application_policy"

    def __post_init__(self):
        if self.action not in ("accept", "reject", "abstain"):
            raise ValueError("Invalid policy action")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.reason):
            raise ValueError("Policy reasons must be bounded machine-readable codes")


@dataclass(frozen=True)
class LookupDecision:
    outcome: Literal["HIT", "MISS", "BYPASS"]
    reason: str
    response: Any = field(default=None, repr=False)
    similarity: float | None = None
    entry_id: int | None = None
    threshold: float | None = None
    _epoch: int | None = field(default=None, repr=False, compare=False)
    _expires_at: float | None = field(default=None, repr=False, compare=False)

    def inspection(self) -> dict:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "similarity": self.similarity,
            "id": self.entry_id,
            "threshold": self.threshold,
            "format_version": 2,
        }


@dataclass(frozen=True)
class StoreDecision:
    outcome: Literal["STORED", "REJECTED"]
    reason: str
    entry_id: int | None = None
