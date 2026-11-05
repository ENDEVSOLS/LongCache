"""Verify consumer typing against the installed LongCache wheel using mypy --strict."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import longcache

CONSUMER_CODE = """\
from typing import Any, assert_type
import numpy as np
import longcache
from longcache import (
    BaseEmbedder,
    CacheConfig,
    EmbeddingDescriptor,
    InspectionReport,
    Instruction,
    LongCache,
    LookupDecision,
    MetricsSnapshot,
    PolicyDecision,
    RequestContext,
    RequestScope,
    StatsSnapshot,
    StoreDecision,
)


class FixtureEmbedder(BaseEmbedder):
    dimension: int = 2
    descriptor: EmbeddingDescriptor = EmbeddingDescriptor("fixture", "v1", 2)

    def embed(self, text: str) -> np.ndarray[Any, Any]:
        return np.array([1.0, 0.0], dtype=np.float32)


async def async_fn() -> str:
    return "async-answer"


async def main() -> None:
    ctx: RequestContext = RequestContext(
        scope=RequestScope("consumer", "user-1"),
        operation="test.run",
        provider="fixture",
        provider_scope="v1",
        model="fixture",
        prompt_version="v1",
    )
    config: CacheConfig = CacheConfig(backend="memory", similarity_threshold=0.9)
    cache: LongCache = LongCache(config, embedder=FixtureEmbedder(), context=ctx)

    # Positive assert_type checks required by v0.5.0 Section 8.1
    with cache:
        assert_type(cache.lookup("q", context=ctx), LookupDecision)
        assert_type(cache.set("q", {"ok": True}, context=ctx), StoreDecision)
        assert_type(cache.get_or_compute("q", lambda: "answer", context=ctx), str)
        assert_type(await cache.aget_or_compute("q", async_fn, context=ctx), str)
        assert_type(cache.inspect("q", context=ctx), InspectionReport)
        assert_type(cache.stats(operator=True), StatsSnapshot)
        assert_type(cache.metrics(operator=True), MetricsSnapshot)

    # Test decorator typing and cache_context_for protocol
    @cache.cached(context=ctx)
    def ask(q: str) -> dict[str, int]:
        return {"len": len(q)}

    assert_type(ask("Alice"), dict[str, int])
    assert_type(ask.cache_context_for(ctx), RequestContext | None)

    @cache.cached
    def add(query: str) -> int:
        return len(query)

    assert_type(add("Alice"), int)
"""

NEGATIVE_CONSUMER_CODE = """\
from typing import Any
from longcache import CacheConfig, LongCache, PolicyDecision, RequestContext

# 1. Invalid Literal for scope_policy
bad_config = CacheConfig(scope_policy="invalid")

# 2. Async hook passed to synchronous AdmissionHook protocol
async def bad_admit(context: RequestContext, response: Any) -> PolicyDecision:
    return PolicyDecision("accept", "ok")

cache = LongCache(admit=bad_admit)

# 3. Mismatched return type from get_or_compute
wrong_type: int = cache.get_or_compute("q", lambda: "str")
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=None)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    # Verify py.typed is packaged in the wheel
    wheel = args.wheel
    if wheel is None:
        wheels = sorted(args.dist_dir.glob("*.whl"))
        if wheels:
            wheel = wheels[-1]

    if wheel is not None:
        import zipfile

        with zipfile.ZipFile(wheel) as archive:
            has_py_typed = any(name.endswith("longcache/py.typed") for name in archive.namelist())
            if not has_py_typed:
                print(f"FAILED: py.typed marker missing from wheel {wheel.name}", file=sys.stderr)
                return 1
            print(f"PASS: py.typed present in {wheel.name}")

    # Verify py.typed in importable longcache package
    longcache_dir = Path(longcache.__file__).resolve().parent
    py_typed_path = longcache_dir / "py.typed"
    if not py_typed_path.is_file():
        print(f"FAILED: py.typed not found at {py_typed_path}", file=sys.stderr)
        return 1
    print(f"PASS: py.typed present at {py_typed_path}")

    # Type-check consumer code against the installed package with mypy --strict
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_consumer.py"
        test_file.write_text(CONSUMER_CODE, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", str(test_file)],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print("FAILED: mypy --strict reported typing issues on consumer code", file=sys.stderr)
            return result.returncode

        print("PASS: mypy --strict passed on positive consumer code")

        neg_file = Path(tmpdir) / "test_negative_consumer.py"
        neg_file.write_text(NEGATIVE_CONSUMER_CODE, encoding="utf-8")
        neg_result = subprocess.run(
            [sys.executable, "-m", "mypy", "--strict", str(neg_file)],
            capture_output=True,
            text=True,
        )
        if neg_result.returncode == 0:
            print(
                "FAILED: mypy --strict unexpectedly succeeded on negative type-rejection fixture",
                file=sys.stderr,
            )
            return 1
        for expected_code in ("[arg-type]", "[assignment]"):
            if expected_code not in neg_result.stdout:
                print(
                    f"FAILED: expected {expected_code} in negative mypy output:\n{neg_result.stdout}",
                    file=sys.stderr,
                )
                return 1
        print("PASS: mypy --strict rejected invalid consumer types ([arg-type], [assignment])")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
