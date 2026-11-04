"""Verify consumer typing against the installed LongCache wheel using mypy --strict."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import longcache

CONSUMER_CODE = """\
from typing import Any
import numpy as np
import longcache
from longcache import (
    BaseEmbedder,
    CacheConfig,
    EmbeddingDescriptor,
    Instruction,
    LongCache,
    LookupDecision,
    PolicyDecision,
    RequestContext,
    RequestScope,
    StoreDecision,
)


class FixtureEmbedder(BaseEmbedder):
    dimension: int = 2
    descriptor: EmbeddingDescriptor = EmbeddingDescriptor("fixture", "v1", 2)

    def embed(self, text: str) -> np.ndarray[Any, Any]:
        return np.array([1.0, 0.0], dtype=np.float32)


def main() -> None:
    context: RequestContext = RequestContext(
        scope=RequestScope("consumer", "user-1"),
        operation="test.run",
        provider="fixture",
        provider_scope="v1",
        model="fixture",
        prompt_version="v1",
    )
    config: CacheConfig = CacheConfig(backend="memory", similarity_threshold=0.9)
    cache: LongCache = LongCache(config, embedder=FixtureEmbedder(), context=context)

    # Test lookup and set decisions
    with cache:
        store_res: StoreDecision = cache.set("hello", {"reply": "world"})
        assert store_res.outcome in ("STORED", "REJECTED")

        lookup_res: LookupDecision = cache.lookup("hello")
        if lookup_res.outcome == "HIT":
            resp: Any = lookup_res.response

    # Test decorator typing
    @cache.cached(namespace="decorated")
    def greet(query: str) -> str:
        return f"Greetings, {query}"

    val: str = greet("Alice")

    @cache.cached
    def add(query: str) -> int:
        return len(query)

    num: int = add("Alice")
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

        print("PASS: mypy --strict passed on consumer code")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
