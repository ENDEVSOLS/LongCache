"""Verify all documented examples and quickstart workflows against the environment."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import httpx


def _run_script(script_path: Path, env: dict[str, str]) -> None:
    print(f"Running {script_path.name}...")
    res = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        print(f"FAILED: {script_path.name}")
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        sys.exit(res.returncode)
    print(f"PASS: {script_path.name}")


async def _verify_fastapi_middleware_example(root: Path) -> None:
    print("Running examples/fastapi_middleware.py with httpx...")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from examples.fastapi_middleware import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "demo",
            "messages": [{"role": "user", "content": "What is the return policy?"}],
        }
        # First call: cache MISS
        resp1 = await client.post("/v1/chat/completions", json=payload)
        assert resp1.status_code == 200, f"Unexpected status: {resp1.status_code}"
        assert resp1.headers.get("x-longcache") == "MISS", (
            f"Expected MISS, got {resp1.headers.get('x-longcache')}"
        )
        data1 = resp1.json()
        assert "choices" in data1
        assert data1["choices"][0]["message"]["content"] == "This is the public demo answer."

        # Second call: cache HIT
        resp2 = await client.post("/v1/chat/completions", json=payload)
        assert resp2.status_code == 200
        assert resp2.headers.get("x-longcache") == "HIT", (
            f"Expected HIT, got {resp2.headers.get('x-longcache')}"
        )
        data2 = resp2.json()
        assert data2 == data1

    print("PASS: examples/fastapi_middleware.py")


def _verify_readme_quickstart() -> None:
    print("Verifying README quickstart workflow...")
    from longcache import CacheConfig, LongCache, RequestContext, RequestScope

    context = RequestContext(
        scope=RequestScope("quickstart-tenant", "quickstart-user"),
        operation="quickstart.answer",
        provider="fixture",
        provider_scope="quickstart-v1",
        model="demo-model",
        prompt_version="v1",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "quickstart.db")
        config = CacheConfig(backend="sqlite", db_path=db_path)
        with LongCache(config, context=context) as cache:
            # 1. Manual get/set
            decision = cache.set("How do I return an item?", "Returns accepted within 30 days.")
            assert decision.outcome == "STORED"
            hit = cache.lookup("How do I return an item?")
            assert hit.outcome == "HIT"
            assert hit.response == "Returns accepted within 30 days."

            # 2. get_or_compute
            calls = 0

            def generate_answer() -> str:
                nonlocal calls
                calls += 1
                return "Fresh generated answer"

            ans1 = cache.get_or_compute("What are shipping fees?", generate_answer)
            assert ans1 == "Fresh generated answer"
            assert calls == 1

            ans2 = cache.get_or_compute("What are shipping fees?", generate_answer)
            assert ans2 == "Fresh generated answer"
            assert calls == 1  # Hit without calling origin!

            # 3. cached decorator
            @cache.cached
            def ask_assistant(question: str) -> dict[str, str]:
                return {"result": f"Answer for: {question}"}

            r1 = ask_assistant("What payment methods are accepted?")
            assert r1 == {"result": "Answer for: What payment methods are accepted?"}
            r2 = ask_assistant("What payment methods are accepted?")
            assert r2 == r1

    print("PASS: README quickstart workflow")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    examples_dir = root / "examples"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    if not env.get("LONGCACHE_TEST_REDIS_URL"):
        env["LONGCACHE_TEST_REDIS_URL"] = "redis://localhost:16379/0"
    if not env.get("REDIS_URL"):
        env["REDIS_URL"] = env["LONGCACHE_TEST_REDIS_URL"]

    standalone_scripts = [
        examples_dir / "safe_reuse.py",
        examples_dir / "reliability.py",
        examples_dir / "versioned_rag.py",
        examples_dir / "multi_instance_rag.py",
        examples_dir / "redis_production.py",
        examples_dir / "basic_langchain.py",
    ]

    for script in standalone_scripts:
        _run_script(script, env)

    asyncio.run(_verify_fastapi_middleware_example(root))
    _verify_readme_quickstart()
    print("ALL 7 EXAMPLES AND QUICKSTART WORKFLOWS PASSED!")


if __name__ == "__main__":
    main()
