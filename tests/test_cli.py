"""CLI maintenance requires explicit scope or local operator access."""

import json
from dataclasses import replace
from unittest.mock import patch
import pytest
from click.testing import CliRunner
from longcache import CacheConfig, LongCache, RequestScope, __version__
from longcache.cli import main
from tests.helpers import HashEmbedder, trusted_context


@pytest.fixture
def populated(tmp_path):
    path = tmp_path / "cache.db"
    alice = trusted_context()
    bob = replace(alice, scope=RequestScope("test-tenant", "other-user"))
    with LongCache(CacheConfig(db_path=str(path), ttl_seconds=0), embedder=HashEmbedder()) as cache:
        cache.set("refund", "alice answer", context=alice)
        cache.set("refund", "bob answer", context=bob)
    context_file = tmp_path / "context.json"
    context_file.write_text(
        json.dumps(
            {
                "scope": {"tenant_id": "test-tenant", "user_id": "test-user"},
                "operation": "test.answer",
                "provider": "fixture",
                "provider_scope": "fixture-v1",
                "model": "fixture-model",
                "prompt_version": "v1",
            }
        )
    )
    return path, context_file


def test_scoped_and_operator_export(populated):
    path, ctx = populated
    runner = CliRunner()
    with patch(
        "longcache.core.embedder.LocalEmbedder._load_model",
        side_effect=AssertionError("Metadata command loaded model"),
    ):
        scoped = runner.invoke(main, ["export", "--db-path", str(path), "--context-file", str(ctx)])
        assert scoped.exit_code == 0, scoped.output
        assert [e["response"] for e in json.loads(scoped.output)] == ["alice answer"]
        both = runner.invoke(main, ["export", "--db-path", str(path), "--operator"])
        assert len(json.loads(both.output)) == 2
        denied = runner.invoke(main, ["export", "--db-path", str(path)])
        assert denied.exit_code != 0


def test_scoped_clear_preserves_other_user(populated):
    path, ctx = populated
    runner = CliRunner()
    denied = runner.invoke(main, ["clear", "--db-path", str(path), "--all", "--yes"])
    assert denied.exit_code != 0
    result = runner.invoke(
        main, ["clear", "--db-path", str(path), "--context-file", str(ctx), "--yes"]
    )
    assert result.exit_code == 0, result.output
    assert "1 entries deleted" in result.output
    exported = runner.invoke(main, ["export", "--db-path", str(path), "--operator"])
    assert [e["response"] for e in json.loads(exported.output)] == ["bob answer"]
    result = runner.invoke(main, ["clear", "--db-path", str(path), "--all", "--operator", "--yes"])
    assert result.exit_code == 0


def test_stats_and_missing_identity_do_not_load_model(populated):
    path, ctx = populated
    runner = CliRunner()
    with patch(
        "longcache.core.embedder.LocalEmbedder._load_model",
        side_effect=AssertionError("Loaded model"),
    ):
        result = runner.invoke(main, ["stats", "--db-path", str(path), "--context-file", str(ctx)])
        assert result.exit_code == 0, result.output
        assert '"total_entries": 1' in result.output
        result = runner.invoke(main, ["inspect", "--db-path", str(path), "--query", "refund"])
        assert result.exit_code == 0
        assert json.loads(result.output)["reason"] == "missing_identity"


def test_inspect_is_redacted_and_does_not_record_hit(populated):
    path, ctx = populated

    def make(config, metadata_only=False):
        return LongCache(config, embedder=HashEmbedder(), metadata_only=metadata_only)

    with patch("longcache.cli._get_cache", side_effect=make):
        result = CliRunner().invoke(
            main,
            ["inspect", "--db-path", str(path), "--context-file", str(ctx), "--query", "refund"],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["outcome"] == "HIT"
    assert "alice answer" not in result.output
    with LongCache(CacheConfig(db_path=str(path)), metadata_only=True) as cache:
        assert all(e.hit_count == 0 for e in cache.entries(operator=True))


def test_cli_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output
