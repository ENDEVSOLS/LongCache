"""Scoped cache inspection and explicit local operator maintenance."""

from dataclasses import asdict
import json
from pathlib import Path
import click
from longcache import CacheConfig, Instruction, LongCache, RequestContext, RequestScope, __version__


def _get_cache(config=None, *, metadata_only=False):
    return LongCache(config or CacheConfig(), metadata_only=metadata_only)


def _context(path):
    if path is None:
        return None
    try:
        data = json.loads(Path(path).read_text())
        scope = data.get("scope")
        data["scope"] = RequestScope(**scope) if scope is not None else None
        data["instructions"] = tuple(Instruction(**item) for item in data.get("instructions", []))
        return RequestContext(**data)
    except (ValueError, TypeError, OSError) as exc:
        raise click.ClickException("Invalid context file") from exc


def common(fn):
    for option in [
        click.option("--db-path", default=".longcache/v2/cache.db", show_default=True),
        click.option("--namespace", default="default"),
        click.option(
            "--scope-policy",
            type=click.Choice(["private", "tenant_shared", "public"]),
            default="private",
        ),
        click.option("--context-file", type=click.Path(exists=True, dir_okay=False)),
        click.option(
            "--operator",
            is_flag=True,
            help="Explicit trusted access to every partition in this store.",
        ),
    ]:
        fn = option(fn)
    return fn


def _open(db_path, namespace, scope_policy, metadata_only):
    return _get_cache(
        CacheConfig(db_path=db_path, namespace=namespace, scope_policy=scope_policy),
        metadata_only=metadata_only,
    )


@click.group()
@click.version_option(version=__version__, prog_name="longcache")
def main():
    """Inspect scoped cache decisions or explicitly maintain an owned v0.2 store."""


@main.command()
@common
def stats(db_path, namespace, scope_policy, context_file, operator):
    """Show counts for the declared scope; process hit/miss counters reset on restart."""
    with _open(db_path, namespace, scope_policy, True) as cache:
        result = cache.stats(context=_context(context_file), operator=operator)
        click.echo("LongCache Statistics (format 2)")
        click.echo(json.dumps(result, indent=2))


@main.command()
@common
@click.option("--query", required=True)
def inspect(db_path, namespace, scope_policy, context_file, operator, query):
    """Explain a lookup without recording a hit. Context is always required."""
    if operator:
        raise click.UsageError(
            "Inspect a specific context; operator mode cannot select a user's identity"
        )
    # A missing context can be explained without loading a model.
    ctx = _context(context_file)
    if ctx is None:
        click.echo(
            json.dumps({"outcome": "BYPASS", "reason": "missing_identity", "format_version": 2})
        )
        return
    with _open(db_path, namespace, scope_policy, False) as cache:
        click.echo(json.dumps(cache.inspect(query, context=ctx), indent=2))


@main.command()
@common
@click.option("--all", "all_partitions", is_flag=True)
@click.option("--yes", is_flag=True, help="Confirm deletion of the selected owned entries.")
def clear(db_path, namespace, scope_policy, context_file, operator, all_partitions, yes):
    """Delete one exact context, or explicitly delete every owned partition."""
    ctx = _context(context_file)
    if all_partitions and not operator:
        raise click.UsageError("--all requires --operator")
    if operator and not all_partitions:
        raise click.UsageError("Operator deletion requires explicit --all")
    if ctx is None and not all_partitions:
        raise click.UsageError("Scoped deletion requires --context-file")
    if not yes:
        click.confirm("Delete the selected owned cache entries?", abort=True)
    with _open(db_path, namespace, scope_policy, True) as cache:
        count = cache.clear(operator=True) if all_partitions else cache.invalidate(context=ctx)
        click.echo(f"{count} entries deleted")


@main.command()
@common
@click.option("--format", "output_format", type=click.Choice(["json"]), default="json")
@click.option("--output", "-o", type=click.Path(dir_okay=False))
def export(db_path, namespace, scope_policy, context_file, operator, output_format, output):
    """Export raw owned payloads only for an explicit context or trusted operator."""
    ctx = _context(context_file)
    if ctx is None and not operator:
        raise click.UsageError("Export requires --context-file or --operator")
    with _open(db_path, namespace, scope_policy, True) as cache:
        records = []
        for entry in cache.entries(context=ctx, operator=operator):
            record = asdict(entry)
            record.pop("vector", None)
            records.append(record)
        encoded = json.dumps(records, indent=2, allow_nan=False)
    if output:
        Path(output).write_text(encoded + "\n")
    else:
        click.echo(encoded)
