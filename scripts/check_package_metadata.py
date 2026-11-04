"""Check package distribution metadata, Python requirement, and assets shipped to PyPI."""

from __future__ import annotations

import argparse
from email import policy
from email.parser import BytesParser
from pathlib import Path
import re
import sys
import tarfile
import tomllib
import zipfile


def read_metadata(artifact: Path):
    """Read metadata directly from an archive without extracting its files."""
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError("expected one wheel METADATA file")
            raw = archive.read(names[0])
    else:
        with tarfile.open(artifact) as archive:
            names = [
                name
                for name in archive.getnames()
                if len(name.split("/")) == 2 and name.endswith("/PKG-INFO")
            ]
            if len(names) != 1:
                raise ValueError("expected one top-level source-distribution PKG-INFO file")
            member = archive.extractfile(names[0])
            if member is None:
                raise ValueError("PKG-INFO is not a regular file")
            raw = member.read()
    return BytesParser(policy=policy.default).parsebytes(raw)


def check_artifact(artifact: Path, project: dict) -> list[str]:
    metadata = read_metadata(artifact)
    errors = []
    for header, expected in (
        ("Name", project["name"]),
        ("Version", project["version"]),
        ("Requires-Python", project["requires-python"]),
        ("Description-Content-Type", "text/markdown"),
    ):
        actual = metadata.get(header)
        if actual != expected:
            errors.append(f"{header}: expected {expected!r}, got {actual!r}")

    description = metadata.get_payload()
    if not isinstance(description, str) or not description.strip():
        return errors + ["missing Markdown project description"]

    # Check for relative image links that would break on PyPI
    relative_images = re.findall(r"!\[([^\]]*)\]\((?!(?:https?://|data:))([^)\s]+)\)", description)
    if relative_images:
        for alt, src in relative_images:
            errors.append(f"Image '{alt}' uses relative URL '{src}' which breaks on PyPI")

    # Check PEP 561 py.typed marker
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            if not any(name.endswith("longcache/py.typed") for name in archive.namelist()):
                errors.append("missing 'longcache/py.typed' marker in wheel")
    elif artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact) as archive:
            if not any(name.endswith("longcache/py.typed") for name in archive.getnames()):
                errors.append("missing 'longcache/py.typed' marker in source distribution")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    try:
        project = tomllib.loads(args.project.read_text(encoding="utf-8"))["project"]
        wheels = sorted(args.dist_dir.glob("*.whl"))
        sdists = sorted(args.dist_dir.glob("*.tar.gz"))
        if not wheels or not sdists:
            raise ValueError(
                f"{args.dist_dir}: both a wheel and a source distribution are required"
            )

        failed = False
        for artifact in wheels + sdists:
            errors = check_artifact(artifact, project)
            for error in errors:
                print(f"ERROR {artifact.name}: {error}", file=sys.stderr)
            if errors:
                failed = True
            else:
                print(f"PASS {artifact.name}: valid distribution metadata and descriptions")
        return int(failed)
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
