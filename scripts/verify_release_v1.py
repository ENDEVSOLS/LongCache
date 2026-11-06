"""Verification script for LongCache v1.0 release readiness and artifact provenance."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import tomllib
import zipfile

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from scripts.check_package_metadata import check_artifact
except ModuleNotFoundError:
    from check_package_metadata import check_artifact


def _source_hash(root: Path) -> str:
    digest = sha256()
    for path in sorted((root / "src" / "longcache").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_release(
    project_file: Path,
    changelog_file: Path,
    dist_dir: Path | None,
    check_evidence: bool,
    root: Path,
) -> list[str]:
    errors: list[str] = []

    # 1. Version and Classifier Synchronization
    if not project_file.exists():
        return [f"Project file not found: {project_file}"]

    project_data = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
    proj_version = project_data["version"]
    classifiers = set(project_data.get("classifiers", []))

    init_file = root / "src" / "longcache" / "__init__.py"
    pkg_version = None
    if init_file.exists():
        match = re.search(
            r'^__version__\s*=\s*["\']([^"\']+)["\']',
            init_file.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            pkg_version = match.group(1)

    if not pkg_version:
        errors.append(f"Could not parse __version__ from {init_file}")
    elif pkg_version != proj_version:
        errors.append(
            f"Version mismatch: pyproject.toml has {proj_version!r}, "
            f"longcache.__init__.__version__ has {pkg_version!r}"
        )

    # In 1.0.0 release line, require Production/Stable and Typing :: Typed
    if proj_version.startswith("1."):
        if "Development Status :: 5 - Production/Stable" not in classifiers:
            errors.append("pyproject.toml missing classifier 'Development Status :: 5 - Production/Stable'")
        if "Development Status :: 4 - Beta" in classifiers:
            errors.append("pyproject.toml still contains classifier 'Development Status :: 4 - Beta'")
        if "Typing :: Typed" not in classifiers:
            errors.append("pyproject.toml missing classifier 'Typing :: Typed'")

    # 2. API Contract Version & Stability Verification
    types_file = root / "src" / "longcache" / "core" / "types.py"
    if types_file.exists():
        types_content = types_file.read_text(encoding="utf-8")
        if 'API_CONTRACT_VERSION: Literal["1.0"] = "1.0"' not in types_content:
            errors.append("src/longcache/core/types.py missing API_CONTRACT_VERSION: Literal['1.0'] = '1.0'")
        if '"stable-candidate"' in types_content:
            errors.append("src/longcache/core/types.py still contains 'stable-candidate' tier")

    # 3. Changelog & Docs Verification
    if not changelog_file.exists():
        errors.append(f"Changelog not found: {changelog_file}")
    else:
        content = changelog_file.read_text(encoding="utf-8")
        escaped_v = re.escape(proj_version)
        pattern = rf"##\s+\[{escaped_v}\]"
        if not re.search(pattern, content):
            errors.append(f"CHANGELOG.md missing entry for version ## [{proj_version}]")

    docs_changelog = root / "docs" / "changelog.md"
    if docs_changelog.exists():
        content = docs_changelog.read_text(encoding="utf-8")
        escaped_v = re.escape(proj_version)
        pattern = rf"##\s+\[{escaped_v}\]"
        if not re.search(pattern, content):
            errors.append(f"docs/changelog.md missing entry for version ## [{proj_version}]")

    # Check 1.x documentation files exist
    for doc_name in ("semver-contract.md", "migration-v1.0.md"):
        doc_path = root / "docs" / doc_name
        if not doc_path.exists():
            errors.append(f"Missing required documentation: docs/{doc_name}")

    # 4. Distribution inspection, SBOM & SHA256SUMS
    if dist_dir and dist_dir.exists():
        wheels = sorted(dist_dir.glob("*.whl"))
        sdists = sorted(dist_dir.glob("*.tar.gz"))
        sboms = sorted(dist_dir.glob("*.sbom.json"))

        if wheels or sdists:
            checksums: list[str] = []
            for artifact in wheels + sdists:
                artifact_errors = check_artifact(artifact, project_data)
                errors.extend(f"{artifact.name}: {err}" for err in artifact_errors)

                # Check for dirty leaked artifacts in wheel
                if artifact.suffix == ".whl":
                    with zipfile.ZipFile(artifact) as zf:
                        for name in zf.namelist():
                            if any(
                                bad in name
                                for bad in ("__pycache__", ".pytest_cache", ".longcache", ".db")
                            ):
                                errors.append(f"{artifact.name} leaked test/cache file: {name}")

                digest = sha256(artifact.read_bytes()).hexdigest()
                checksums.append(f"{digest}  {artifact.name}\n")

            # Check SBOM presence and validity
            expected_sbom_name = f"longcache-{proj_version}.sbom.json"
            sbom_file = dist_dir / expected_sbom_name
            if not sbom_file.exists():
                errors.append(f"Missing CycloneDX SBOM artifact at {sbom_file}")
            else:
                try:
                    sbom_data = json.loads(sbom_file.read_text(encoding="utf-8"))
                    if sbom_data.get("bomFormat") != "CycloneDX":
                        errors.append(f"{sbom_file.name}: invalid bomFormat (expected CycloneDX)")
                    digest = sha256(sbom_file.read_bytes()).hexdigest()
                    checksums.append(f"{digest}  {sbom_file.name}\n")
                except Exception as exc:
                    errors.append(f"Failed to parse SBOM {sbom_file}: {exc}")

            # Write or verify SHA256SUMS.txt
            sums_file = dist_dir / "SHA256SUMS.txt"
            sums_file.write_text("".join(checksums), encoding="utf-8")
            print(f"Recorded SHA256SUMS.txt for {len(checksums)} distribution artifacts")

    # 5. Check benchmark evidence
    if check_evidence:
        reports_dir = root / "evaluation" / "reports"
        v1_report = reports_dir / "v1_0_qualification.json"
        qual_report = reports_dir / "v0_6_qualification.json"
        smoke_report = reports_dir / "v0_6_qualification-smoke.json"

        target_report = None
        for rep in (v1_report, qual_report, smoke_report):
            if rep.exists():
                target_report = rep
                break

        if not target_report:
            errors.append(f"Qualification benchmark report not found in {reports_dir}")
        else:
            try:
                rep_data = json.loads(target_report.read_text(encoding="utf-8"))
                current_hash = _source_hash(root)
                report_hash = rep_data.get("candidate_hash")
                if report_hash != current_hash:
                    errors.append(
                        f"Benchmark report candidate_hash ({report_hash}) "
                        f"does not match current source tree hash ({current_hash})"
                    )
                if not rep_data.get("qualification_gates_passed"):
                    errors.append(
                        f"Benchmark report {target_report.name} indicates qualification gates failed"
                    )
            except Exception as exc:
                errors.append(f"Failed to parse benchmark report {target_report}: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--check-evidence", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    errors = verify_release(
        project_file=args.project,
        changelog_file=args.changelog,
        dist_dir=args.dist_dir if args.dist_dir.exists() else None,
        check_evidence=args.check_evidence,
        root=root,
    )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("PASS: Release verification succeeded with zero errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
