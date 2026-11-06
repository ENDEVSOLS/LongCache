"""Generate a deterministic CycloneDX v1.5 JSON SBOM for LongCache distribution artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
import tomllib
from uuid import uuid5, NAMESPACE_DNS


def generate_sbom(
    project_file: Path,
    lock_file: Path,
    dist_dir: Path | None,
    output_file: Path | None = None,
) -> Path:
    if not project_file.is_file():
        raise FileNotFoundError(f"Project file not found: {project_file}")

    project_data = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
    proj_name = project_data["name"]
    proj_version = project_data["version"]
    description = project_data.get("description", "")

    # Direct dependencies
    direct_reqs = set()
    for dep in project_data.get("dependencies", []):
        match = re.match(r"^([a-zA-Z0-9_\-]+)", dep)
        if match:
            direct_reqs.add(match.group(1).lower().replace("_", "-"))
    optional_reqs = set()
    for extra, deps in project_data.get("optional-dependencies", {}).items():
        for dep in deps:
            match = re.match(r"^([a-zA-Z0-9_\-]+)", dep)
            if match:
                optional_reqs.add(match.group(1).lower().replace("_", "-"))

    # Parse uv.lock packages
    components = []
    if lock_file.is_file():
        lock_data = tomllib.loads(lock_file.read_text(encoding="utf-8"))
        packages = lock_data.get("package", [])
        for pkg in packages:
            p_name = pkg["name"]
            p_norm = p_name.lower().replace("_", "-")
            if p_norm == proj_name.lower().replace("_", "-"):
                continue
            p_version = pkg["version"]
            purl = f"pkg:pypi/{p_name}@{p_version}"

            scope = "required" if p_norm in direct_reqs else ("optional" if p_norm in optional_reqs else "transitive")

            comp: dict = {
                "type": "library",
                "bom-ref": purl,
                "name": p_name,
                "version": p_version,
                "purl": purl,
                "scope": scope,
            }

            # Hashes from wheel or sdist if available
            hashes = []
            sdist = pkg.get("sdist")
            if sdist and "hash" in sdist:
                h_val = sdist["hash"]
                if ":" in h_val:
                    alg, val = h_val.split(":", 1)
                    alg_name = "SHA-256" if alg == "sha256" else alg.upper()
                    hashes.append({"alg": alg_name, "content": val})
            if not hashes and pkg.get("wheels"):
                w = pkg["wheels"][0]
                if "hash" in w and ":" in w["hash"]:
                    alg, val = w["hash"].split(":", 1)
                    alg_name = "SHA-256" if alg == "sha256" else alg.upper()
                    hashes.append({"alg": alg_name, "content": val})
            if hashes:
                comp["hashes"] = hashes

            components.append(comp)

    # Sort components deterministically by name
    components.sort(key=lambda c: (c["name"], c["version"]))

    # Distribution artifacts
    root_hashes = []
    if dist_dir and dist_dir.is_dir():
        for artifact in sorted(dist_dir.glob("*.whl")) + sorted(dist_dir.glob("*.tar.gz")):
            digest = sha256(artifact.read_bytes()).hexdigest()
            root_hashes.append({
                "alg": "SHA-256",
                "content": digest,
                "comment": artifact.name,
            })

    # Deterministic serial number from name + version
    serial_uuid = uuid5(NAMESPACE_DNS, f"{proj_name}-{proj_version}")
    serial_number = f"urn:uuid:{serial_uuid}"

    timestamp = os.environ.get("SOURCE_DATE_EPOCH", "2026-11-06T00:00:00Z")
    if timestamp.isdigit():
        import datetime
        dt = datetime.datetime.fromtimestamp(int(timestamp), tz=datetime.timezone.utc)
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    sbom = {
        "$schema": "http://cyclonedx.org/schema/bom-1.5.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": serial_number,
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "ENDEVSOLS",
                    "name": "longcache-sbom-generator",
                    "version": "1.0.0",
                }
            ],
            "component": {
                "type": "library",
                "bom-ref": f"pkg:pypi/{proj_name}@{proj_version}",
                "name": proj_name,
                "version": proj_version,
                "description": description,
                "purl": f"pkg:pypi/{proj_name}@{proj_version}",
                "hashes": root_hashes,
            },
        },
        "components": components,
    }

    if output_file is None:
        target_dir = dist_dir if dist_dir and dist_dir.is_dir() else project_file.parent / "dist"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_file = target_dir / f"{proj_name}-{proj_version}.sbom.json"

    output_file.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Generated CycloneDX 1.5 SBOM: {output_file} ({len(components)} components)")
    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        generate_sbom(
            project_file=args.project,
            lock_file=args.lock,
            dist_dir=args.dist_dir if args.dist_dir.exists() else None,
            output_file=args.output,
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
