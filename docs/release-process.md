# Release Engineering & Rollback Runbook

This runbook outlines the official process for cutting, verifying, and publishing releases of LongCache to PyPI, including release candidate procedures and emergency rollback/yank guidelines.

---

## 1. Release Lifecycle & SemVer Rules

LongCache follows [Semantic Versioning 2.0.0](https://semver.org/):

* **`0.x` Releases (`0.1.0` through `0.6.0`):** Fast-moving stabilization, qualification, and API cleanup. Breaking changes require documentation in `docs/migration-v0.x.md`.
* **`1.0.0` Release:** The first frozen public API and support contract.
* **`1.x.y` Releases:** Backward-compatible additions and patch fixes.
* **Pre-releases (`0.6.0rc1`, `1.0.0rc1`):** Cut with an explicit validation window to allow downstream testing before official tags.

---

## 2. Pre-Release Checklist & Verification

Before cutting any release tag or release candidate:

1. **Clean Working Tree & Locked Dependencies:**
   ```bash
   uv sync --locked --extra all --extra dev --extra docs
   ```
2. **Lint, Format & Strict Typing:**
   ```bash
   make lint
   make typecheck
   ```
3. **Full Test Suite & Redis Stress Checks:**
   ```bash
   export LONGCACHE_TEST_REDIS_URL=redis://localhost:16379/0
   export LONGCACHE_TEST_MODEL=1
   uv run --no-sync pytest tests/ -v --tb=short -rs
   ```
4. **All Documented Examples & Quickstarts:**
   ```bash
   uv run --no-sync python scripts/wheel_examples_v6.py
   ```
5. **Qualification Benchmark & Evidence Hash:**
   ```bash
   uv run --no-sync python -m evaluation.qualification_v6 \
       --baseline-root /tmp/longcache-v03-baseline \
       --smoke
   ```
6. **Documentation Build:**
   ```bash
   uv run --no-sync mkdocs build --strict
   ```

---

## 3. Package Distribution & Integrity Checks

Build and verify the distribution packages:

```bash
rm -rf dist/
uv run --no-sync python -m build
uv run --no-sync twine check --strict dist/*
uv run --no-sync python scripts/check_package_metadata.py --dist-dir dist
uv run --no-sync python scripts/verify_release_v6.py --dist-dir dist --check-evidence
```

This verifies:
* `src/longcache/py.typed` is present in both `.whl` and `.tar.gz`.
* No test cache artifacts (`.pytest_cache`, `__pycache__`, `.longcache`, `*.db`) leaked into the archive.
* Exact SHA256 checksums are generated in `dist/SHA256SUMS.txt`.
* The benchmark evidence hash matches the current source tree.

---

## 4. PyPI Publishing (Trusted Publishing / OIDC)

LongCache uses PyPI's official [Trusted Publishing (OIDC)](https://docs.pypi.org/trusted-publishers/) workflow via `.github/workflows/publish.yml`:

1. No API tokens or passwords are stored in repository secrets.
2. The GitHub Environment `pypi` grants short-lived access tokens directly to GitHub Actions jobs running from release tags.
3. Once a release tag is pushed (or GitHub Release created), `.github/workflows/publish.yml` executes `pypa/gh-action-pypi-publish@release/v1`.

---

## 5. Emergency Rollback & Yank Procedure

PyPI **never allows re-uploading or modifying an existing version filename**. If a critical defect (data corruption, security flaw, or package breakage) is discovered post-release:

### When to Yank a Release
Yank a release if:
* The wheel fails to install or crashes on basic imports.
* A severe privacy, isolation, or data corruption flaw is identified.

*Yanking* marks the release as deprecated on PyPI without breaking existing pinned builds (`requirements.txt` with `longcache==X.Y.Z` will continue to resolve, but bare `pip install longcache` will skip the yanked version).

### Step-by-Step Yank Procedure
1. Navigate to [PyPI Project Management](https://pypi.org/manage/project/longcache/releases/).
2. Select the affected release version.
3. Click **Options** → **Yank release**.
4. Provide an explicit reason (e.g., *"Yanked due to issue #XYZ; upgrade to v0.6.1"*).
5. If the defect involves a shared Redis persistence incompatibility, notify operators to rotate `generation` in `CacheConfig(generation="...")`.
6. Cut and publish an immediate forward patch release (`0.6.1`).

