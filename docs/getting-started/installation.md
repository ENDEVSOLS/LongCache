# Installation

LongCache v1.0 requires Python 3.11 or newer; Python 3.11 and 3.12 are the supported and tested interpreters. v0.1.1 remains the legacy Python 3.10-compatible line.

v1.0.0 is the Stable Foundation release (`Development Status :: 5 - Production/Stable`). To install from a local repository checkout:

```bash
uv sync --locked --extra all --extra dev --extra docs
```

To install the published package:

```bash
uv pip install longcache
uv pip install 'longcache[openai,server,redis]'
```

The local model downloads weights on first use. Pin `embedder_revision` for a deployment, then cache the weights for offline operation. Custom embedders still use the current core package dependencies. See [upgrading to v1.0](../migration-v1.0.md), [1.x SemVer contract](../semver-contract.md), and the [support matrix](../support-matrix.md) before upgrading. Compatible Format-2 stores from v0.2–v0.6 require no reset. For v0.1.1 stores, follow the separate [legacy migration](../migration-v0.2.md); those stores require a fresh path/generation.

For the tested CPU runtime, install `torch==2.14.0+cpu` from `https://download.pytorch.org/whl/cpu` before installing the wheel. Installer index settings in the project's lockfile do not become wheel dependency metadata.
